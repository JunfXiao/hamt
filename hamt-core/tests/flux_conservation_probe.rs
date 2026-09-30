#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Instrumentation probe to LOCALIZE the interior surface flux-vs-stored-water
//! offset (see `doc/flux_offset_investigation.md`).
//!
//! Single 0.02 m EN-15026 wall, sealed exterior, **single-step `solve_timestep()`**
//! (no adaptive substepper → removes the substep accumulator as a variable). After
//! a gentle interior RH step (absorption) we measure, summed over the run:
//!
//!   • Δwater     — true stored-water change (`SurfaceReport::water_total`).
//!   • flux_diag  — Σ `moisture_flux(Interior)`·A·dt  (the reported diagnostic;
//!                  backward-Euler endpoint flux = the step's flux integral).
//!   • flux_recon — Σ (vp_virtual − vp_cellN1)/R_vapor(cellN1,virtual)·dt
//!                  (the solver's actual cell-N1 vapour inflow, reconstructed
//!                  from converged cell state).
//!
//! Localization:
//!   flux_diag ≈ flux_recon ≠ Δwater  ⇒ solver transports less than it stores.
//!   flux_recon ≈ Δwater ≠ flux_diag  ⇒ the diagnostic query mis-measures.
//!
//! H3 test: rerun with `latent_heat_enabled = false` (l_eff → 0 on the
//! BlockTridiagonal path, cross-coupling vanishes, ~isothermal). If the offset
//! collapses, the gap is the T↔φ / vp_sat(T) latent coupling.

mod common;

use anyhow::Result;
use common::{CONSTRUCTION_NAME, create_library, init_tracing};

use hamt_core::boundary::{BoundaryCondition, celsius_to_kelvin};
use hamt_core::cell::{BoundaryLocation, CellType};
use hamt_core::hamt_manager::{HamtManager, HamtSettings, SolverMethod};
use hamt_core::surface::Surface;
use hamt_core::uom::units::{
    dimensionless, hours, kelvin, kg_per_m2_s, kg_per_m2_s_pa, kg_per_s, kilograms, pa_s_per_kg,
    seconds, square_meter, w_per_m2_k,
};

struct ProbeOut {
    n_material: usize,
    dwater_kg: f64,
    flux_diag_kg: f64,
    flux_recon_int_kg: f64,
    flux_recon_ext_kg: f64,
    t_swing_k: f64,
}

/// Reconstruct the solver's surface vapour inflow [kg/s] (env→wall, +into wall)
/// from converged cell state: for each Virtual cell on `location`, take its
/// material neighbour and the vapour resistance between them,
/// flux = (vp_virtual − vp_material)/R_vapor.
fn reconstruct_surface_vapor_flux(
    surface: &Surface,
    bc: &BoundaryCondition,
    location: BoundaryLocation,
) -> f64 {
    let mut total = 0.0; // kg/s
    for vidx in surface
        .virtual_cells(location.clone())
        .copied()
        .collect::<Vec<_>>()
    {
        let virtual_cell = match surface.cells.get(vidx) {
            Some(c) => c,
            None => continue,
        };
        let midx = match surface.neighbor_toward_center(vidx, &location) {
            Some(m) => m,
            None => continue,
        };
        let material = match surface.cells.get(midx) {
            Some(c) => c,
            None => continue,
        };
        if !matches!(material.cell_type, CellType::Material(_)) {
            continue;
        }
        let r = Surface::calculate_resistance_triple(material, virtual_cell, bc.pressure_air, 0);
        if r.vapor > 1e-12 * pa_s_per_kg {
            let dvp = virtual_cell.cell_vp - material.cell_vp;
            // strip: external diagnostic — raw kg/s from typed flux (Pa / (Pa·s/kg)) for summation.
            total += (dvp / r.vapor).value_in(kg_per_s);
        }
    }
    total
}

fn run_probe(
    num_div: usize,
    latent: bool,
    min_relax: f64,
    dt_h: f64,
    run_hours: f64,
    label: &str,
) -> Result<ProbeOut> {
    init_tracing();

    let library = create_library(20.0, 0.5, vec![0.02], vec![Some(num_div)]);
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = dt_h * hours;
    settings.solver_method = SolverMethod::BlockTridiagonal;
    settings.coupled_heat_moisture_enabled = true;
    settings.latent_heat_enabled = latent;
    settings.temp_convergence_tolerance = 1e-4 * kelvin;
    settings.rh_convergence_tolerance = 1e-5 * dimensionless;
    settings.max_iterations = 3000;
    settings.n_div = 1;
    settings.min_relaxation_factor = min_relax;

    let bc_init = BoundaryCondition {
        heat_transfer_coeff_int: 8.0 * w_per_m2_k,
        heat_transfer_coeff_conv_int: 8.0 * w_per_m2_k,
        vapor_transport_coeff_int: 1e-8 * kg_per_m2_s_pa,
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * kg_per_m2_s_pa,
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        rh_outside_air: 0.5 * dimensionless,
        rh_inside_air: 0.5 * dimensionless,
        ..Default::default()
    };

    let mut mgr = HamtManager::new(
        Some(settings),
        Some(bc_init),
        construction,
        "FluxProbe".to_string(),
        1.0 * square_meter,
    )?;
    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * kg_per_m2_s_pa;

    let n_material = mgr
        .surface_curr
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)))
        .count();
    let area_m2 = mgr.surface_curr.area.value_in(square_meter); // strip: scalar area
    let dt_s = (dt_h * hours).value_in(seconds);

    // Equilibrate (benign — wall already at w(0.5), BC at 0.5).
    for _ in 0..((6.0 / dt_h).round() as usize) {
        mgr.solve_timestep()?;
        mgr.tick_timestep()?;
    }

    let water_start = mgr.surface_curr.report(&mgr.boundary_condition)?.water_total;
    // strip: scalar surface temperature for the swing diagnostic.
    let t0 = mgr
        .surface_curr
        .report(&mgr.boundary_condition)?
        .temp_surf_int
        .value_in(kelvin);
    let mut t_min = t0;
    let mut t_max = t0;

    // STEP: humidify the room air → absorption transient.
    mgr.boundary_condition.rh_inside_air = 0.65 * dimensionless;

    let mut flux_diag_kg = 0.0;
    let mut flux_recon_int_kg = 0.0;
    let mut flux_recon_ext_kg = 0.0;
    let n_steps = (run_hours / dt_h).round() as usize;
    for _ in 0..n_steps {
        mgr.solve_timestep()?;
        mgr.tick_timestep()?;

        // Diagnostic surface flux (env→surface, +into wall) × area × dt.
        // strip: scalar flux density for the diagnostic integral.
        let m_int = mgr
            .surface_curr
            .moisture_flux(BoundaryLocation::Interior, &mgr.boundary_condition)?
            .value_in(kg_per_m2_s);
        flux_diag_kg += m_int * area_m2 * dt_s;

        // Reconstructed solver surface vapour inflow (total kg/s) × dt — both faces.
        flux_recon_int_kg += reconstruct_surface_vapor_flux(
            &mgr.surface_curr,
            &mgr.boundary_condition,
            BoundaryLocation::Interior,
        ) * dt_s;
        flux_recon_ext_kg += reconstruct_surface_vapor_flux(
            &mgr.surface_curr,
            &mgr.boundary_condition,
            BoundaryLocation::Exterior,
        ) * dt_s;

        let ts = mgr
            .surface_curr
            .report(&mgr.boundary_condition)?
            .temp_surf_int
            .value_in(kelvin);
        t_min = t_min.min(ts);
        t_max = t_max.max(ts);
    }

    let water_end = mgr.surface_curr.report(&mgr.boundary_condition)?.water_total;
    // strip: scalar masses for the conservation comparison.
    let dwater_kg = (water_end - water_start).value_in(kilograms);

    let net_boundary_kg = flux_recon_int_kg + flux_recon_ext_kg;
    let off = |x: f64| (x - dwater_kg) / dwater_kg * 100.0;
    println!(
        "  [{label:<14}] N={n_material:>3} relax≥{min_relax:.2} | Δwater={dwater_kg:>10.4e} | \
         diag_int={flux_diag_kg:>10.4e}({:>+5.1}%) | recon_int={flux_recon_int_kg:>10.4e} | \
         recon_ext={flux_recon_ext_kg:>10.4e} | net(int+ext)={net_boundary_kg:>10.4e}({:>+5.1}%) | Tsw={:.3}K",
        off(flux_diag_kg),
        off(net_boundary_kg),
        t_max - t_min,
    );

    Ok(ProbeOut {
        n_material,
        dwater_kg,
        flux_diag_kg,
        flux_recon_int_kg,
        flux_recon_ext_kg,
        t_swing_k: t_max - t_min,
    })
}

#[test]
fn flux_conservation_probe() -> Result<()> {
    println!("=== Flux-vs-stored-water conservation probe (single-step solve, sealed exterior) ===");
    println!("    0.02 m EN-15026 wall, interior RH step 0.5→0.65, dt=0.1 h, 12 h absorption");

    let dt_h = 0.1;
    let run_h = 12.0;

    println!("\n  -- Under-relaxed (min_relax 0.5): full coupling --");
    let r05_12 = run_probe(12, true, 0.5, dt_h, run_h, "relax0.5")?;
    let r05_48 = run_probe(48, true, 0.5, dt_h, run_h, "relax0.5")?;

    println!("\n  -- No relaxation (min_relax 1.0): full coupling [H5 test] --");
    let r10_12 = run_probe(12, true, 1.0, dt_h, run_h, "relax1.0")?;
    let r10_48 = run_probe(48, true, 1.0, dt_h, run_h, "relax1.0")?;

    println!("\n  -- Latent heat OFF (isothermal), no relaxation --");
    let iso10_12 = run_probe(12, false, 1.0, dt_h, run_h, "iso relax1.0")?;
    let iso10_48 = run_probe(48, false, 1.0, dt_h, run_h, "iso relax1.0")?;

    // moisture_flux now reports the TOTAL surface flux (vapour + liquid), so its
    // integral must equal the wall's stored-water change (conservation). The
    // manual vapour-only reconstruction is kept to show the liquid contribution
    // it omits — that gap was the original 13–40 % offset before the fix.
    let off_total = |p: &ProbeOut| (p.flux_diag_kg - p.dwater_kg) / p.dwater_kg * 100.0;
    let off_vap_only = |p: &ProbeOut| {
        (p.flux_recon_int_kg + p.flux_recon_ext_kg - p.dwater_kg) / p.dwater_kg * 100.0
    };

    println!("\n=== Summary (offset = flux−Δwater, % of Δwater) ===");
    let all = [
        ("relax0.5    N=12", &r05_12),
        ("relax0.5    N=48", &r05_48),
        ("relax1.0    N=12", &r10_12),
        ("relax1.0    N=48", &r10_48),
        ("iso relax1.0 N=12", &iso10_12),
        ("iso relax1.0 N=48", &iso10_48),
    ];
    for (tag, p) in all {
        println!(
            "  {tag}: moisture_flux(total) {:>+6.2}%, vapour-only {:>+6.1}%, Tswing {:.3} K",
            off_total(p),
            off_vap_only(p),
            p.t_swing_k
        );
    }

    // Regression guard: the reported moisture_flux integral must conserve mass
    // against the wall's stored-water change. Before the fix this was −13…−40 %
    // (and grew with N); the fix (report vapour + surface liquid) brought it to
    // ≲ 1.4 %, and the implicit surface node + consistent neighbour Jacobian
    // (solver/adi.rs, "Conservation of the linearised system") to < 0.01 % in every
    // variant above. The integral here is the backward-Euler right-Riemann sum
    // (no substepper), i.e. exactly what the implicit update stores, so the only
    // residual is the outer-loop exit tolerance: 0.5 % leaves a 50× margin over
    // what is measured and still fails the pre-fix solver (+1.09 % at N=12).
    for (tag, p) in all {
        assert!(
            off_total(p).abs() < 0.5,
            "{tag}: reported moisture_flux deviates {:.2}% from the wall's stored-water change \
             — surface mass conservation regression (see doc/flux_offset_investigation.md)",
            off_total(p)
        );
    }
    Ok(())
}
