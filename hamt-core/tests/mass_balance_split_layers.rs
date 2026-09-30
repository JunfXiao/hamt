#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//! Moisture MASS CONSERVATION and grid independence of the 1-D solver on the
//! production wall (gypsum lining | PE foil | wood fibre | LivingBoard | cellulose |
//! AGEPAN, n_div = 30 per layer, geometric grading towards every layer face), with
//! the room-side gypsum lining split into `k` IDENTICAL sub-layers of the same total
//! thickness.
//!
//! Physically nothing changes when one homogeneous layer is split into `k` identical
//! ones. Numerically it used to: every sub-layer adds two tiny grading cells, and the
//! room-side surface cell shrinks from 0.114 mm (k=1) to 14 µm (k=8). The old
//! assembly lagged the massless surface (Virtual) node and dropped the neighbour's
//! `P_sat(T)` term from the off-diagonal blocks, so
//!
//!   * the outer iteration contracted by only `ρ ≈ 1 − (h + G_in)/G_1 → 1` per
//!     iteration (thermal ρ = 0.994 at k=1, 0.9993 at k=8; vapour 0.84 → 0.98), the
//!     step-size exit test (|ΔT| < 1e-3 K, |Δφ| < 1e-4) fired long before
//!     convergence and the "converged" state was off by up to `tol/(1−ρ)`;
//!   * every unconverged iterate was non-conservative (interface flux `F_ij + F_ji ≠ 0`,
//!     surface flux in the equation ≠ surface flux reported), i.e. it created or
//!     destroyed water in proportion to `G ∝ 1/Δx`.
//!
//! Measured on this scenario with the old assembly (k = 1 / 2 / 4 / 8), the stored
//! water change over the 48 h humidification window was +455 / +607 / −59 / +471 g
//! against a converged +389 g, the stored change and the ∫reported-flux disagreed
//! by up to 1.5 kg (k = 4, hold window), and the fixed-step solver did not converge
//! at dt = 600 s or 3600 s at all.
//!
//! The scenario mirrors the production evidence (hourly TRNSYS macro steps, the
//! production 14-input boundary condition, `solve_timestep_adaptive` + flux
//! accumulator exactly as `Type395::iterate` drives them):
//!
//!   0 –  24 h  room 20 °C / 50 % RH     (wall in equilibrium → no exchange)
//!   24 –  72 h  RH step 50 → 65 % at 20 °C            (isothermal absorption)
//!   72 –  96 h  T step 20 → 25 °C at CONSTANT vapour pressure (sharp thermal
//!               transient → desorption; the case in which the old solver even
//!               absorbed water while the surface heated up)
//!   96 – 192 h  hold
//!
//! The exterior is sealed (β_out = h_out = 0), so the water stored in ALL layers must
//! equal the time integral of the interior moisture flux reported to TRNSYS.

use anyhow::Result;
use std::sync::Arc;

use hamt_core::boundary::BoundaryCondition;
use hamt_core::cell::BoundaryLocation;
use hamt_core::hamt_manager::{AdaptiveState, HamtManager, HamtSettings};
use hamt_core::io::load_hamt_by_content;
use hamt_core::material::{Construction, Material};
use hamt_core::uom::units::{
    hours, kelvin, kg_per_m2_s, kilograms, pascals, seconds, square_meter,
};

/// The production wall (same IDF the TRNSYS decks use; IDF layers are listed
/// inside→outside and reversed by the loader, so the gypsum is the LAST cell layer,
/// i.e. the one next to the interior boundary).
const GKP_IDF: &str = include_str!("../fixtures/gkp_withfoil_1x25.idf");
const CONSTRUCTION: &str = "20260113_GKP_WithFoil";
const AREA_M2: f64 = 10.0;
const TOTAL_HOURS: usize = 192;

/// (first hour, last hour, label) of the analysis windows.
const WINDOWS: [(usize, usize, &str); 4] = [
    (0, 24, "0-24 h equilibrium"),
    (24, 72, "24-72 h RH step 50->65 %"),
    (72, 96, "72-96 h T step 20->25 C"),
    (96, 192, "96-192 h hold"),
];

/// Wall whose room-side gypsum layer is split into `k` identical sub-layers.
fn build_construction(k: usize) -> Result<Arc<Construction>> {
    let lib = load_hamt_by_content(GKP_IDF)?;
    let base = lib.get_construction(CONSTRUCTION)?;
    let gyp = base.layers.last().expect("layers").clone();
    assert!(
        gyp.name.starts_with("Gipskarton"),
        "last (room-side) layer should be the gypsum board, got {}",
        gyp.name
    );
    let mut layers: Vec<Arc<Material>> = base.layers[..base.layers.len() - 1].to_vec();
    if k == 1 {
        layers.push(gyp);
    } else {
        let mut sub = (*gyp).clone();
        sub.thickness = gyp.thickness / k as f64;
        sub.name = format!("{}_split{k}", gyp.name);
        let sub = Arc::new(sub);
        layers.extend(std::iter::repeat_n(sub, k));
    }
    Ok(Arc::new(Construction {
        name: format!("GKP_split{k}"),
        layers,
        surface_vapor_coef: base.surface_vapor_coef.clone(),
    }))
}

fn psat_pa(t_c: f64) -> f64 {
    hamt_core::physics::psy_psat_fn_temp((t_c + 273.15) * kelvin).value_in(pascals)
}

/// Room climate of hour `h`: (T [°C], RH [0..1]).
fn room_at(h: usize) -> (f64, f64) {
    if h < 24 {
        (20.0, 0.50)
    } else if h < 72 {
        (20.0, 0.65)
    } else {
        // heating at the constant vapour pressure of the 65 % / 20 °C state
        (25.0, 0.65 * psat_pa(20.0) / psat_pa(25.0))
    }
}

/// Production 14-input Type 395 vector (room = interior side, exterior sealed).
/// h_conv / β values are the ones logged by the production runs.
fn type395_inputs(t_c: f64, rh: f64) -> [f64; 14] {
    [
        t_c,        // 0  TSTAR
        15.0,       // 1  T_out (irrelevant: sealed)
        rh * 100.0, // 2  RH_int [%]
        50.0,       // 3  RH_out (irrelevant: sealed)
        63.475242,  // 4  combined h_int [kJ/(h m2 K)]
        0.0,        // 5  h_out = 0  -> adiabatic exterior
        0.0,        // 6  q_int
        0.0,        // 7  q_out
        6.057e-8,   // 8  beta_int [kg/(m2 s Pa)]
        0.0,        // 9  beta_out = 0 -> vapour-tight exterior
        0.0,        // 10 q_userdef
        t_c,        // 11 TAIR
        31.152233,  // 12 HCONVI [kJ/(h m2 K)]
        t_c,        // 13 TMR
    ]
}

struct Hour {
    /// Change of the water stored in ALL material cells over this hour [g].
    d_stored_g: f64,
    /// Interior moisture flux integrated over this hour, env → wall [g].
    flux_g: f64,
    /// Interior surface temperature [°C] and relative humidity [-] at the hour's end.
    t_surf_c: f64,
    rh_surf: f64,
}

fn new_manager(k: usize, n_div: i32) -> Result<HamtManager> {
    let (t0, rh0) = room_at(0);
    let mut settings = HamtSettings {
        timestep: 1.0 * hours,
        n_div,
        ..Default::default()
    };
    // The production DLL overrides exactly this one default (`Type395::init`).
    settings.min_relaxation_factor = 0.1;
    HamtManager::new(
        Some(settings),
        Some(BoundaryCondition::from_trnsys_inputs(&type395_inputs(
            t0, rh0,
        ))),
        build_construction(k)?,
        format!("split{k}"),
        AREA_M2 * square_meter,
    )
}

fn stored_kg(mgr: &HamtManager) -> Result<f64> {
    Ok(mgr
        .surface_curr
        .report(&mgr.boundary_condition)?
        .water_total
        .value_in(kilograms))
}

/// Production driver: one `solve_timestep_adaptive` per hourly macro step, flux taken
/// from the substep accumulator (what TRNSYS receives as Type 395 output 10).
fn run_production_driver(k: usize, n_div: i32) -> Result<Vec<Hour>> {
    let mut mgr = new_manager(k, n_div)?;
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    let area = mgr.surface_curr.area;
    let mut w_prev = stored_kg(&mgr)?;
    let mut out = Vec::with_capacity(TOTAL_HOURS);
    for h in 0..TOTAL_HOURS {
        let (t, rh) = room_at(h);
        mgr.boundary_condition = BoundaryCondition::from_trnsys_inputs(&type395_inputs(t, rh));
        adaptive.current_time = (h as f64) * hours;
        mgr.solve_timestep_adaptive(&mut adaptive, None)?;
        let flux_g = (adaptive.flux_accumulator.m_int_dt * area).value_in(kilograms) * 1000.0;
        let rep = mgr.surface_curr.report(&mgr.boundary_condition)?;
        let w = stored_kg(&mgr)?;
        out.push(Hour {
            d_stored_g: (w - w_prev) * 1000.0,
            flux_g,
            t_surf_c: rep.temp_surf_int.value_in(kelvin) - 273.15,
            rh_surf: rep.rh_surf_int.value(),
        });
        w_prev = w;
        mgr.tick_timestep()?;
    }
    Ok(out)
}

/// Fixed-step backward Euler WITHOUT the adaptive substepper. The flux integral is
/// the right-Riemann sum `Σ flux_end·dt`, which is exactly what backward Euler
/// stores, so this isolates the spatial/linearisation consistency of the solver from
/// the flux accumulator's quadrature.
fn run_fixed_step(k: usize, n_div: i32, dt_s: f64) -> Result<Vec<Hour>> {
    let mut mgr = new_manager(k, n_div)?;
    let n_sub = (3600.0 / dt_s).round() as usize;
    let mut w_prev = stored_kg(&mgr)?;
    let mut out = Vec::with_capacity(TOTAL_HOURS);
    for h in 0..TOTAL_HOURS {
        let (t, rh) = room_at(h);
        mgr.boundary_condition = BoundaryCondition::from_trnsys_inputs(&type395_inputs(t, rh));
        let mut flux_kg = 0.0;
        for _ in 0..n_sub {
            mgr.settings.timestep = dt_s * seconds;
            let converged = mgr.solve_timestep()?;
            assert!(
                converged,
                "backward-Euler solve (k={k}, n_div={n_div}, dt={dt_s} s) did not converge at h={h}"
            );
            let m_int = mgr
                .surface_curr
                .moisture_flux(BoundaryLocation::Interior, &mgr.boundary_condition)?
                .value_in(kg_per_m2_s);
            flux_kg += m_int * AREA_M2 * dt_s;
            mgr.tick_timestep()?;
        }
        let rep = mgr.surface_curr.report(&mgr.boundary_condition)?;
        let w = stored_kg(&mgr)?;
        out.push(Hour {
            d_stored_g: (w - w_prev) * 1000.0,
            flux_g: flux_kg * 1000.0,
            t_surf_c: rep.temp_surf_int.value_in(kelvin) - 273.15,
            rh_surf: rep.rh_surf_int.value(),
        });
        w_prev = w;
    }
    Ok(out)
}

/// (Δstored, ∫flux) in g over hours `[a, b)`.
fn window_sums(run: &[Hour], a: usize, b: usize) -> (f64, f64) {
    (
        run[a..b].iter().map(|x| x.d_stored_g).sum(),
        run[a..b].iter().map(|x| x.flux_g).sum(),
    )
}

/// Print the table and assert per-window conservation:
/// `|Δstored − ∫flux| ≤ rel_tol · max(|Δstored|, |∫flux|) + 0.05 g`.
fn check_conservation(label: &str, run: &[Hour], rel_tol: f64) {
    for (a, b, name) in WINDOWS {
        let (ds, fl) = window_sums(run, a, b);
        let imb = ds - fl;
        let scale = ds.abs().max(fl.abs());
        let pct = if scale > 0.5 {
            format!("{:>+6.2} %", 100.0 * imb / scale)
        } else {
            "   n/a  ".to_string()
        };
        println!(
            "  {label:<26} {name:<26} dStored {ds:>+9.2} g | int.flux {fl:>+9.2} g | imbalance {imb:>+7.2} g ({pct})"
        );
        assert!(
            imb.abs() <= rel_tol * scale + 0.05,
            "{label}, window {name}: moisture mass balance violated — stored water changed by \
             {ds:+.2} g but the reported interior flux integrates to {fl:+.2} g \
             (imbalance {imb:+.2} g = {:.2} % of the exchanged mass, limit {:.2} %)",
            100.0 * imb.abs() / scale.max(1e-12),
            100.0 * rel_tol
        );
    }
}

/// `run` must reproduce the reference run (k = 1, n_div = 30) in every window to
/// `rel_tol` of the window's exchanged mass, and the interior surface state to
/// `t_tol` K / `rh_tol`.
fn check_grid_independence(
    label: &str,
    run: &[Hour],
    reference: &[Hour],
    rel_tol: f64,
    t_tol: f64,
    rh_tol: f64,
) {
    for (a, b, name) in WINDOWS {
        let (ds, _) = window_sums(run, a, b);
        let (ds_ref, _) = window_sums(reference, a, b);
        let scale = ds_ref.abs().max(ds.abs());
        assert!(
            (ds - ds_ref).abs() <= rel_tol * scale + 0.05,
            "{label}, window {name}: stored-water change {ds:+.2} g differs from the k=1/n_div=30 \
             reference {ds_ref:+.2} g by more than {:.1} % — identical layers must give the same wall",
            100.0 * rel_tol
        );
    }
    let mut max_dt = 0.0_f64;
    let mut max_drh = 0.0_f64;
    for (x, y) in run.iter().zip(reference) {
        max_dt = max_dt.max((x.t_surf_c - y.t_surf_c).abs());
        max_drh = max_drh.max((x.rh_surf - y.rh_surf).abs());
    }
    println!("  {label:<26} max |dT_surf| {max_dt:.4} K, max |dRH_surf| {max_drh:.5} vs reference");
    assert!(
        max_dt <= t_tol && max_drh <= rh_tol,
        "{label}: interior surface state depends on the grid (max |dT|={max_dt:.4} K, \
         max |dRH|={max_drh:.5}; limits {t_tol} K / {rh_tol})"
    );
}

/// Production driver (adaptive substepper + flux accumulator), k = 1, 2, 4, 8.
///
/// The accumulator integrates the flux with the END-POINT rule `Σ flux_end·dt`,
/// which is exactly what the backward-Euler update stores in the wall, so the
/// reported interior flux must reproduce the stored-water change to the outer-loop
/// exit tolerance: measured ≤ 0.02 % of the exchanged mass in every window (with the
/// previous composite-trapezoidal accumulator the quadrature alone left 1.2 % /
/// 0.8 % in the two transient windows). The limit is 0.5 %, i.e. > 20× the measured
/// value and 2–3 orders of magnitude tighter than the 60–1000 % violations of the
/// lagged-surface-node assembly. Grid independence is measured at ≤ 0.3 % (window
/// sums) / 0.002 K / 0.0005 RH.
#[test]
fn split_lining_conserves_moisture_production_driver() -> Result<()> {
    println!("=== production driver (adaptive substepper), n_div = 30 ===");
    let reference = run_production_driver(1, 30)?;
    check_conservation("k=1 (1x25 mm)", &reference, 0.005);
    for k in [2usize, 4, 8] {
        let run = run_production_driver(k, 30)?;
        let label = format!("k={k} ({k}x{:.4} m)", 0.025 / k as f64);
        check_conservation(&label, &run, 0.005);
        check_grid_independence(&label, &run, &reference, 0.01, 0.01, 0.002);
    }
    Ok(())
}

/// Fixed-step backward Euler (no substepper, right-Riemann flux): the discrete
/// balance of the linearised solver itself. The remaining imbalance is the
/// difference between the secant storage `ξ(φ^k)·Δφ` and the exact `w(φ^{k+1})−w^n`
/// at the exit tolerance — measured ≤ 0.2 %; the limit is 0.5 %. The old assembly
/// did not even converge at these step sizes (every `solve_timestep` returned
/// `false` and the adaptive substepper collapsed `dt`).
#[test]
fn split_lining_conserves_moisture_fixed_step_backward_euler() -> Result<()> {
    for dt_s in [3600.0, 600.0] {
        println!("=== fixed-step backward Euler, dt = {dt_s} s ===");
        let reference = run_fixed_step(1, 30, dt_s)?;
        check_conservation("k=1 (1x25 mm)", &reference, 0.005);
        let run = run_fixed_step(8, 30, dt_s)?;
        check_conservation("k=8 (8x3.125 mm)", &run, 0.005);
        check_grid_independence("k=8 (8x3.125 mm)", &run, &reference, 0.01, 0.01, 0.002);
    }
    Ok(())
}

/// Refining the room-side surface cell (n_div = 20 … 60 → first cell 0.376 mm …
/// 3.9 µm) must not change the answer. The old scheme's smallest-cell dependence
/// (ρ → 1) showed up here as +405 / +455 / +25 / +260 g for n_div = 20 / 30 / 40 / 60.
#[test]
fn surface_cell_refinement_is_grid_independent() -> Result<()> {
    println!("=== surface-cell refinement, production driver, k = 1 ===");
    let reference = run_production_driver(1, 30)?;
    for n_div in [20, 40, 60] {
        let run = run_production_driver(1, n_div)?;
        let label = format!("k=1, n_div={n_div}");
        check_conservation(&label, &run, 0.005);
        check_grid_independence(&label, &run, &reference, 0.01, 0.01, 0.002);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Exposed exterior under a harsh outdoor climate
// ---------------------------------------------------------------------------

/// Hourly outdoor climate of hour `h`: T swings ≈ −10…+30 °C with a cold day every
/// third day (abrupt day-to-day jumps), RH 30…99.9 % with a saturation spike every
/// 7th hour — the sub-zero (ice-branch `P_sat`), near-saturation (steep isotherm,
/// liquid transport) and rapid-transient regime the room-side scenario never visits.
fn outdoor_at(h: usize) -> (f64, f64) {
    let phase = 2.0 * std::f64::consts::PI * (h as f64 - 15.0) / 24.0;
    let t = 10.0 + 14.0 * phase.cos() + if (h / 24) % 3 == 1 { -14.0 } else { 0.0 };
    let rh =
        (0.68 - 0.32 * phase.cos()).clamp(0.3, 1.0) + if h.is_multiple_of(7) { 0.2 } else { 0.0 };
    (t, rh.clamp(0.3, 0.999))
}

/// Interior SEALED (h = 0, β ≈ 0), exterior exposed with the production
/// coefficients (h_out = 64 kJ/(h m2 K), β_out = 1.244e-7 kg/(m2 s Pa)).
fn exposed_exterior_inputs(t_out: f64, rh_out: f64) -> [f64; 14] {
    [
        20.0,
        t_out,
        50.0,
        rh_out * 100.0,
        0.0,
        64.0,
        0.0,
        0.0,
        1e-20,
        1.244e-7,
        0.0,
        20.0,
        0.0,
        20.0,
    ]
}

/// (Δstored, ∫exterior flux) in g over `hours` hourly macro steps.
fn run_exposed_exterior(k: usize, hours_total: usize) -> Result<(f64, f64)> {
    let (t0, rh0) = outdoor_at(0);
    let mut settings = HamtSettings {
        timestep: 1.0 * hours,
        n_div: 30,
        ..Default::default()
    };
    settings.min_relaxation_factor = 0.1;
    let mut mgr = HamtManager::new(
        Some(settings),
        Some(BoundaryCondition::from_trnsys_inputs(
            &exposed_exterior_inputs(t0, rh0),
        )),
        build_construction(k)?,
        format!("exposed{k}"),
        AREA_M2 * square_meter,
    )?;
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    let area = mgr.surface_curr.area;
    let w0 = stored_kg(&mgr)?;
    let mut flux_kg = 0.0;
    for h in 0..hours_total {
        let (t, rh) = outdoor_at(h);
        mgr.boundary_condition =
            BoundaryCondition::from_trnsys_inputs(&exposed_exterior_inputs(t, rh));
        adaptive.current_time = (h as f64) * hours;
        mgr.solve_timestep_adaptive(&mut adaptive, None)?;
        flux_kg += (adaptive.flux_accumulator.m_ext_dt * area).value_in(kilograms);
        mgr.tick_timestep()?;
    }
    Ok(((stored_kg(&mgr)? - w0) * 1000.0, flux_kg * 1000.0))
}

/// The same mass balance on the EXTERIOR surface (four boundary cells behind one
/// surface node, steep isotherm / liquid transport near saturation, sub-zero
/// temperatures) under a harsh outdoor climate, for the gypsum lining split into 1
/// and 8 sub-layers. Old assembly, 240 h: k = 1 stored +839 g vs ∫flux +445 g;
/// k = 8 created 30.5 kg of water and took 5.4 min instead of 20 s. Fixed:
/// imbalance < 0.1 % and k-independent.
#[test]
fn exposed_exterior_conserves_moisture_harsh_climate() -> Result<()> {
    const HOURS: usize = 72;
    println!("=== exposed exterior, harsh outdoor climate, {HOURS} h ===");
    let (ds1, fl1) = run_exposed_exterior(1, HOURS)?;
    let (ds8, fl8) = run_exposed_exterior(8, HOURS)?;
    for (label, ds, fl) in [("k=1", ds1, fl1), ("k=8", ds8, fl8)] {
        let imb = ds - fl;
        let scale = ds.abs().max(fl.abs());
        println!(
            "  {label}: dStored {ds:>+9.2} g | int.flux {fl:>+9.2} g | imbalance {imb:>+7.2} g ({:>+5.2} %)",
            100.0 * imb / scale
        );
        assert!(
            imb.abs() <= 0.005 * scale + 0.05,
            "{label}: exterior moisture balance violated: stored {ds:+.2} g vs reported flux              {fl:+.2} g (imbalance {imb:+.2} g)"
        );
    }
    assert!(
        (ds8 - ds1).abs() <= 0.01 * ds1.abs().max(ds8.abs()) + 0.05,
        "identical layers give different walls under the exterior climate: k=1 {ds1:+.2} g vs          k=8 {ds8:+.2} g"
    );
    Ok(())
}
