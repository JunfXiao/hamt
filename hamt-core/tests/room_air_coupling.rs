#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Near-real room-air ↔ wall coupling test.
//!
//! A single hygroscopic wall (built from the EN 15026 material via the shared
//! `common` harness) is coupled to ONE lumped indoor-air node. The air node is
//! the only thing the surface exchanges heat and moisture with — no ventilation,
//! no other surface. A prescribed **zero-mean diurnal** internal load (vapour +
//! sensible) drives the zone; the wall is its sole buffer. Each macro-timestep:
//!
//!   1. push the current room (T_air, RH) onto the interior boundary condition,
//!   2. solve the wall for the step (adaptive substepping, BlockTridiagonal),
//!   3. read back the time-weighted interior surface fluxes (heat, moisture),
//!   4. advance the room air energy + moisture balance with those fluxes plus
//!      the diurnal load.
//!
//! WHY A ZERO-MEAN LOAD. The EN 15026 sorption isotherm is very flat in the
//! mid-RH range and rises extremely steeply toward saturation (w → 146 kg/m³ as
//! RH → 1). In a CLOSED zone whose only moisture sink is the wall, any net
//! positive vapour generation must eventually drive RH to 1 (the room air holds
//! only ~0.5 kg of vapour, while the wall's diffusion-limited uptake over a few
//! days is a few kg) — pushing the state onto the stiff near-saturation part of
//! that sharp curve. A zero-mean (alternating humidify/dry) load instead makes
//! the zone oscillate around its initial 50 % RH, staying on the well-behaved
//! part of the isotherm and reaching a bounded periodic steady state buffered
//! solely by the wall. That is exactly the moisture-buffering regime of
//! interest, and it keeps the test robust against the sharp curve.
//!
//! The test serves two purposes:
//!
//!   * CONVERGENCE — sweep the material cell count and check that the COUPLED
//!     room response (diurnal RH swing) converges as the mesh is refined. This
//!     is the realistic regime behind the "cell count changes the moisture
//!     curve" investigation. Finding: coarse meshes under-buffer (larger,
//!     less-damped room RH swing); refining converges by N ≈ 12–24, consistent
//!     with the standalone core-solver probe
//!     (`din_en_15026_benchmark::test_zz_cell_count_convergence_probe`).
//!
//!   * CONSERVATION / FLUX-QUERY OFFSET — at the moment of peak wall moisture
//!     uptake, compare the cumulative interior surface-flux integral (the
//!     diagnostic `Surface::moisture_flux`, time-weighted by the substepper)
//!     against the wall's actual stored-water excursion
//!     (`SurfaceReport::water_total`). Their gap quantifies the known
//!     diagnostic flux-query offset: the discrete balance conserves the
//!     linearised capacity `dwd_phi·V·Δφ` (endpoint tangent of the isotherm),
//!     while `water_total` tracks the true secant `V·Δw`. The offset is an
//!     implicit-Euler constitutive-linearisation effect (shrinks with Δt), NOT
//!     a geometric control-volume error. A separate closed-room balance
//!     residual validates the harness bookkeeping itself.

mod common;

use anyhow::Result;
use common::{CONSTRUCTION_NAME, create_library, init_tracing};

use hamt_core::boundary::{BoundaryCondition, celsius_to_kelvin};
use hamt_core::cell::CellType;
use hamt_core::hamt_manager::{AdaptiveState, HamtManager, HamtSettings, SolverMethod};
use hamt_core::uom::units::{
    dimensionless, hours, kelvin, kg_per_m2_s, kg_per_m2_s_pa, kilograms, pascals, seconds,
    square_meter, w_per_m2, w_per_m2_k,
};

// ---------------------------------------------------------------------------
// Lumped indoor-air node (sensible + latent), all in raw SI f64.
//
// The room is an external model coupled to HAMT only at the BC boundary, so
// working in raw f64 here is the sanctioned external-API escape hatch; we
// convert to/from typed quantities only when talking to the solver.
// ---------------------------------------------------------------------------

/// Water-vapour gas constant [J/(kg·K)] — matches `common::r_h2o`.
const R_V: f64 = 462.0;
/// Dry-air density [kg/m³] at room conditions.
const RHO_AIR: f64 = 1.2;
/// Dry-air specific heat [J/(kg·K)].
const CP_AIR: f64 = 1005.0;

/// Saturation vapour pressure [Pa] from the SAME Magnus fit the solver uses, so
/// the room and the wall agree on the RH↔VP mapping at the interface.
fn psat_pa(temp_k: f64) -> f64 {
    // strip: external room model works in raw SI; re-tag only to call the solver helper.
    hamt_core::physics::psy_psat_fn_temp(temp_k * kelvin).value_in(pascals)
}

struct RoomAir {
    volume_m3: f64,
    temp_k: f64,
    vapor_density: f64, // kg/m³ of water vapour
}

impl RoomAir {
    fn new(volume_m3: f64, temp_c: f64, rh: f64) -> Self {
        let temp_k = temp_c + 273.15;
        let vapor_density = rh * psat_pa(temp_k) / (R_V * temp_k);
        Self {
            volume_m3,
            temp_k,
            vapor_density,
        }
    }

    /// Relative humidity [0,1] implied by the current (T, ρ_v), clamped for use
    /// as a boundary condition.
    fn rh(&self) -> f64 {
        (self.vapor_density * R_V * self.temp_k / psat_pa(self.temp_k)).clamp(0.0, 1.0)
    }

    fn vapor_mass_kg(&self) -> f64 {
        self.vapor_density * self.volume_m3
    }

    fn heat_capacity_j_per_k(&self) -> f64 {
        RHO_AIR * CP_AIR * self.volume_m3
    }

    /// Advance the air balance by `dt` seconds.
    /// `q_into_wall` [W/m²] and `m_into_wall` [kg/m²·s] are the interior surface
    /// fluxes with the env→surface sign convention (positive = leaving the room
    /// into the wall). `s_heat` [W] and `s_moist` [kg/s] are the diurnal loads.
    fn step(
        &mut self,
        q_into_wall: f64,
        m_into_wall: f64,
        area_m2: f64,
        s_heat: f64,
        s_moist: f64,
        dt_s: f64,
    ) {
        // Moisture: closed room, single wall + prescribed load.
        let d_rho = (s_moist - m_into_wall * area_m2) * dt_s / self.volume_m3;
        self.vapor_density = (self.vapor_density + d_rho).max(0.0);
        // Sensible energy: single wall + prescribed load. (Latent is handled
        // inside the wall; the air node tracks sensible temperature only.)
        let d_temp = (s_heat - q_into_wall * area_m2) * dt_s / self.heat_capacity_j_per_k();
        self.temp_k += d_temp;
    }
}

// ---------------------------------------------------------------------------
// Coupled run
// ---------------------------------------------------------------------------

struct RunResult {
    n_material: usize,
    rh_min_last_day: f64,
    rh_max_last_day: f64,
    temp_swing_last_day_k: f64,
    /// Cumulative ∫ m_int·A dt at the instant of peak wall water uptake [kg]
    /// (diagnostic flux query, time-weighted).
    cum_flux_at_peak_kg: f64,
    /// Wall stored-water excursion at that same instant (water_total − start) [kg].
    wall_water_peak_kg: f64,
    /// Closed-room mass-balance residual over the whole run:
    /// ∫S − Δ(room vapour) − ∫(into wall) [kg]. ~0 validates the harness.
    room_balance_residual_kg: f64,
}

const AREA_M2: f64 = 12.0;
const VOLUME_M3: f64 = 30.0;
const INIT_T_C: f64 = 20.0;
const INIT_RH: f64 = 0.5;
/// Zero-mean diurnal vapour-load amplitude [kg/s] (~36 g/h peak). Sized so the
/// buffered zone RH stays well inside [0.35, 0.65] even on the coarsest mesh,
/// keeping the state on the well-behaved mid-RH part of the sharp EN 15026
/// isotherm (no saturation, no stiff near-w_sat region).
const M_AMP: f64 = 1.0e-5;
/// Zero-mean diurnal sensible-load amplitude [W]. Small swing about 20 °C.
const H_AMP: f64 = 50.0;
/// Load period [h].
const PERIOD_H: f64 = 24.0;

/// Zero-mean diurnal load shape in [-1, 1]: +half = humidify/heat, −half =
/// dry/cool. Net zero over each 24 h period, so the closed zone reaches a
/// bounded periodic steady state instead of drifting to saturation.
fn diurnal_shape(t_hours: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t_hours / PERIOD_H).sin()
}

fn run_room_coupling(num_div: usize, days: f64, dt_h: f64) -> Result<RunResult> {
    init_tracing();

    // Single hygroscopic layer; num_div pins the cell count. 20 mm so the diurnal
    // moisture wave resolves the whole thickness (the daily skin depth in this
    // material is ~1 cm), letting the buffered response reach its grid-converged
    // limit at a moderate cell count rather than leaving a thin unresolved skin.
    let library = create_library(INIT_T_C, INIT_RH, vec![0.02], vec![Some(num_div)]);
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = dt_h * hours;
    settings.solver_method = SolverMethod::BlockTridiagonal;
    settings.temp_convergence_tolerance = 1e-3 * kelvin;
    settings.rh_convergence_tolerance = 1e-4 * dimensionless;
    settings.min_div = 1; // num_div on the material wins; keep clamps wide
    settings.max_div = 200;
    // Cluster cells toward the surfaces (the daily moisture wave lives near the
    // interior face), so a given cell count resolves the active skin better.
    settings.div_distribution_factor = 0.7;
    settings.min_relaxation_factor = 0.5;

    let bc_init = BoundaryCondition {
        // Interior: realistic surface film + Lewis-range vapour coefficient.
        heat_transfer_coeff_int: 8.0 * w_per_m2_k,
        heat_transfer_coeff_conv_int: 8.0 * w_per_m2_k,
        vapor_transport_coeff_int: 1e-8 * kg_per_m2_s_pa,
        // Exterior: sealed (the wall is a pure buffer for the room).
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * kg_per_m2_s_pa,
        temp_outside_air: celsius_to_kelvin(INIT_T_C),
        temp_inside_air: celsius_to_kelvin(INIT_T_C),
        temp_outside_air_moisture: celsius_to_kelvin(INIT_T_C),
        temp_inside_air_moisture: celsius_to_kelvin(INIT_T_C),
        rh_outside_air: INIT_RH * dimensionless,
        rh_inside_air: INIT_RH * dimensionless,
        ..Default::default()
    };

    let mut mgr = HamtManager::new(
        Some(settings),
        Some(bc_init),
        construction,
        "RoomWall".to_string(),
        AREA_M2 * square_meter,
    )?;

    // create_library hardcodes a near-infinite internal SurfaceVaporCoef
    // (internal_value 1e10) which HamtManager::new imprints onto the BC; reset
    // both faces to the realistic coefficients we actually want to drive.
    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * kg_per_m2_s_pa;

    let n_material = mgr
        .surface_curr
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)))
        .count();

    // Room starts at equilibrium with the wall (both at INIT_RH / INIT_T).
    let mut room = RoomAir::new(VOLUME_M3, INIT_T_C, INIT_RH);
    let room_vapor_start_kg = room.vapor_mass_kg();
    // strip: external scalar mass bookkeeping
    let wall_water_start_kg = mgr
        .surface_curr
        .report(&mgr.boundary_condition)?
        .water_total
        .value_in(kilograms);

    let dt_s = (dt_h * hours).value_in(seconds);
    let area = mgr.surface_curr.area;
    let total_steps = (days * 24.0 / dt_h).round() as usize;

    let mut adap: AdaptiveState<()> = AdaptiveState::default();
    let mut cum_flux_into_wall_kg = 0.0;
    let mut total_generated_kg = 0.0;

    // Peak-uptake conservation tracking.
    let mut wall_water_peak_kg = 0.0; // max (water_total − start)
    let mut cum_flux_at_peak_kg = 0.0;

    // Last-day statistics.
    let mut rh_min_last = f64::MAX;
    let mut rh_max_last = f64::MIN;
    let mut t_min_last = f64::MAX;
    let mut t_max_last = f64::MIN;

    for step in 0..total_steps {
        let t_now_h = step as f64 * dt_h;
        let shape = diurnal_shape(t_now_h);
        let s_moist = M_AMP * shape;
        let s_heat = H_AMP * shape;

        // 1. push room state onto the interior BC.
        mgr.boundary_condition.temp_inside_air = room.temp_k * kelvin;
        mgr.boundary_condition.temp_inside_air_moisture = room.temp_k * kelvin;
        mgr.boundary_condition.temp_inside_mean_radiant = room.temp_k * kelvin;
        mgr.boundary_condition.rh_inside_air = room.rh() * dimensionless;

        // 2. solve the wall for this macro-step.
        mgr.solve_timestep_adaptive(&mut adap, None)?;
        mgr.tick_timestep()?;

        // 3. read back the time-weighted interior surface fluxes for THIS step
        //    (the accumulator is reset per macro-step, so it holds exactly this
        //    step's ∫q dt / ∫m dt). The time-average is the conservative choice
        //    for the room update.
        // strip: external-model scalars at the BC boundary
        let m_int_avg = adap.flux_accumulator.avg_m_int().value_in(kg_per_m2_s);
        let q_int_avg = adap.flux_accumulator.avg_q_int().value_in(w_per_m2);
        let into_wall_step_kg = (adap.flux_accumulator.m_int_dt * area).value_in(kilograms);
        cum_flux_into_wall_kg += into_wall_step_kg;
        total_generated_kg += s_moist * dt_s;

        // 4. advance the room with the conserved step-average fluxes.
        room.step(q_int_avg, m_int_avg, AREA_M2, s_heat, s_moist, dt_s);

        // Conservation: track the instant of peak wall water uptake.
        // strip: scalar mass for conservation comparison
        let wall_water_now_kg = mgr
            .surface_curr
            .report(&mgr.boundary_condition)?
            .water_total
            .value_in(kilograms);
        let excursion = wall_water_now_kg - wall_water_start_kg;
        if excursion > wall_water_peak_kg {
            wall_water_peak_kg = excursion;
            cum_flux_at_peak_kg = cum_flux_into_wall_kg;
        }

        // Last-day statistics.
        if t_now_h >= (days - 1.0) * 24.0 {
            let rh = room.rh();
            rh_min_last = rh_min_last.min(rh);
            rh_max_last = rh_max_last.max(rh);
            let tc = room.temp_k - 273.15;
            t_min_last = t_min_last.min(tc);
            t_max_last = t_max_last.max(tc);
        }
    }

    let room_vapor_change_kg = room.vapor_mass_kg() - room_vapor_start_kg;
    // Closed room: generated = Δ(room vapour) + ∫(into wall). Residual ≈ 0.
    let room_balance_residual_kg =
        total_generated_kg - room_vapor_change_kg - cum_flux_into_wall_kg;

    let offset_pct = if wall_water_peak_kg > 0.0 {
        (cum_flux_at_peak_kg - wall_water_peak_kg) / wall_water_peak_kg * 100.0
    } else {
        f64::NAN
    };

    println!(
        "  N(mat)={:>3} | RH last-day [min {:.4}, max {:.4}, swing {:.4}] | T swing {:.3} K | \
         peak uptake: ∫flux={:>9.4e} kg vs Δwater={:>9.4e} kg (offset {:>+5.1}%) | \
         room-balance resid={:>9.2e} kg",
        n_material,
        rh_min_last,
        rh_max_last,
        rh_max_last - rh_min_last,
        t_max_last - t_min_last,
        cum_flux_at_peak_kg,
        wall_water_peak_kg,
        offset_pct,
        room_balance_residual_kg,
    );

    Ok(RunResult {
        n_material,
        rh_min_last_day: rh_min_last,
        rh_max_last_day: rh_max_last,
        temp_swing_last_day_k: t_max_last - t_min_last,
        cum_flux_at_peak_kg,
        wall_water_peak_kg,
        room_balance_residual_kg,
    })
}

/// Near-real coupling test: sweep cell count, assert the coupled room response
/// converges with mesh refinement, and validate conservation.
#[test]
fn test_room_air_coupling_converges_and_conserves() -> Result<()> {
    println!("=== Near-real room-air ↔ wall coupling (closed zone, one wall) ===");
    println!(
        "    V={VOLUME_M3} m³, A={AREA_M2} m², 0.02 m EN-15026 layer, zero-mean diurnal load \
         (vapour ±{M_AMP:.1e} kg/s, sensible ±{H_AMP} W)"
    );

    let days = 5.0; // reach periodic steady state; analyse the last day
    let dt_h = 0.1;
    let cell_counts = [12usize, 24, 48];

    let mut results = Vec::new();
    for &n in &cell_counts {
        results.push(run_room_coupling(n, days, dt_h)?);
    }

    // ── Physical sanity ────────────────────────────────────────────────────
    for r in &results {
        assert!(
            r.rh_min_last_day > 0.0 && r.rh_max_last_day < 1.0,
            "N={}: room RH left the physical (0,1) range: [{}, {}]",
            r.n_material,
            r.rh_min_last_day,
            r.rh_max_last_day
        );
        assert!(
            r.rh_max_last_day - r.rh_min_last_day > 1e-3,
            "N={}: expected a non-trivial diurnal RH swing, got {:.5}",
            r.n_material,
            r.rh_max_last_day - r.rh_min_last_day
        );
        assert!(
            r.wall_water_peak_kg > 0.0,
            "N={}: wall should show a positive peak moisture uptake, got {:.3e} kg",
            r.n_material,
            r.wall_water_peak_kg
        );
        // Harness bookkeeping must close (closed-room mass balance).
        let scale = cum_scale(r);
        assert!(
            (r.room_balance_residual_kg / scale).abs() < 1e-6,
            "N={}: closed-room mass balance did not close: residual {:.3e} kg (scale {:.3e} kg)",
            r.n_material,
            r.room_balance_residual_kg,
            scale
        );
    }

    // ── Convergence of the COUPLED room response with cell count ────────────
    // results[] is coarse → fine. The diurnal RH swing is the buffering metric:
    // finer meshes resolve the near-surface vapour wave better, buffer more, and
    // produce a smaller / more damped zone swing.
    let swing = |r: &RunResult| r.rh_max_last_day - r.rh_min_last_day;
    let s_coarse = swing(&results[0]);
    let s_med = swing(&results[1]);
    let s_fine = swing(&results[2]);
    println!(
        "\n  Coupled RH-swing vs mesh (coarse→fine): N={} → {:.4}, N={} → {:.4}, N={} → {:.4}",
        results[0].n_material, s_coarse, results[1].n_material, s_med, results[2].n_material, s_fine,
    );
    println!(
        "  Coupled T-swing  vs mesh (coarse→fine): N={} → {:.3} K, N={} → {:.3} K, N={} → {:.3} K",
        results[0].n_material,
        results[0].temp_swing_last_day_k,
        results[1].n_material,
        results[1].temp_swing_last_day_k,
        results[2].n_material,
        results[2].temp_swing_last_day_k,
    );

    // Grid-convergence of the ROOM-FACING moisture response. On a thin wall the
    // daily vapour wave penetrates the full thickness, so the diurnal RH swing
    // converges early and the two finest meshes must agree closely. We assert
    // grid-INDEPENDENCE of the room result rather than a strict monotone trend:
    // once converged the swing varies only at the sub-percent (noise) level, so
    // a monotonicity test would be meaningless. (NB: the wall's INTERNAL
    // near-surface water profile — and hence `wall_water_peak` below — keeps
    // sharpening with N because the steep EN-15026 isotherm concentrates storage
    // in a thin high-φ skin; the room only feels the surface vapour flux, which
    // is what converges here.)
    let d_coarse = (s_med - s_coarse).abs();
    let d_fine = (s_fine - s_med).abs();
    let conv_rel = d_fine / s_fine;
    println!(
        "  Successive RH-swing change: |med−coarse|={:.4}, |fine−med|={:.4}; two-finest agreement {:.2}%",
        d_coarse,
        d_fine,
        conv_rel * 100.0
    );
    assert!(
        conv_rel < 0.08,
        "Room-facing RH swing not grid-converged: two finest meshes (N={}, N={}) disagree by \
         {:.2}% (>8%) — use more near-surface cells before trusting the coupled result.",
        results[1].n_material,
        results[2].n_material,
        conv_rel * 100.0
    );

    // ── AUDIT: diagnostic flux-query vs true stored-water offset ─────────────
    // At peak uptake, the cumulative interior surface-flux integral
    // (`moisture_flux`, the value reported to TRNSYS) must match the wall's true
    // stored-water excursion (`water_total`) — mass conservation, sealed exterior.
    // Before the surface-liquid fix this offset was −22…−38 % and grew with N
    // (the diagnostic reported only the convective vapour film term and omitted
    // the surface liquid/capillary flux the solver also drives). `moisture_flux`
    // now reports the TOTAL surface flux, so the coupling conserves mass.
    // See doc/flux_offset_investigation.md.
    println!("\n  Surface mass conservation — reported flux integral vs true stored water at peak uptake:");
    for r in &results {
        let offset = (r.cum_flux_at_peak_kg - r.wall_water_peak_kg) / r.wall_water_peak_kg;
        println!(
            "    N={:>3}: ∫flux(reported)={:>10.4e} kg, Δwater(true)={:>10.4e} kg, offset={:>+6.2}%",
            r.n_material,
            r.cum_flux_at_peak_kg,
            r.wall_water_peak_kg,
            offset * 100.0
        );
        assert!(
            offset.abs() < 0.03,
            "N={}: reported surface-flux integral deviates {:.2}% from the wall's true stored-water \
             change — surface mass-conservation regression (the reported flux must include the \
             surface liquid term; see doc/flux_offset_investigation.md).",
            r.n_material,
            offset * 100.0
        );
    }

    Ok(())
}

/// Scale for the closed-room balance residual check (avoid divide-by-zero when
/// the net cumulative flux is ~0 for a perfectly periodic run).
fn cum_scale(r: &RunResult) -> f64 {
    r.wall_water_peak_kg.abs().max(r.cum_flux_at_peak_kg.abs()).max(1e-6)
}
