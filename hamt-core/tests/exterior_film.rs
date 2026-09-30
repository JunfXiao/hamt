#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Exactly ONE exterior film: side 2 of a Type 395 surface is CONVECTIVE ONLY.
//!
//! The deck's side-2 coefficient (`hconv2`, e.g. 64 kJ/(h·m²·K) ≈ 17.8 W/(m²·K)) is a
//! convective coefficient; the side-2 longwave / solar exchange enters separately
//! through the deck's `qWG2` input (`BoundaryCondition::additional_energy_ext`).
//!
//! The exterior boundary sheet consists of four cells (Convection, Radiation, Sky,
//! Ground) that all hang on the exterior Virtual (surface) node. They used to carry
//! the SAME `h_ext` and `T_out`, i.e. four parallel films: the solver applied
//! `4·h_ext` while `Surface::heat_flux(Exterior)` (and therefore Type 395 output 9,
//! "HeatFlux_out") reported a single one — a quarter of the flux actually applied.
//! Measured on this wall before the fix: 117.0 W/m² instead of 81.3 W/m² for
//! h = 10 W/(m²·K) on both sides, and a reported exterior flux of −29.3 W/m².
//!
//! Wall: 0.1 m of the EN 15026 material (λ = 1.5 + 15.8e-3·w = 2.18 W/(m·K) at 50 % RH),
//! film 10 W/(m²·K) on both sides, 20 °C / 0 °C, vapour-tight (β ≈ 0), so only heat
//! moves and λ stays at its 50 %-RH value:
//!
//!   q = ΔT / (1/h_in + d/λ + 1/h_out) = 20 / (0.1 + 0.0459 + 0.1) = 81.3 W/m².

mod common;

use anyhow::Result;
use common::{CONSTRUCTION_NAME, create_library};

use hamt_core::boundary::BoundaryCondition;
use hamt_core::cell::{BoundaryLocation, CellType};
use hamt_core::hamt_manager::{AdaptiveState, HamtManager, HamtSettings};
use hamt_core::uom::units::{
    hours, kg_per_m2_s_pa, meter, square_meter, w_per_m_k, w_per_m2, w_per_m2_k,
};

/// Film coefficient on both sides [W/(m²·K)].
const H_W_M2K: f64 = 10.0;
/// Wall thickness [m].
const THICKNESS_M: f64 = 0.1;
const T_IN_C: f64 = 20.0;
const T_OUT_C: f64 = 0.0;
/// λ of the EN 15026 material at 50 % RH [W/(m·K)] (nominal, for the headline check).
const LAMBDA_NOMINAL: f64 = 2.18;
/// Hourly macro steps; the wall's thermal time constant is ≈ 12 h, so 600 h is steady.
const STEADY_HOURS: usize = 600;

/// Legacy 12-input Type 395 vector: one combined film per side (`hconv1`, `hconv2`),
/// no separate radiant input, both vapour coefficients ≈ 0 (vapour-tight).
/// `q_add_w` is the side-2 energy gain `qWG2` [W].
fn inputs(q_add_w: f64) -> [f64; 12] {
    let h_kj = H_W_M2K * 3.6; // kJ/(h·m²·K)
    [
        T_IN_C,
        T_OUT_C,
        50.0,          // RH in [%]
        50.0,          // RH out [%]
        h_kj,          // hconv1
        h_kj,          // hconv2
        0.0,           // qWG1
        q_add_w * 3.6, // qWG2 [kJ/h]
        1e-20,         // beta1
        1e-20,         // beta2
        0.0,           // user-defined heat
        T_IN_C,        // TAIR
    ]
}

struct Steady {
    /// Reported interior flux, environment → wall [W/m²].
    q_int: f64,
    /// Reported exterior flux, environment → wall [W/m²].
    q_ext: f64,
    /// Thermal resistance of the wall from the solver's own λ(w) [m²K/W].
    r_wall: f64,
    /// (htc > 0) interior / exterior boundary cells of the first column.
    active_films_int: usize,
    active_films_ext: usize,
    /// Exterior boundary cells with a non-zero vapour coefficient.
    active_vapour_ext: usize,
}

fn settings() -> HamtSettings {
    let mut s = HamtSettings {
        timestep: 1.0 * hours,
        n_div: 1, // the material's num_div wins
        ..Default::default()
    };
    s.min_relaxation_factor = 0.1; // as the DLL
    s
}

/// Boundary-cell census and wall resistance of column (0,0).
fn census(mgr: &HamtManager) -> (f64, usize, usize, usize) {
    let mut r_wall = 0.0;
    let (mut films_int, mut films_ext, mut vap_ext) = (0, 0, 0);
    for ((_, j, k), cell) in mgr.surface_curr.cells.indexed_iter() {
        if j != 0 || k != 0 {
            continue;
        }
        match &cell.cell_type {
            CellType::Material(_) => {
                // strip: test-side scalar bookkeeping, R = Σ dx/λ [m²K/W]
                r_wall += cell.size.x().value_in(meter) / cell.w_thermal_c.value_in(w_per_m_k);
            }
            CellType::Boundary(_, BoundaryLocation::Interior) => {
                if cell.htc.value_in(w_per_m2_k) > 0.0 {
                    films_int += 1;
                }
            }
            CellType::Boundary(_, BoundaryLocation::Exterior) => {
                if cell.htc.value_in(w_per_m2_k) > 0.0 {
                    films_ext += 1;
                }
                if cell.vtc.value_in(kg_per_m2_s_pa) > 0.0 {
                    vap_ext += 1;
                }
            }
            CellType::Virtual(_) => {}
        }
    }
    (r_wall, films_int, films_ext, vap_ext)
}

/// Advance `n_hours` hourly macro steps with the production driver.
fn run_hours(mgr: &mut HamtManager, q_add_w: f64, n_hours: usize) -> Result<()> {
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    let input = inputs(q_add_w);
    for h in 0..n_hours {
        mgr.boundary_condition = BoundaryCondition::from_trnsys_inputs(&input);
        adaptive.current_time = (h as f64) * hours;
        mgr.solve_timestep_adaptive(&mut adaptive, None)?;
        mgr.tick_timestep()?;
    }
    Ok(())
}

fn steady_state(mut mgr: HamtManager, q_add_w: f64) -> Result<Steady> {
    run_hours(&mut mgr, q_add_w, STEADY_HOURS)?;
    let rep = mgr.surface_curr.report(&mgr.boundary_condition)?;
    let (r_wall, films_int, films_ext, vap_ext) = census(&mgr);
    Ok(Steady {
        q_int: rep.q_surf_int.value_in(w_per_m2),
        q_ext: rep.q_surf_ext.value_in(w_per_m2),
        r_wall,
        active_films_int: films_int,
        active_films_ext: films_ext,
        active_vapour_ext: vap_ext,
    })
}

fn manager_1d() -> Result<HamtManager> {
    let library = create_library(10.0, 0.5, vec![THICKNESS_M], vec![Some(20)]);
    let construction = library.get_construction(CONSTRUCTION_NAME)?;
    HamtManager::new(
        Some(settings()),
        Some(BoundaryCondition::from_trnsys_inputs(&inputs(0.0))),
        construction,
        "exterior_film_1d".to_string(),
        1.0 * square_meter,
    )
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Headline regression: the steady-state flux through the wall is the flux of ONE
/// film per side, and the reported exterior flux is the flux actually applied.
#[test]
fn exterior_side_has_exactly_one_film_steady_state_u() -> Result<()> {
    let s = steady_state(manager_1d()?, 0.0)?;
    let q_nominal = (T_IN_C - T_OUT_C) / (2.0 / H_W_M2K + THICKNESS_M / LAMBDA_NOMINAL);
    let q_solver_lambda = (T_IN_C - T_OUT_C) / (2.0 / H_W_M2K + s.r_wall);
    println!(
        "  nominal U-test: q_int = {:.3} W/m2 (nominal {:.3}, with the solver's λ {:.3}), \
         reported q_ext = {:.3} W/m2, R_wall = {:.5} m2K/W; active films int/ext = {}/{}, \
         exterior vapour films = {}",
        s.q_int,
        q_nominal,
        q_solver_lambda,
        s.q_ext,
        s.r_wall,
        s.active_films_int,
        s.active_films_ext,
        s.active_vapour_ext
    );
    assert!(
        rel(s.q_int, q_nominal) < 0.005,
        "steady-state flux {:.3} W/m2 differs by {:.2} % from the one-film-per-side value {:.3} W/m2 \
         (four exterior films would give 117.0 W/m2)",
        s.q_int,
        100.0 * rel(s.q_int, q_nominal),
        q_nominal
    );
    assert!(
        rel(s.q_int, q_solver_lambda) < 0.002,
        "steady-state flux {:.3} W/m2 is not ΔT/(2/h + R_wall) = {:.3} W/m2 with the solver's own λ",
        s.q_int,
        q_solver_lambda
    );
    // Energy conservation of the REPORTED fluxes: what enters at the interior leaves at
    // the exterior (q_ext is env → wall, hence negative).
    assert!(
        rel(-s.q_ext, s.q_int) < 0.005,
        "reported exterior flux {:.3} W/m2 is not the interior flux {:.3} W/m2 at steady state: \
         Type 395 output 9 must be the flux actually applied",
        s.q_ext,
        s.q_int
    );
    Ok(())
}

/// Side-2 energy gain (`qWG2`, where the deck puts solar + longwave exchange) is
/// deposited in the exterior surface cell on top of the ONE convective film:
/// q_int + q_ext + q_add = 0 at steady state, q_c = (h·ΔT − q_add)/(2 + h·R_wall).
#[test]
fn exterior_film_with_side2_energy_gain_balances() -> Result<()> {
    const Q_ADD: f64 = 20.0; // W (= W/m², A = 1 m²)
    let s = steady_state(manager_1d()?, Q_ADD)?;
    let q_c = (H_W_M2K * (T_IN_C - T_OUT_C) - Q_ADD) / (2.0 + H_W_M2K * s.r_wall);
    println!(
        "  side-2 gain {Q_ADD} W/m2: q_int = {:.3} (expected {:.3}), q_ext = {:.3} (expected {:.3}), \
         residual q_int + q_ext + q_add = {:.4} W/m2",
        s.q_int,
        q_c,
        s.q_ext,
        -(q_c + Q_ADD),
        s.q_int + s.q_ext + Q_ADD
    );
    assert!(
        rel(s.q_int, q_c) < 0.005,
        "interior flux {:.3} W/m2 differs from the one-film closed form {:.3} W/m2",
        s.q_int,
        q_c
    );
    let residual = s.q_int + s.q_ext + Q_ADD;
    assert!(
        residual.abs() < 0.005 * q_c.abs(),
        "wall energy balance q_int + q_ext + q_add = {residual:.4} W/m2 ≠ 0: the reported exterior \
         flux {:.3} W/m2 is not the film flux actually applied",
        s.q_ext
    );
    Ok(())
}

/// Structure: exactly one active exterior film (Convection) and one exterior vapour
/// film; the Radiation / Sky / Ground placeholders are inert (htc = vtc = 0) but still
/// exist (cell layout and the Type395_*.out column indices are unchanged). The interior
/// legacy (12-input) path keeps its single Convection film.
#[test]
fn exterior_radiation_sky_ground_cells_are_inert() -> Result<()> {
    let mut mgr = manager_1d()?;
    run_hours(&mut mgr, 0.0, 3)?;
    let (_, films_int, films_ext, vap_ext) = census(&mgr);
    assert_eq!(films_int, 1, "interior films (legacy 12-input deck)");
    assert_eq!(
        films_ext, 1,
        "exterior boundary cells with htc > 0: only the Convection cell may carry h_ext"
    );
    // β = 1e-20 is > 0, so the Convection cell keeps it; the others must not.
    assert_eq!(
        vap_ext, 1,
        "exterior boundary cells with a vapour coefficient: only the Convection cell"
    );
    let mut kinds: Vec<String> = mgr
        .surface_curr
        .cells
        .iter()
        .filter_map(|c| match &c.cell_type {
            CellType::Boundary(b, BoundaryLocation::Exterior) => Some(format!("{b:?}")),
            _ => None,
        })
        .collect();
    kinds.sort();
    assert_eq!(kinds, ["Convection", "Ground", "Radiation", "Sky"]);
    Ok(())
}

/// The 2-D lattice builder stacks the same four exterior cells per column: the film
/// must be single there too (ADI path, two identical columns, adiabatic sides).
#[test]
fn two_d_wall_has_one_exterior_film_per_column() -> Result<()> {
    let library = create_library(10.0, 0.5, vec![THICKNESS_M], vec![Some(10)]);
    let construction = library.get_construction(CONSTRUCTION_NAME)?;
    let mgr = HamtManager::new_2d(
        Some(settings()),
        Some(BoundaryCondition::from_trnsys_inputs(&inputs(0.0))),
        construction,
        "exterior_film_2d".to_string(),
        1.0 * square_meter,
        2,
        1.0 * meter,
    )?;
    let s = steady_state(mgr, 0.0)?;
    let q_nominal = (T_IN_C - T_OUT_C) / (2.0 / H_W_M2K + THICKNESS_M / LAMBDA_NOMINAL);
    println!(
        "  2-D (2 columns): q_int = {:.3} W/m2 (nominal {:.3}), reported q_ext = {:.3} W/m2",
        s.q_int, q_nominal, s.q_ext
    );
    assert!(
        rel(s.q_int, q_nominal) < 0.005,
        "2-D steady-state flux {:.3} W/m2 differs from {:.3} W/m2 by {:.2} %",
        s.q_int,
        q_nominal,
        100.0 * rel(s.q_int, q_nominal)
    );
    assert!(
        rel(-s.q_ext, s.q_int) < 0.005,
        "2-D reported exterior flux {:.3} W/m2 is not the interior flux {:.3} W/m2",
        s.q_ext,
        s.q_int
    );
    Ok(())
}
