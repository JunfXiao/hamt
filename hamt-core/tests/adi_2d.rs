#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Gating test for the 2-D ADI path (Phase 2a, Task 9).
//!
//! A `(nx, ny, 1)` wall that is **uniform in y** — `ny` identical x-columns,
//! adiabatic side faces (the `Surface::new` default), and 1-D-equivalent
//! boundary conditions on every column — must reproduce the trusted DIN EN
//! 15026 1-D analytical benchmark on *every* y-row when solved with
//! `SolverMethod::Adi`. Any spurious y-gradient (a stencil/grid/cross-lag bug)
//! shows up either as a per-row mismatch against the 1-D reference or as a
//! non-zero cross-row residual.
//!
//! The physics harness (curves, library, BC schedule, preheat + run loop) and
//! the reference tables are copied from `din_en_15026_benchmark.rs`; only the
//! manager construction (`new_2d`) and the comparison (per-row + cross-row,
//! against the same tolerances `compare_results` uses) differ.

use anyhow::{Ok, Result};
use hamt_core::material::{Material, MaterialLibrary, SurfaceVaporCoef};
use hamt_core::physics::{Curve, CurveScale, LogY};
use hamt_core::uom::dimensions::{
    Density, Dimensionless, SpecificHeatCapacity, Temperature, Time, VolumetricHeatCapacity,
    WaterContent,
};
use hamt_core::uom::units::{
    days, dimensionless, hours, j_per_kg_k, j_per_m3_k, kelvin, kg_per_m2_s_pa, kg_per_m3, m_per_s,
    m2_per_s, meter, square_meter, w_per_m_k, w_per_m2_k,
};
use hamt_core::uom::{
    DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
    Dimension, Quantity,
};
use hamt_core::{
    boundary::{BoundaryCondition, celsius_to_kelvin, kelvin_to_celsius},
    uom::units::seconds,
};
use hamt_core::{
    cell::CellType,
    hamt_manager::{AdaptiveState, HamtManager, HamtSettings, SolverMethod},
};
use std::collections::HashSet;
use std::ops::Div;
use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt};

// ---------------------------------------------------------------------------
// Reference results — copied verbatim from din_en_15026_benchmark.rs
// (Tabelle B.1 / B.2). Same data the 1-D `compare_results` validates against.
// ---------------------------------------------------------------------------
const REF_WATER_CONTENT: &[(i32, f64, f64)] = &[
    // (days, depth_m, wc_kg_m3)
    (7, 0.01, 52.4),
    (7, 0.02, 43.4),
    (7, 0.03, 42.9),
    (7, 0.04, 42.9),
    (7, 0.05, 42.9),
    (7, 0.06, 42.9),
    (7, 0.08, 42.9),
    (7, 0.10, 42.9),
    (30, 0.01, 83.1),
    (30, 0.02, 53.3),
    (30, 0.03, 45.7),
    (30, 0.04, 43.6),
    (30, 0.05, 43.0),
    (30, 0.06, 42.9),
    (30, 0.08, 42.9),
    (30, 0.10, 42.9),
    (365, 0.01, 119.7),
    (365, 0.02, 106.6),
    (365, 0.03, 90.8),
    (365, 0.04, 75.7),
    (365, 0.05, 64.9),
    (365, 0.06, 57.8),
    (365, 0.08, 50.0),
    (365, 0.10, 46.3),
];

const REF_TEMPERATURE: &[(i32, f64, f64)] = &[
    // (days, depth_m, temp_c)
    (7, 0.5, 26.64),
    (7, 1.0, 23.84),
    (7, 1.5, 21.91),
    (7, 2.0, 20.82),
    (7, 2.5, 20.30),
    (7, 3.0, 20.09),
    (7, 4.0, 20.01),
    (7, 5.0, 20.00),
    (30, 0.5, 28.35),
    (30, 1.0, 26.75),
    (30, 1.5, 25.29),
    (30, 2.0, 24.01),
    (30, 2.5, 22.93),
    (30, 3.0, 22.07),
    (30, 4.0, 20.92),
    (30, 5.0, 20.35),
    (365, 0.5, 29.54),
    (365, 1.0, 29.06),
    (365, 1.5, 28.58),
    (365, 2.0, 28.11),
    (365, 2.5, 27.64),
    (365, 3.0, 27.19),
    (365, 4.0, 26.30),
    (365, 5.0, 25.47),
];

// Same tolerances `compare_results` applies.
const PERCENTAGE_RANGE_T: f64 = 0.01;
const PERCENTAGE_RANGE_RH: f64 = 0.025;
// Cross-row uniformity residual: a truly y-uniform wall must keep every row
// identical to numerical round-off. Loose enough for f64 accumulation across a
// year of substeps, tight enough to catch any real y-coupling leak.
const CROSS_ROW_TOL_T_K: f64 = 1e-6;
const CROSS_ROW_TOL_W: f64 = 1e-6; // kg/m3

// ---------------------------------------------------------------------------
// Physics harness — copied from din_en_15026_benchmark.rs.
// ---------------------------------------------------------------------------
fn t_ref() -> Temperature<f64> {
    293.15 * kelvin
}
fn rho_w() -> Density<f64> {
    1000.0 * kg_per_m3
}
fn r_h2o() -> SpecificHeatCapacity<f64> {
    462.0 * j_per_kg_k
}
fn water_sat() -> Quantity<f64, DIM_WATER_CONTENT> {
    146.0 * kg_per_m3
}

fn curve_by_func<const DX: Dimension, const DY: Dimension, S: CurveScale>(
    func: fn(Quantity<f64, DX>) -> Quantity<f64, DY>,
    x_range: (Quantity<f64, DX>, Quantity<f64, DX>),
    delta: Quantity<f64, DX>,
) -> Curve<DX, DY, S>
where
    Quantity<f64, DX>: Div<Quantity<f64, DX>, Output = Dimensionless<f64>>,
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    let data_len = ((x_range.1 - x_range.0) / delta) + 1.0f64;
    let x_series: Vec<Quantity<f64, DX>> = (0..data_len.value() as usize)
        .map(|i| x_range.0 + i as f64 * delta)
        .collect();
    curve_by_func_and_x_values(func, x_series)
}

fn curve_by_func_and_x_values<const DX: Dimension, const DY: Dimension, S: CurveScale>(
    func: fn(Quantity<f64, DX>) -> Quantity<f64, DY>,
    x_values: impl IntoIterator<Item = Quantity<f64, DX>>,
) -> Curve<DX, DY, S>
where
    Quantity<f64, DX>: Div<Quantity<f64, DX>, Output = Dimensionless<f64>>,
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    let x_values: Vec<_> = x_values.into_iter().collect();
    let y_series: Vec<Quantity<f64, DY>> = x_values.iter().map(|x| func(*x)).collect();
    Curve::<DX, DY, S>::try_new(x_values.into_iter().collect(), y_series)
        .expect("benchmark curve well-formed")
}

fn sorption_isotherm(rh: Dimensionless<f64>) -> Quantity<f64, DIM_WATER_CONTENT> {
    let w_sat = 146.0 * kg_per_m3;
    let r_h2o = r_h2o().value_in(j_per_kg_k);
    let t_ref = t_ref().value_in(kelvin);
    let rho_w = rho_w().value_in(kg_per_m3);

    w_sat
        * (1.0f64
            + (8.0e-8 * r_h2o * t_ref * rho_w * rh.value().ln())
                .abs()
                .powf(1.6))
        .powf(-0.375)
}

fn interpolate_data_points(
    start: f64,
    end: f64,
    d_point_len: usize,
    proportion: f64,
) -> impl Iterator<Item = f64> {
    assert!(proportion > 0.0 && proportion < 1.0);
    let k = proportion * 5.0;
    let denom = 1.0 - (-k).exp();
    let n = (d_point_len - 1) as f64;
    (0..d_point_len).map(move |i| {
        let t = i as f64 / n;
        let numer = 1.0 - (-k * t).exp();
        let ratio = numer / denom;
        start + ratio * (end - start)
    })
}

fn get_sorption_isotherm_curve() -> Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> {
    let mut x_values: Vec<_> =
        interpolate_data_points(0.0, 1.0, INTERPOLATION_POINTS, INTERPOLATION_DELTA)
            .map(|v| v * dimensionless)
            .collect();
    if let Some(last) = x_values.last_mut() {
        *last = 1.0 * dimensionless;
    }
    curve_by_func_and_x_values(|x: Dimensionless<f64>| sorption_isotherm(x), x_values)
}

fn get_vapor_diffusion_curve() -> Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> {
    let x_values: Vec<_> =
        interpolate_data_points(0.0, 1.0, INTERPOLATION_POINTS, INTERPOLATION_DELTA)
            .map(|v| v * dimensionless)
            .collect();
    curve_by_func_and_x_values(
        |x: Dimensionless<f64>| {
            let x = x.value().clamp(0.0, 1.0) * dimensionless;
            let water_content_ratio = sorption_isotherm(x) / water_sat();
            let rest_water_ratio = 1.0 - water_content_ratio.value();
            200.0 * (0.503 * rest_water_ratio.powf(2.0) + 0.497) / rest_water_ratio * dimensionless
        },
        x_values,
    )
}

fn get_thermal_conductivity_curve() -> Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> {
    let x_range = (0.0 * water_sat(), 1.0 * water_sat());
    let delta = 0.01 * water_sat();
    curve_by_func(
        |x: Density<f64>| (1.5 + 15.8e-3 * x.value_in(kg_per_m3)) * w_per_m_k,
        x_range,
        delta,
    )
}

fn get_liquid_transport_coeff_curve() -> Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> {
    let x_values: Vec<WaterContent<f64>> = interpolate_data_points(
        0.0 * water_sat().value_in(kg_per_m3),
        0.95 * water_sat().value_in(kg_per_m3),
        INTERPOLATION_POINTS,
        INTERPOLATION_DELTA,
    )
    .map(|x| x * kg_per_m3)
    .collect();

    curve_by_func_and_x_values(
        |x: WaterContent<f64>| {
            let w = x.value_in(kg_per_m3);
            let w_sat = water_sat().value_in(kg_per_m3);
            let w_ratio = w / w_sat;
            let p_suc = 0.125e8f64 * ((w_ratio).powf(-1.0f64 / 0.375) - 1.0).powf(0.625);

            let k_w = (-39.2619 + 0.0704 * (w - 73.0)
                - 1.7420e-4 * (w - 73.0).powi(2)
                - 2.7953e-6 * (w - 73.0).powi(3)
                - 1.1566e-7 * (w - 73.0).powi(4)
                + 2.5969e-9 * (w - 73.0).powi(5))
            .exp();
            let dpc_dw = p_suc * 0.625f64 / (1.0f64 - w_ratio.powf(1.0 / 0.375)) / 0.375 / w;
            k_w * dpc_dw * m2_per_s
        },
        x_values,
    )
}

fn volumetric_heat_capacity() -> VolumetricHeatCapacity<f64> {
    1.824e6 * j_per_m3_k
}
fn rho_mat() -> Density<f64> {
    200.0 * kg_per_m3
}
fn spec_heat_capacity() -> SpecificHeatCapacity<f64> {
    volumetric_heat_capacity() / rho_mat()
}

const MAT_NAME: &str = "DIN EN 15026 Mat";
const CONSTRUCTION_NAME: &str = "DIN EN 15026 Construction";
const INTERPOLATION_POINTS: usize = 200;
const INTERPOLATION_DELTA: f64 = 0.6;

fn create_library(
    init_temp_celsius: f64,
    init_rh: f64,
    thicknesses: Vec<f64>,
    num_divs: Vec<Option<usize>>,
) -> MaterialLibrary {
    let thermal_conductivity = get_thermal_conductivity_curve();
    let sorption_isotherm = get_sorption_isotherm_curve();
    let base_material = Material {
        name: MAT_NAME.to_string(),
        thickness: 0.0 * meter,
        density: rho_mat(),
        thermal_conductivity,
        spec_heat: spec_heat_capacity(),
        porosity: water_sat() / rho_w(),
        initial_water_content: sorption_isotherm.interpolate(init_rh * dimensionless).0 / rho_mat(),
        suction: get_liquid_transport_coeff_curve(),
        redistribution: get_liquid_transport_coeff_curve(),
        sorption_isotherm,
        vapor_diffusion_resistance: get_vapor_diffusion_curve(),
        sorption_enthalpy: None,
        initial_temp: celsius_to_kelvin(init_temp_celsius),
        num_div: None,
        div_distribution_factor: 0.3,
    };

    let lib = MaterialLibrary::new();
    let mut layer_names = Vec::new();
    for (i, (d, num_div)) in thicknesses.iter().zip(num_divs.into_iter()).enumerate() {
        let mut mat = base_material.clone();
        mat.name = format!("{}_{}", MAT_NAME, i);
        mat.thickness = *d * meter;
        mat.num_div = num_div;
        layer_names.push(mat.name.clone());
        lib.add_material(mat).expect("Cannot add material.");
    }

    let surface_coeff = SurfaceVaporCoef {
        is_external_const: true,
        external_value: 0.0,
        internal_value: 1.0e10,
        is_internal_const: true,
    };
    lib.create_construction(CONSTRUCTION_NAME.to_string(), layer_names, surface_coeff)
        .expect("Cannot create construction");
    lib
}

static INIT: Once = Once::new();
fn init_tracing() {
    INIT.call_once(|| {
        fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("info,hamt_core=info")),
            )
            .with_test_writer()
            .init();
    });
}

// ---------------------------------------------------------------------------
// 2-D comparison helpers.
// ---------------------------------------------------------------------------

/// Read a material-cell quantity along y-row `j` at the depth nearest `depth_m`.
/// Returns (value, [i,j,0]). Mirrors `find_cell_idx_at_depth` but constrained to
/// a single y-row so we can prove uniformity per row.
fn cell_value_at_depth_row<F, const D: Dimension>(
    surface: &hamt_core::surface::Surface,
    depth_m: f64,
    j: usize,
    f: F,
) -> (f64, [usize; 3])
where
    F: Fn(&hamt_core::cell::Cell) -> Quantity<f64, D>,
{
    let dims = surface.cells.raw_dim();
    let nx = dims[0];
    let target = depth_m * meter;
    let mut best: Option<([usize; 3], f64, f64)> = None; // (idx, dist, value)
    for i in 0..nx {
        let cell = &surface.cells[[i, j, 0]];
        if !matches!(cell.cell_type, CellType::Material(_)) {
            continue;
        }
        // strip: scalar nearest-cell distance compare in metres
        let dist = (cell.origin.x().value_in(meter) - target.value_in(meter)).abs();
        let take = match best {
            Some((_, best_dist, _)) => dist < best_dist,
            None => true,
        };
        if take {
            best = Some(([i, j, 0], dist, f(cell).value_unchecked()));
        }
    }
    let (idx, _, value) = best.expect("no material cell on row");
    (value, idx)
}

fn ny_of(surface: &hamt_core::surface::Surface) -> usize {
    surface.cells.raw_dim()[1]
}

/// Compare a 2-D uniform surface against the 1-D reference at `duration_day`,
/// on EVERY y-row, plus a cross-row uniformity check. Returns the number of
/// failures (0 = pass).
fn compare_2d(surface: &hamt_core::surface::Surface, duration_day: i32) -> u32 {
    let ny = ny_of(surface);
    let mut fail_count = 0u32;

    println!("=== ADI 2-D comparison at day {} (ny={}) ===", duration_day, ny);

    // --- Temperature: each y-row vs 1-D reference. ---
    println!(
        "{:>8} {:>6} {:>9} {:>9} {:>8} {:>6}",
        "depth", "row", "T(C)", "ref(C)", "diff%", "ok"
    );
    let mut t_xrow_max_resid = 0.0_f64;
    for (d, depth, ref_t) in REF_TEMPERATURE.iter().filter(|(d, _, _)| *d == duration_day) {
        let _ = d;
        // Collect every row's value (nearest-cell, NO interpolation, so cross-row
        // residual is a pure cell-to-cell comparison).
        let row_vals: Vec<f64> = (0..ny)
            .map(|j| {
                let (t_k, _) = cell_value_at_depth_row(surface, *depth, j, |c| c.cell_temp);
                kelvin_to_celsius(t_k * kelvin)
            })
            .collect();
        let row_min = row_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = row_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        t_xrow_max_resid = t_xrow_max_resid.max(row_max - row_min);

        for (j, &cell_t) in row_vals.iter().enumerate() {
            let dev = (cell_t - ref_t).abs() / ref_t;
            let ok = dev <= PERCENTAGE_RANGE_T;
            if !ok {
                fail_count += 1;
            }
            if j == 0 || !ok {
                println!(
                    "{:>8.2} {:>6} {:>9.3} {:>9.3} {:>7.2}% {:>6}",
                    depth,
                    j,
                    cell_t,
                    ref_t,
                    dev * 100.0,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
    }

    // --- Water content: each y-row vs 1-D reference. ---
    println!(
        "{:>8} {:>6} {:>12} {:>12} {:>8} {:>6}",
        "depth", "row", "W(kg/m3)", "ref", "diff%", "ok"
    );
    let mut w_xrow_max_resid = 0.0_f64;
    for (d, depth, ref_wc) in REF_WATER_CONTENT.iter().filter(|(d, _, _)| *d == duration_day) {
        let _ = d;
        let row_vals: Vec<f64> = (0..ny)
            .map(|j| {
                let (w, _) = cell_value_at_depth_row(surface, *depth, j, |c| c.volumetric_water);
                w // already kg/m3 (SI base for water content)
            })
            .collect();
        let row_min = row_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = row_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w_xrow_max_resid = w_xrow_max_resid.max(row_max - row_min);

        for (j, &cell_w) in row_vals.iter().enumerate() {
            let dev = (cell_w - ref_wc).abs() / ref_wc;
            let ok = dev <= PERCENTAGE_RANGE_RH;
            if !ok {
                fail_count += 1;
            }
            if j == 0 || !ok {
                println!(
                    "{:>8.2} {:>6} {:>12.3} {:>12.3} {:>7.2}% {:>6}",
                    depth,
                    j,
                    cell_w,
                    ref_wc,
                    dev * 100.0,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
    }

    println!(
        "cross-row max residual: ΔT={:.3e} K (tol {:.0e}), ΔW={:.3e} kg/m3 (tol {:.0e})",
        t_xrow_max_resid, CROSS_ROW_TOL_T_K, w_xrow_max_resid, CROSS_ROW_TOL_W
    );
    if t_xrow_max_resid > CROSS_ROW_TOL_T_K {
        eprintln!(
            "CROSS-ROW T residual {:.3e} K exceeds {:.0e} — wall is NOT y-uniform (stencil/cross-lag bug)",
            t_xrow_max_resid, CROSS_ROW_TOL_T_K
        );
        fail_count += 1;
    }
    if w_xrow_max_resid > CROSS_ROW_TOL_W {
        eprintln!(
            "CROSS-ROW W residual {:.3e} kg/m3 exceeds {:.0e} — wall is NOT y-uniform (stencil/cross-lag bug)",
            w_xrow_max_resid, CROSS_ROW_TOL_W
        );
        fail_count += 1;
    }

    fail_count
}

// ---------------------------------------------------------------------------
// The gating test.
// ---------------------------------------------------------------------------
#[test]
fn adi_2d_uniform_reduces_to_1d() -> Result<()> {
    init_tracing();

    let time_step: Time<f64> = 2.0 * hours;
    let ny = 4usize;
    let y_extent = 1.0 * meter;

    // Same 5-layer construction as the 1-D benchmark.
    let library = create_library(
        20.0,
        0.5,
        vec![0.1, 0.9, 4.0, 10.0, 20.0],
        vec![Some(60), Some(60), Some(60), Some(60), Some(60)],
    );
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = time_step;
    settings.max_iterations = 3000;
    settings.temp_convergence_tolerance = 1e-6 * kelvin;
    settings.rh_convergence_tolerance = 1e-4 * dimensionless;
    settings.max_div = 50;
    settings.min_div = 20;
    settings.div_length = 1e-5 * meter;
    settings.adj_length = 1e-4 * meter;
    settings.min_relaxation_factor = 1.0;
    settings.solver_method = SolverMethod::Adi;

    let boundary_coeff = BoundaryCondition {
        heat_transfer_coeff_ext: 9e9 * w_per_m2_k,
        heat_transfer_coeff_int: 9e-9 * w_per_m2_k,
        mass_convection_coeff_ext: 9e9 * m_per_s,
        mass_convection_coeff_int: 9e-9 * m_per_s,
        vapor_transport_coeff_ext: 9e9 * kg_per_m2_s_pa,
        vapor_transport_coeff_int: 9e-9 * kg_per_m2_s_pa,
        ..Default::default()
    };

    let boundary_condition_init = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        rh_outside_air: 0.5 * dimensionless,
        rh_inside_air: 0.5 * dimensionless,
        ..boundary_coeff
    };

    let boundary_condition = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(30.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(30.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        rh_outside_air: 0.95 * dimensionless,
        rh_inside_air: 0.5 * dimensionless,
        ..boundary_coeff
    };

    // 2-D manager: ny identical columns stacked along y, adiabatic side faces.
    let mut mgr = HamtManager::new_2d(
        Some(settings.clone()),
        Some(boundary_condition_init),
        construction,
        "AdiBenchmarkSurface".to_string(),
        1.0 * square_meter,
        ny,
        y_extent,
    )?;

    let dim = mgr.surface_curr.cells.raw_dim();
    println!(
        "2-D grid shape: ({}, {}, {}), cells={}",
        dim[0],
        dim[1],
        dim[2],
        mgr.surface_curr.cells.len()
    );
    assert_eq!(dim[1], ny, "expected ny columns");
    assert_eq!(dim[2], 1, "expected nz=1");

    // Checkpoint days (union of both reference tables), sorted ascending.
    let mut checkpoints: Vec<Time<f64>> = REF_TEMPERATURE
        .iter()
        .chain(REF_WATER_CONTENT.iter())
        .map(|(d, _, _)| *d)
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|d| (d as f64) * days)
        .collect();
    checkpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // --- Preheat to a stable uniform initial state (same criterion as 1-D). ---
    {
        let mut constant_time: Time<f64> = 0.0 * seconds;
        let allowed_err: Dimensionless<f64> = 1e-4 * dimensionless;
        let mut adap_state: AdaptiveState<()> = AdaptiveState::default();

        println!("Preheating (ADI 2-D) ...");
        while constant_time < 7.0 * days {
            mgr.solve_timestep_adaptive(&mut adap_state, None)?;
            mgr.tick_timestep()?;

            let mats = || {
                mgr.surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            };
            let min_rh = mats().map(|c| c.cell_rh).reduce(|a, b| a.min(b)).unwrap_or_default();
            let max_rh = mats().map(|c| c.cell_rh).reduce(|a, b| a.max(b)).unwrap_or_default();
            let min_t = mats().map(|c| c.cell_temp).reduce(|a, b| a.min(b)).unwrap_or_default();
            let max_t = mats().map(|c| c.cell_temp).reduce(|a, b| a.max(b)).unwrap_or_default();
            let prev_delta_t: Temperature<f64> = (max_t - min_t).abs();
            let prev_delta_rh: Dimensionless<f64> = (max_rh - min_rh).abs();

            if prev_delta_t.value_in(kelvin) <= allowed_err.value()
                && prev_delta_rh.value() <= allowed_err.value()
            {
                constant_time += mgr.settings.timestep;
            } else {
                constant_time = 0.0 * hours;
            }
        }
        println!("Preheat complete.");
    }

    // --- Simulation run with the EN 15026 step-change BC. ---
    mgr.boundary_condition = boundary_condition;

    let end_time: Time<f64> = *checkpoints.last().unwrap() + 1.0 * hours;
    let mut adap_state: AdaptiveState<()> = AdaptiveState::default();
    let mut current_time: Time<f64> = 0.0 * hours;
    let mut hit: HashSet<i32> = HashSet::new();

    while current_time <= end_time {
        mgr.solve_timestep_adaptive(&mut adap_state, None)?;
        mgr.tick_timestep()?;
        current_time = adap_state.current_time;

        if let Some(cp) = checkpoints.iter().find(|duration| {
            (duration.value_in(hours) - current_time.value_in(hours)).abs()
                <= mgr.settings.timestep.value_in(hours) * 0.5
        }) {
            let sim_day = cp.value_in(days) as i32;
            if !hit.insert(sim_day) {
                continue;
            }
            println!("Checkpoint reached: {} days", sim_day);
            let fails = compare_2d(&mgr.surface_next, sim_day);
            if fails > 0 {
                return Err(anyhow::format_err!(
                    "ADI 2-D vs 1-D EN 15026 mismatch at day {}: {} failing comparisons \
                     (per-row T/water or cross-row uniformity). The 2-D ADI path does NOT \
                     reduce to the 1-D benchmark.",
                    sim_day,
                    fails
                ));
            }
        }
    }

    println!("ADI 2-D uniform grid matches the 1-D EN 15026 benchmark on all rows.");
    Ok(())
}
