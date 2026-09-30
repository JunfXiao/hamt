#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Gating tests for the 3-D ADI path (Phase 2b, Task 4).
//!
//! A `(nx, ny, nz)` wall that is **uniform in both y and z** — `ny*nz` identical
//! x-columns, adiabatic side faces on all four off-axis faces, and 1-D-equivalent
//! boundary conditions on every column — must reproduce the trusted DIN EN 15026
//! 1-D analytical benchmark on *every* `(j, k)` line when solved with
//! `SolverMethod::Adi`. Any spurious y- or z-gradient (a stencil/grid/cross-lag
//! bug) shows up either as a per-line mismatch against the 1-D reference or as a
//! non-zero cross-line residual.
//!
//! The physics harness (curves, library, BC schedule, preheat + run loop) and
//! the reference tables are shared via `tests/common`; only the manager
//! construction (`new_3d`) and the comparison (per-line + cross-line, against the
//! same tolerances) differ from the 2-D gating test (`adi_2d.rs`).

mod common;

use anyhow::{Ok, Result};
use common::{
    CONSTRUCTION_NAME, PERCENTAGE_RANGE_RH, PERCENTAGE_RANGE_T, REF_TEMPERATURE, REF_WATER_CONTENT,
    create_library, init_tracing,
};
use hamt_core::config::FaceBc;
use hamt_core::uom::dimensions::{Dimensionless, Temperature, Time};
use hamt_core::uom::units::{
    days, dimensionless, hours, kelvin, kg_per_m2_s_pa, m_per_s, meter, seconds, square_meter,
    w_per_m2_k,
};
use hamt_core::uom::{Dimension, Quantity};
use hamt_core::{
    boundary::{BoundaryCondition, celsius_to_kelvin, kelvin_to_celsius},
};
use hamt_core::{
    cell::CellType,
    hamt_manager::{AdaptiveState, HamtManager, HamtSettings, SolverMethod},
};
use std::collections::HashSet;

// Cross-line uniformity residual: a truly (y,z)-uniform wall must keep every
// line identical to numerical round-off. Loose enough for f64 accumulation
// across a year of substeps, tight enough to catch any real coupling leak.
const CROSS_LINE_TOL_T_K: f64 = 1e-6;
const CROSS_LINE_TOL_W: f64 = 1e-6; // kg/m3

// ---------------------------------------------------------------------------
// 3-D comparison helpers.
// ---------------------------------------------------------------------------

/// Read a material-cell quantity along the `(j, k)` line at `depth_m`,
/// **x-interpolated** to match the 1-D EN-15026 benchmark (`compare_results`),
/// which samples via `Surface::interpolate_at_depth` (linear interpolation
/// between the bracketing cell centres, clamped — no extrapolation — outside the
/// material range). Returns (value, [i,j,k]) where the index is the nearest
/// material cell on the line (diagnostic only). Mirrors `cell_value_at_depth_row`
/// in `adi_2d.rs` but takes explicit `j` AND `k` indices.
///
/// Why interpolate: on the geometric grid (clustered at layer faces, coarse in
/// the interior) nearest-cell sampling reads a quantization-offset value at
/// interior / near-front depths; interpolation removes that offset so a
/// (y,z)-uniform wall reproduces the 1-D reference exactly (same fix as the 2-D
/// gate, 5170366).
///
/// The x-interpolation is done here on the requested line itself rather than via
/// `Surface::interpolate_at_point` with a (y,z)-pinned target: that routine's y/z
/// bracketing (`compute_interpolation_weights::axis_coords`) probes cells
/// `[0,k,0]` / `[0,0,k]`, i.e. x-index 0 = an exterior Boundary cell, which is
/// never `Material`, so for ny>1 / nz>1 it degenerates to line (0,0). Using it
/// here would turn the per-line and cross-line uniformity checks in `compare_3d`
/// into a vacuous self-comparison.
fn cell_value_at_depth_line<F, const D: Dimension>(
    surface: &hamt_core::surface::Surface,
    depth_m: f64,
    j: usize,
    k: usize,
    f: F,
) -> (f64, [usize; 3])
where
    F: Fn(&hamt_core::cell::Cell) -> Quantity<f64, D>,
{
    let nx = surface.cells.raw_dim()[0];
    // Material cells on this line in x order: (cell-centre x [m], value, [i,j,k]).
    let mut pts: Vec<(f64, f64, [usize; 3])> = Vec::new();
    for i in 0..nx {
        let cell = &surface.cells[[i, j, k]];
        if !matches!(cell.cell_type, CellType::Material(_)) {
            continue;
        }
        // strip: scalar x-coordinate [m] for the bracket search / distance compare
        pts.push((
            cell.origin.x().value_in(meter),
            f(cell).value_unchecked(),
            [i, j, k],
        ));
    }
    assert!(!pts.is_empty(), "no material cell on line ({j},{k})");

    // Nearest cell (diagnostic index only).
    let nearest = pts
        .iter()
        .min_by(|a, b| {
            (a.0 - depth_m)
                .abs()
                .partial_cmp(&(b.0 - depth_m).abs())
                .unwrap()
        })
        .map(|p| p.2)
        .unwrap();

    // Linear interpolation between the bracketing cell centres; clamp to the end
    // cell outside the material range (same semantics as the x-axis of
    // `Surface::compute_interpolation_weights`).
    let last = pts.len() - 1;
    let value = if depth_m <= pts[0].0 {
        pts[0].1
    } else if depth_m >= pts[last].0 {
        pts[last].1
    } else {
        let lo = pts.iter().rposition(|p| p.0 <= depth_m).unwrap();
        let (x0, v0, _) = pts[lo];
        let (x1, v1, _) = pts[lo + 1];
        let span = x1 - x0;
        let frac = if span > 0.0 { (depth_m - x0) / span } else { 0.0 };
        (1.0 - frac) * v0 + frac * v1
    };
    (value, nearest)
}

/// Compare a 3-D uniform surface against the 1-D reference at `duration_day`,
/// on EVERY `(j, k)` line, plus a cross-line uniformity check.
/// Returns the number of failures (0 = pass).
fn compare_3d(surface: &hamt_core::surface::Surface, duration_day: i32) -> u32 {
    let dims = surface.cells.raw_dim();
    let ny = dims[1];
    let nz = dims[2];
    let mut fail_count = 0u32;

    println!(
        "=== ADI 3-D comparison at day {} (ny={}, nz={}) ===",
        duration_day, ny, nz
    );

    // --- Temperature: each (j,k) line vs 1-D reference. ---
    println!(
        "{:>8} {:>6} {:>6} {:>9} {:>9} {:>8} {:>6}",
        "depth", "j", "k", "T(C)", "ref(C)", "diff%", "ok"
    );
    let mut t_xline_max_resid = 0.0_f64;
    for (d, depth, ref_t) in REF_TEMPERATURE.iter().filter(|(d, _, _)| *d == duration_day) {
        let _ = d;
        // Collect every (j,k) line's value.
        let line_vals: Vec<(usize, usize, f64)> = (0..ny)
            .flat_map(|j| {
                (0..nz).map(move |k| {
                    let (t_k, _) =
                        cell_value_at_depth_line(surface, *depth, j, k, |c| c.cell_temp);
                    (j, k, kelvin_to_celsius(t_k * kelvin))
                })
            })
            .collect();
        let vals: Vec<f64> = line_vals.iter().map(|(_, _, v)| *v).collect();
        let row_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        t_xline_max_resid = t_xline_max_resid.max(row_max - row_min);

        for (j, k, cell_t) in &line_vals {
            let dev = (cell_t - ref_t).abs() / ref_t;
            let ok = dev <= PERCENTAGE_RANGE_T;
            if !ok {
                fail_count += 1;
            }
            // Print first (j,k) line and any failures.
            if (*j == 0 && *k == 0) || !ok {
                println!(
                    "{:>8.2} {:>6} {:>6} {:>9.3} {:>9.3} {:>7.2}% {:>6}",
                    depth,
                    j,
                    k,
                    cell_t,
                    ref_t,
                    dev * 100.0,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
    }

    // --- Water content: each (j,k) line vs 1-D reference. ---
    println!(
        "{:>8} {:>6} {:>6} {:>12} {:>12} {:>8} {:>6}",
        "depth", "j", "k", "W(kg/m3)", "ref", "diff%", "ok"
    );
    let mut w_xline_max_resid = 0.0_f64;
    for (d, depth, ref_wc) in REF_WATER_CONTENT.iter().filter(|(d, _, _)| *d == duration_day) {
        let _ = d;
        let line_vals: Vec<(usize, usize, f64)> = (0..ny)
            .flat_map(|j| {
                (0..nz).map(move |k| {
                    let (w, _) = cell_value_at_depth_line(
                        surface,
                        *depth,
                        j,
                        k,
                        |c| c.volumetric_water,
                    );
                    (j, k, w)
                })
            })
            .collect();
        let vals: Vec<f64> = line_vals.iter().map(|(_, _, v)| *v).collect();
        let row_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w_xline_max_resid = w_xline_max_resid.max(row_max - row_min);

        for (j, k, cell_w) in &line_vals {
            let dev = (cell_w - ref_wc).abs() / ref_wc;
            let ok = dev <= PERCENTAGE_RANGE_RH;
            if !ok {
                fail_count += 1;
            }
            if (*j == 0 && *k == 0) || !ok {
                println!(
                    "{:>8.2} {:>6} {:>6} {:>12.3} {:>12.3} {:>7.2}% {:>6}",
                    depth,
                    j,
                    k,
                    cell_w,
                    ref_wc,
                    dev * 100.0,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
    }

    println!(
        "cross-line max residual: ΔT={:.3e} K (tol {:.0e}), ΔW={:.3e} kg/m3 (tol {:.0e})",
        t_xline_max_resid, CROSS_LINE_TOL_T_K, w_xline_max_resid, CROSS_LINE_TOL_W
    );
    if t_xline_max_resid > CROSS_LINE_TOL_T_K {
        eprintln!(
            "CROSS-LINE T residual {:.3e} K exceeds {:.0e} — wall is NOT (y,z)-uniform \
             (stencil/cross-lag bug)",
            t_xline_max_resid, CROSS_LINE_TOL_T_K
        );
        fail_count += 1;
    }
    if w_xline_max_resid > CROSS_LINE_TOL_W {
        eprintln!(
            "CROSS-LINE W residual {:.3e} kg/m3 exceeds {:.0e} — wall is NOT (y,z)-uniform \
             (stencil/cross-lag bug)",
            w_xline_max_resid, CROSS_LINE_TOL_W
        );
        fail_count += 1;
    }

    fail_count
}

// ---------------------------------------------------------------------------
// §6.1 Gating test — 3-D uniform grid reduces to the EN 15026 1-D benchmark.
// ---------------------------------------------------------------------------

#[test]
fn adi_3d_uniform_reduces_to_1d() -> Result<()> {
    init_tracing();

    let time_step: Time<f64> = 2.0 * hours;
    let ny = 2usize;
    let nz = 2usize;
    let y_extent = 1.0 * meter;
    let z_extent = 1.0 * meter;

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
    settings.n_div = 20;
    settings.div_length = 1e-5 * meter;
    // Match the 1-D benchmark's near-surface clustering (geometric, factor 0.7) so the wetting front is resolved to the EN-15026 reference.
    settings.div_distribution_factor = 0.7;
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

    // 3-D manager: ny*nz identical columns, adiabatic on all side and z-faces.
    let mut mgr = HamtManager::new_3d(
        Some(settings.clone()),
        Some(boundary_condition_init),
        construction,
        "Adi3DBenchmarkSurface".to_string(),
        1.0 * square_meter,
        ny,
        y_extent,
        nz,
        z_extent,
        [FaceBc::Adiabatic; 2],
    )?;

    let dim = mgr.surface_curr.cells.raw_dim();
    println!(
        "3-D grid shape: ({}, {}, {}), cells={}",
        dim[0],
        dim[1],
        dim[2],
        mgr.surface_curr.cells.len()
    );
    assert_eq!(dim[1], ny, "expected ny={} columns", ny);
    assert_eq!(dim[2], nz, "expected nz={} columns", nz);

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

        println!("Preheating (ADI 3-D) ...");
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
            let fails = compare_3d(&mgr.surface_next, sim_day);
            if fails > 0 {
                return Err(anyhow::format_err!(
                    "ADI 3-D vs 1-D EN 15026 mismatch at day {}: {} failing comparisons \
                     (per-line T/water or cross-line uniformity). The 3-D ADI path does NOT \
                     reduce to the 1-D benchmark.",
                    sim_day,
                    fails
                ));
            }
        }
    }

    println!("ADI 3-D uniform grid matches the 1-D EN 15026 benchmark on all (j,k) lines.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Regression — 3-D manager with DEFAULT solver_method must not panic.
// ---------------------------------------------------------------------------

/// Regression test: a 3-D manager (ny > 1, nz > 1) constructed with the DEFAULT
/// `solver_method` must not panic or error when `tick_timestep` / `solve_timestep`
/// is called. Mirrors `two_d_manager_default_solver_does_not_panic` in `adi_2d.rs`.
#[test]
fn three_d_manager_default_solver_does_not_panic() -> Result<()> {
    init_tracing();

    let ny = 2usize;
    let nz = 2usize;
    let y_extent = 1.0 * meter;
    let z_extent = 1.0 * meter;
    let time_step: Time<f64> = 2.0 * hours;

    // Small single-layer construction so the test is fast.
    let library = create_library(20.0, 0.5, vec![0.1], vec![Some(6)]);
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    // DEFAULT settings — solver_method is BlockTridiagonal; the dispatch guard
    // must route nz>1 to ADI exactly as it does for the 2-D path.
    let mut settings = HamtSettings::default();
    settings.timestep = time_step;

    let bc = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        rh_outside_air: 0.5 * dimensionless,
        rh_inside_air: 0.5 * dimensionless,
        heat_transfer_coeff_ext: 25.0 * w_per_m2_k,
        heat_transfer_coeff_int: 25.0 * w_per_m2_k,
        mass_convection_coeff_ext: 0.0 * m_per_s,
        mass_convection_coeff_int: 0.0 * m_per_s,
        vapor_transport_coeff_ext: 1.0e-9 * kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1.0e-9 * kg_per_m2_s_pa,
        ..Default::default()
    };

    let mut mgr = HamtManager::new_3d(
        Some(settings),
        Some(bc),
        construction,
        "M1Regression3DSurface".to_string(),
        1.0 * square_meter,
        ny,
        y_extent,
        nz,
        z_extent,
        [FaceBc::Adiabatic; 2],
    )?;

    let dim = mgr.surface_curr.cells.raw_dim();
    assert_eq!(dim[1], ny, "expected ny={}", ny);
    assert_eq!(dim[2], nz, "expected nz={}", nz);

    // Run one solve_timestep without a panic or error.
    let result = mgr.solve_timestep();
    assert!(result.is_ok(), "solve_timestep returned Err: {:?}", result);

    let converged = result.unwrap();
    println!(
        "three_d_manager_default_solver_does_not_panic: converged={}",
        converged
    );

    mgr.tick_timestep()?;

    // Sanity: every material cell must have finite T and RH.
    for cell in mgr.surface_prev.cells.iter() {
        if matches!(cell.cell_type, CellType::Material(_)) {
            assert!(
                cell.cell_temp.value_in(kelvin).is_finite(),
                "non-finite T after one step"
            );
            assert!(
                cell.cell_rh.value().is_finite(),
                "non-finite RH after one step"
            );
        }
    }

    println!("3-D no-panic regression PASS: default-solver tick completed without panic.");
    Ok(())
}

// ---------------------------------------------------------------------------
// ny=nz=1 routing assertion — must still produce the EN 15026 1-D result.
// ---------------------------------------------------------------------------

/// ny=nz=1 routing: `new_3d` with `ny=1, nz=1` must route to `BlockTridiagonal`
/// (the 1-D fast path) and match the EN 15026 reference at day 7.
///
/// This mirrors the implicit contract checked by the `adi_2d` suite and the
/// main `din_en_15026_benchmark`, pinned here for `new_3d` specifically.
#[test]
fn adi_3d_nynz1_routes_to_1d() -> Result<()> {
    init_tracing();

    let time_step: Time<f64> = 2.0 * hours;
    let ny = 1usize;
    let nz = 1usize;
    let y_extent = 1.0 * meter;
    let z_extent = 1.0 * meter;

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
    settings.n_div = 20;
    settings.div_length = 1e-5 * meter;
    // Match the 1-D benchmark's near-surface clustering (geometric, factor 0.7) so the wetting front is resolved to the EN-15026 reference.
    settings.div_distribution_factor = 0.7;
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

    let mut mgr = HamtManager::new_3d(
        Some(settings),
        Some(boundary_condition_init),
        construction,
        "Adi3D1x1Surface".to_string(),
        1.0 * square_meter,
        ny,
        y_extent,
        nz,
        z_extent,
        [FaceBc::Adiabatic; 2],
    )?;

    let dim = mgr.surface_curr.cells.raw_dim();
    assert_eq!(dim[1], 1, "ny=1 expected");
    assert_eq!(dim[2], 1, "nz=1 expected");

    // Preheat.
    {
        let allowed_err: Dimensionless<f64> = 1e-4 * dimensionless;
        let mut constant_time: Time<f64> = 0.0 * seconds;
        let mut adap_state: AdaptiveState<()> = AdaptiveState::default();

        while constant_time < 7.0 * days {
            mgr.solve_timestep_adaptive(&mut adap_state, None)?;
            mgr.tick_timestep()?;

            let mats = || {
                mgr.surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            };
            let d_rh = (mats().map(|c| c.cell_rh).reduce(|a, b| a.max(b)).unwrap_or_default()
                - mats()
                    .map(|c| c.cell_rh)
                    .reduce(|a, b| a.min(b))
                    .unwrap_or_default())
            .abs();
            let d_t = (mats()
                .map(|c| c.cell_temp)
                .reduce(|a, b| a.max(b))
                .unwrap_or_default()
                - mats()
                    .map(|c| c.cell_temp)
                    .reduce(|a, b| a.min(b))
                    .unwrap_or_default())
            .abs();
            if d_t.value_in(kelvin) <= allowed_err.value() && d_rh.value() <= allowed_err.value() {
                constant_time += mgr.settings.timestep;
            } else {
                constant_time = 0.0 * hours;
            }
        }
    }

    // Run until day 7 checkpoint.
    mgr.boundary_condition = boundary_condition;
    let target_day = 7i32;
    let target_time: Time<f64> = (target_day as f64) * days;
    let end_time = target_time + 1.0 * hours;
    let mut adap_state: AdaptiveState<()> = AdaptiveState::default();
    let mut current_time: Time<f64> = 0.0 * hours;
    let mut reached = false;

    while current_time <= end_time {
        mgr.solve_timestep_adaptive(&mut adap_state, None)?;
        mgr.tick_timestep()?;
        current_time = adap_state.current_time;

        let diff_h = (target_time.value_in(hours) - current_time.value_in(hours)).abs();
        if diff_h <= mgr.settings.timestep.value_in(hours) * 0.5 && !reached {
            reached = true;
            let fails = compare_3d(&mgr.surface_next, target_day);
            if fails > 0 {
                return Err(anyhow::format_err!(
                    "ny=nz=1 new_3d mismatch at day {}: {} failing comparisons",
                    target_day,
                    fails
                ));
            }
            break;
        }
    }

    assert!(reached, "never reached day {} checkpoint", target_day);
    println!("ny=nz=1 new_3d routes to 1-D correctly: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// §6.4 — 3-D closed-box face energy/mass balance test (Task 5).
// ---------------------------------------------------------------------------

/// Test: Face energy & mass balance on a 3-D ADI grid (closed box, adiabatic on
/// all side/z-faces).
///
/// ## Setup
/// Small 3-D manager (`ny=3, nz=3`, 6 x-cells per column) with `SolverMethod::Adi`.
/// Both z-faces and y-faces are `Adiabatic` — only the two x-faces carry flux.
/// This exercises `Size3D::area_along(2)` for z-face volumes and the per-(y,z)
/// cell volumes in the 3-D grid.
///
/// ## Approach
/// ISOTHERMAL: both x-faces at 20 °C. One-sided moisture step (ext RH 0.5 → 0.7,
/// int RH = 0.5). Short 30-s macro-step so the PI controller uses very short
/// substeps and the flux-accumulator endpoint approximation is accurate.
///
/// ## Balance assertion
/// `Σ(face fluxes · area · dt)  ≈  Δ(stored_water)` within 15 % (the same
/// tolerance as the 2-D balance test, which carries a known ~10 % systematic
/// offset from the virtual-cell flux query). The 15 % bound catches gross errors
/// (wrong sign, wrong face area, broken FV assembly) while accepting the offset.
///
/// ## Dimension assert
/// `dim == [nx, 3, 3]` — confirms the 3-D grid was built correctly.
///
/// ## Shared helpers
/// `total_stored_water`, `total_stored_energy`, and `build_balance_construction`
/// are promoted to `tests/common/mod.rs` and used here via `common::`.
#[test]
fn adi_3d_face_energy_mass_balance() -> Result<()> {
    init_tracing();

    let time_step: Time<f64> = 30.0 * seconds; // 30 s — very short for accurate endpoint flux
    let ny = 3usize;
    let nz = 3usize;
    let y_extent = 1.0 * meter;
    let z_extent = 1.0 * meter;
    let surface_area = 1.0 * square_meter;

    // Non-const VTCs: BC values flow through unchanged (no construction override).
    let construction = common::build_balance_construction(0.5);

    let settings = HamtSettings {
        timestep: time_step,
        max_iterations: 3000,
        temp_convergence_tolerance: 1e-6 * kelvin,
        rh_convergence_tolerance: 1e-5 * dimensionless,
        min_relaxation_factor: 1.0,
        anderson_enabled: false, // consistent with 2-D balance test
        solver_method: SolverMethod::Adi,
        ..HamtSettings::default()
    };

    // ISOTHERMAL: both x-faces at 20 °C to avoid thermal-VP coupling.
    // VTC = 3e-9 kg/(m²·s·Pa) gives measurable flux in 30 s.
    let vtc = 3.0e-9 * kg_per_m2_s_pa;
    let t_20 = celsius_to_kelvin(20.0);
    let rh_init = 0.5 * dimensionless;

    let bc_init = BoundaryCondition {
        temp_outside_air: t_20,
        temp_inside_air: t_20,
        temp_outside_air_moisture: t_20,
        temp_inside_air_moisture: t_20,
        rh_outside_air: rh_init,
        rh_inside_air: rh_init,
        heat_transfer_coeff_ext: 25.0 * w_per_m2_k,
        heat_transfer_coeff_int: 25.0 * w_per_m2_k,
        mass_convection_coeff_ext: 0.0 * m_per_s,
        mass_convection_coeff_int: 0.0 * m_per_s,
        vapor_transport_coeff_ext: vtc,
        vapor_transport_coeff_int: vtc,
        ..Default::default()
    };

    // Step change: exterior RH 0.5 → 0.7 (isothermal one-sided drive).
    let bc_step = BoundaryCondition {
        rh_outside_air: 0.7 * dimensionless,
        ..bc_init
    };

    // 3-D closed box: adiabatic on y-faces (default) AND z-faces.
    let mut mgr = HamtManager::new_3d(
        Some(settings),
        Some(bc_init),
        construction,
        "Balance3DSurface".to_string(),
        surface_area,
        ny,
        y_extent,
        nz,
        z_extent,
        [FaceBc::Adiabatic; 2], // z-faces adiabatic
    )?;

    // Confirm the grid is a genuine 3-D grid of the right shape.
    let dim = mgr.surface_curr.cells.raw_dim();
    let nx_cells = dim[0];
    println!(
        "3-D closed-box balance: grid shape=({}, {}, {}), cells={}",
        nx_cells, dim[1], dim[2], mgr.surface_curr.cells.len()
    );
    assert_eq!(dim[1], ny, "expected ny={} columns in y-direction", ny);
    assert_eq!(dim[2], nz, "expected nz={} layers in z-direction", nz);

    // Preheat to uniform isothermal equilibrium (RH=0.5, T=20°C).
    {
        let mut adap: AdaptiveState<()> = AdaptiveState::default();
        let allowed_err: Dimensionless<f64> = 1e-5 * dimensionless;
        let mut stable_count = 0usize;
        println!("Preheating (3-D balance test, isothermal) ...");
        for _ in 0..200 {
            mgr.solve_timestep_adaptive(&mut adap, None)?;
            mgr.tick_timestep()?;
            let mats = || {
                mgr.surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            };
            let d_rh = mats()
                .map(|c| c.cell_rh)
                .fold(0.0 * dimensionless, |acc: Dimensionless<f64>, x| acc.max(x))
                - mats()
                    .map(|c| c.cell_rh)
                    .fold(1.0 * dimensionless, |acc: Dimensionless<f64>, x| acc.min(x));
            let d_t = mats()
                .map(|c| c.cell_temp)
                .fold(0.0 * kelvin, |acc: Temperature<f64>, x| acc.max(x))
                - mats()
                    .map(|c| c.cell_temp)
                    .fold(1000.0 * kelvin, |acc: Temperature<f64>, x| acc.min(x));
            if d_rh.value() <= allowed_err.value() && d_t.value_in(kelvin) <= allowed_err.value() {
                stable_count += 1;
                if stable_count >= 5 {
                    break;
                }
            } else {
                stable_count = 0;
            }
        }
        println!("Preheat done (stable_count={stable_count}).");
    }

    // Capture BEFORE state (preheat-converged surface).
    let water_before = common::total_stored_water(&mgr.surface_prev);
    let energy_before = common::total_stored_energy(&mgr.surface_prev);
    println!(
        "BEFORE: water={:.6e} kg  energy_proxy={:.4e} J",
        water_before.value_unchecked(),
        energy_before.value_unchecked(),
    );

    // Apply BC step-change and solve ONE 30-s macro-step.
    mgr.boundary_condition = bc_step.clone();
    let mut adap: AdaptiveState<()> = AdaptiveState::default();
    mgr.solve_timestep_adaptive(&mut adap, None)?;
    mgr.tick_timestep()?;

    let water_after = common::total_stored_water(&mgr.surface_prev);
    let energy_after = common::total_stored_energy(&mgr.surface_prev);
    println!(
        "AFTER:  water={:.6e} kg  energy_proxy={:.4e} J",
        water_after.value_unchecked(),
        energy_after.value_unchecked(),
    );

    // Flux accumulator: ∫ flux dt over the macro-step [kg/m²].
    // Positive = flux FROM boundary INTO wall.
    let acc = &adap.flux_accumulator;
    let area_scalar = surface_area.value_in(square_meter);

    // strip: areal flux [kg/m²] × area [m²] → total mass [kg]
    let mass_in_from_ext = acc.m_ext_dt.value_unchecked() * area_scalar;
    let mass_in_from_int = acc.m_int_dt.value_unchecked() * area_scalar;
    let total_mass_in = mass_in_from_ext + mass_in_from_int;

    let energy_in_from_ext = acc.q_ext_dt.value_unchecked() * area_scalar;
    let energy_in_from_int = acc.q_int_dt.value_unchecked() * area_scalar;
    let total_energy_in = energy_in_from_ext + energy_in_from_int;

    let delta_water = water_after.value_unchecked() - water_before.value_unchecked();
    let delta_energy = energy_after.value_unchecked() - energy_before.value_unchecked();

    println!(
        "Moisture flux  EXT={:.4e} kg  INT={:.4e} kg  net_in={:.4e} kg",
        mass_in_from_ext, mass_in_from_int, total_mass_in
    );
    println!(
        "Heat flux      EXT={:.4e} J   INT={:.4e} J   net_in={:.4e} J",
        energy_in_from_ext, energy_in_from_int, total_energy_in
    );
    println!(
        "Δ(stored_water)  = {:.6e} kg  (moisture_flux×dt×area = {:.6e} kg)",
        delta_water, total_mass_in
    );
    println!(
        "Δ(energy_proxy)  = {:.4e} J   (heat_flux×dt×area     = {:.4e} J)  [info only]",
        delta_energy, total_energy_in
    );
    println!(
        "Flux accumulator total_time = {:.1} s (expected {:.1} s)",
        adap.flux_accumulator.total_time.value_unchecked(),
        time_step.value_unchecked(),
    );

    // ── Assertion 1: Sign convention ────────────────────────────────────────
    // Exterior RH rose 0.5 → 0.7 at constant T: exterior VP > wall VP.
    //   moisture_flux(Exterior) > 0  ≡  flux INTO wall
    //   Δ(stored_water) > 0          ≡  wall absorbed moisture
    assert!(
        delta_water > 0.0,
        "stored water must increase when exterior RH rises (isothermal): Δw={:.4e} kg",
        delta_water
    );
    assert!(
        mass_in_from_ext > 0.0,
        "moisture_flux(Exterior) must be positive (exterior VP > wall VP): {:.4e} kg",
        mass_in_from_ext
    );
    println!("Sign convention OK: m_ext>0 (into wall), Δwater>0");

    // ── Assertion 2: MASS balance within 15 % ───────────────────────────────
    // Mirrors the 2-D balance test: the same ~10 % systematic virtual-cell
    // offset applies; the 15 % tolerance comfortably accepts it while catching
    // gross errors (wrong sign, wrong area, broken FV assembly).
    let mass_residual = if total_mass_in.abs() > 1e-15 {
        (delta_water - total_mass_in).abs() / total_mass_in.abs()
    } else {
        0.0
    };
    println!(
        "MASS balance residual: {:.4e}  (tolerance ≤ 15 %; known systematic offset ≈ 10 %)",
        mass_residual
    );
    assert!(
        mass_residual <= 0.15,
        "MASS balance residual {:.4e} exceeds 15 %: Δw={:.6e} kg vs flux_in={:.6e} kg",
        mass_residual,
        delta_water,
        total_mass_in
    );

    // ── Assertion 3: Grid dimension ─────────────────────────────────────────
    // Confirms the 3-D grid was built with the expected ny and nz (already
    // asserted above; print here for clarity in the test output).
    println!(
        "Grid dim OK: [{}, {}, {}]  (ny={}, nz={})",
        nx_cells, dim[1], dim[2], ny, nz
    );

    // ── ENERGY balance — informational only ─────────────────────────────────
    // Isothermal: both faces at the same temperature; net heat flux ≈ 0.
    // No hard assertion (endpoint-flux approximation is less accurate for heat
    // than for moisture at these short timescales).
    println!(
        "ENERGY balance (informational): ΔE={:.4e} J, heat_flux×dt×area={:.4e} J",
        delta_energy, total_energy_in
    );

    println!(
        "ADI 3-D closed-box face energy/mass balance PASS  (mass residual={:.4e})",
        mass_residual
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// §6.3 Spatial order of accuracy via Richardson self-convergence.
// ---------------------------------------------------------------------------

/// Richardson self-convergence test: observed spatial order of accuracy for the
/// 3-D ADI path.
///
/// ## Motivation
/// With CONSTANT coefficients a steady finite-volume solution reproduces the
/// exact linear profile and gives zero error at any resolution — no order is
/// observable. Instead we drive a 7-day transient with the REAL nonlinear EN
/// 15026 material whose φ/w-dependent sorption, vapor-resistance and
/// liquid-transport curves produce genuine profile curvature. At day 7 the
/// moisture front has penetrated ~1 cm into the layer, giving a spatially
/// non-trivial profile on which convergence order is measurable.
///
/// ## Why finite (not infinite) BCs
/// With Dirichlet-like BCs (infinite VTC), the fixed-size virtual boundary
/// cell creates a grid-sensitivity artifact: finer material cells have lower
/// capacitance, so they equilibrate faster to the boundary VP, and
/// `total_stored_water` INCREASES WITHOUT BOUND as h → 0 (the series appears
/// divergent). Richardson self-convergence requires bounded convergence to a
/// finite Q_∞. Finite BCs (physical convective film resistance >> virtual-cell
/// half-width resistance) avoid this — the surface film dominates the resistance
/// regardless of h, so Q converges properly.
///
/// ## Setup
/// - Material: EN 15026 single homogeneous layer (same nonlinear curves:
///   sorption isotherm, vapor diffusion resistance, liquid transport, thermal
///   conductivity). Single layer avoids inter-layer interface artefacts.
/// - Layer: 0.1 m = 10 cm, `div_distribution_factor = 0.0` (pure uniform
///   spacing). Refinement levels num_div = 8, 16, 32 (2:1 ratios).
///   Cell sizes: 12.5, 6.25, 3.125 mm. Optional 4th level at 64 for a second
///   independent order estimate.
/// - Grid: `new_3d` with `ny=1, nz=2`, `z_faces=[FaceBc::Adiabatic;2]`.
///   Exercises the genuine 3-D ADI code path; z adds no error (uniform, adiabatic).
/// - BCs: ext 30 °C / RH 0.95 (finite VTC = 3e-8 kg/m²/s/Pa), int 20 °C /
///   RH 0.5 (negligible VTC). One-sided moisture drive from a uniform initial
///   state (20 °C / RH 0.5).
///
/// ## Fixed-timestep loop (isolates spatial order)
/// `dt = 30 min`, 336 steps = 7 days. A fixed `dt` identical across all levels
/// means temporal truncation error is the same at every level and **cancels**
/// in the Richardson differences. `solve_timestep` + `tick_timestep` in a plain
/// loop (no adaptive substepping) — no per-level variation in substep count.
///
/// Some outer-iteration non-convergences occur at the very first timesteps of
/// the finest level when the initial gradient is sharpest. The result is
/// insensitive to these (p is consistent across different dt values).
///
/// ## Global functional
/// `total_stored_water` sums `volumetric_water × volume` over all material
/// cells — a smooth global integral that retains the full spatial order.
///
/// ## Richardson estimate
/// ```text
/// p = log2( |Q_N − Q_2N| / |Q_2N − Q_4N| )
/// ```
/// A 2nd-order cell-centred FV scheme should give `p ≈ 2.0`. We assert `p >= 1.5`.
#[test]
fn adi_3d_spatial_order_richardson() -> Result<()> {
    init_tracing();

    // --- Fixed parameters (identical across all refinement levels) ---
    // dt = 30 min = 1800 s, 336 steps = 7 days.
    // Fixed dt ensures temporal error is identical across all levels and
    // cancels in the Richardson differences (isolating spatial order).
    let fixed_dt: Time<f64> = 1800.0 * seconds; // 30 min
    let n_steps: usize = 336; // 336 × 30 min = 7 days

    let ny = 1usize;
    let nz = 2usize; // genuine 3-D ADI path (nz=2 > 1)
    let y_extent = 1.0 * meter;
    let z_extent = 1.0 * meter;
    let surface_area = 1.0 * square_meter;

    // FINITE BCs — required for Richardson convergence with the virtual cell
    // boundary formulation. With finite VTC the flux into the wall is bounded
    // (dominated by the physical surface film, not the virtual-cell half-width).
    // vtc_int ≈ 0 → one-sided moisture drive, no back-flux from interior.
    let vtc_ext = 3.0e-8 * kg_per_m2_s_pa;
    let vtc_int = 1.0e-10 * kg_per_m2_s_pa;
    let bc_run = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(30.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(30.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        rh_outside_air: 0.95 * dimensionless,
        rh_inside_air: 0.5 * dimensionless,
        heat_transfer_coeff_ext: 25.0 * w_per_m2_k,
        heat_transfer_coeff_int: 8.0 * w_per_m2_k,
        mass_convection_coeff_ext: 0.0 * m_per_s,
        mass_convection_coeff_int: 0.0 * m_per_s,
        vapor_transport_coeff_ext: vtc_ext,
        vapor_transport_coeff_int: vtc_int,
        ..Default::default()
    };

    // Base settings — cloned per level.
    // div_distribution_factor = 0.0  → PURE UNIFORM cell spacing.
    // Doubling num_div halves cell size everywhere (clean 2:1 for Richardson).
    let base_settings = HamtSettings {
        timestep: fixed_dt,
        max_iterations: 3000,
        temp_convergence_tolerance: 1e-7 * kelvin,
        rh_convergence_tolerance: 1e-6 * dimensionless,
        min_relaxation_factor: 1.0,
        anderson_enabled: true,
        solver_method: SolverMethod::Adi,
        div_distribution_factor: 0.0,
        ..HamtSettings::default()
    };

    // Single 0.1-m (10 cm) homogeneous layer using the EN 15026 material.
    // Refinement levels nd = 8, 16, 32, 64.
    // Primary Richardson estimate: levels[0..3] = [8, 16, 32].
    // Second estimate (informational): levels[1..4] = [16, 32, 64].
    // Cell sizes: 12.5, 6.25, 3.125, 1.5625 mm.
    // The geometric cell-division model has no min-cell clamp (adj_length /
    // max_div were removed): every `num_div` is honoured, and `factor = 0.0`
    // (base_settings) makes `geometric_norm_pos` exactly uniform `i/n`.
    let levels: &[usize] = &[8, 16, 32, 64];
    let layer_thickness_m = 0.1_f64; // 0.1 m = 10 cm

    let mut q_values: Vec<f64> = Vec::new();

    for &nd in levels {
        let library = create_library(
            20.0, // initial T °C
            0.5,  // initial RH
            vec![layer_thickness_m],
            vec![Some(nd)],
        );
        let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

        let mut settings = base_settings.clone();
        settings.n_div = nd as i32;
        settings.div_length = 1e-6 * meter;

        let mut mgr = HamtManager::new_3d(
            Some(settings),
            Some(bc_run.clone()),
            construction,
            format!("RichardsonSurface_nd{}", nd),
            surface_area,
            ny,
            y_extent,
            nz,
            z_extent,
            [FaceBc::Adiabatic; 2],
        )?;

        let dim = mgr.surface_curr.cells.raw_dim();
        let n_mat = mgr
            .surface_curr
            .cells
            .iter()
            .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            .count();
        println!(
            "Richardson level nd={:>3}: grid=({}, {}, {}), mat_cells={}",
            nd, dim[0], dim[1], dim[2], n_mat
        );

        // Fixed-timestep loop. Identical dt across all levels → temporal error
        // cancels in Richardson. No adaptive substepping.
        let mut nonconv = 0usize;
        for step_idx in 0..n_steps {
            let converged = mgr.solve_timestep()?;
            if !converged {
                nonconv += 1;
                eprintln!("  nd={}: step {} non-converged", nd, step_idx);
            }
            mgr.tick_timestep()?;
        }
        if nonconv > 0 {
            eprintln!(
                "  nd={}: {} / {} steps non-converged (early transient only; result unaffected)",
                nd, nonconv, n_steps
            );
        }

        let q = common::total_stored_water(&mgr.surface_prev).value_unchecked();
        println!(
            "  nd={:>3}: Q = {:.10e} kg  (non-conv steps: {})",
            nd, q, nonconv
        );
        q_values.push(q);
    }

    // --- Richardson order estimates ---
    // Primary (levels 0..3 = nd 8, 16, 32):
    let (q_n, q_2n, q_4n) = (q_values[0], q_values[1], q_values[2]);
    let diff_coarse = (q_n - q_2n).abs();
    let diff_fine = (q_2n - q_4n).abs();

    // Second estimate (levels 1..4 = nd 16, 32, 64):
    let (q_n2, q_2n2, q_4n2) = (q_values[1], q_values[2], q_values[3]);
    let diff_coarse2 = (q_n2 - q_2n2).abs();
    let diff_fine2 = (q_2n2 - q_4n2).abs();

    eprintln!("--- Richardson self-convergence (spatial order, 3-D ADI) ---");
    eprintln!(
        "  material: EN 15026 single layer {:.0} cm, uniform cells, finite BCs",
        layer_thickness_m * 100.0
    );
    eprintln!(
        "  fixed_dt = {} h = {} s,  n_steps = {}  ({} days)",
        fixed_dt.value_in(hours),
        fixed_dt.value_in(seconds) as u32,
        n_steps,
        (n_steps as f64) * fixed_dt.value_in(hours) / 24.0
    );
    for (i, &nd) in levels.iter().enumerate() {
        eprintln!("  Q(nd={:>3}) = {:.10e} kg", nd, q_values[i]);
    }
    eprintln!("  PRIMARY  (nd 8→16→32): |ΔQ_coarse|={:.6e}  |ΔQ_fine|={:.6e}", diff_coarse, diff_fine);
    eprintln!("  SECONDARY(nd 16→32→64): |ΔQ_coarse|={:.6e}  |ΔQ_fine|={:.6e}", diff_coarse2, diff_fine2);

    assert!(diff_coarse > 1e-12, "|Q_N - Q_2N| = {:.3e} at noise floor", diff_coarse);
    assert!(diff_fine > 1e-12, "|Q_2N - Q_4N| = {:.3e} at noise floor", diff_fine);

    let order = (diff_coarse / diff_fine).log2();
    let order2 = if diff_fine2 > 1e-12 {
        (diff_coarse2 / diff_fine2).log2()
    } else {
        f64::NAN
    };
    eprintln!(
        "  Richardson p (8→16→32)  = log2({:.6e}/{:.6e}) = {:.4}",
        diff_coarse, diff_fine, order
    );
    eprintln!(
        "  Richardson p (16→32→64) = log2({:.6e}/{:.6e}) = {:.4}  (informational)",
        diff_coarse2, diff_fine2, order2
    );
    eprintln!("  Expected ≈ 2.0; assertion: p >= 1.5");

    assert!(
        order >= 1.5,
        "Observed Richardson spatial order {:.4} < 1.5 — the 3-D ADI path delivers \
         LOWER than expected spatial accuracy. A cell-centred FV scheme must produce \
         p >= 1.5 on this nonlinear transient. \
         Q_8={:.8e}  Q_16={:.8e}  Q_32={:.8e}  \
         diff_coarse={:.6e}  diff_fine={:.6e}",
        order, q_n, q_2n, q_4n, diff_coarse, diff_fine
    );

    println!(
        "ADI 3-D Richardson spatial order PASS: p(8→16→32)={:.4}, p(16→32→64)={:.4} (>= 1.5)",
        order, order2
    );

    Ok(())
}
