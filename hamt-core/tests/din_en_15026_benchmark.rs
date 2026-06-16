#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Test suite for the DIN EN 15026 analytical benchmark (Annex B).

mod common;

use anyhow::{Ok, Result};
use common::{
    CONSTRUCTION_NAME, PERCENTAGE_RANGE_RH, PERCENTAGE_RANGE_T, REF_TEMPERATURE, REF_WATER_CONTENT,
    create_library, get_sorption_isotherm_curve, init_tracing, interpolate_data_points, rho_w,
    water_sat,
};
use crossterm::{
    ExecutableCommand, QueueableCommand, cursor,
    style::{self, Attribute, Color, Print},
    terminal::{self, ClearType},
};
use hamt_core::uom::dimensions::{Dimensionless, Temperature, Time};
use hamt_core::uom::units::{
    days, dimensionless, hours, j_per_kg_k, kelvin, kg_per_m2_s, kg_per_m2_s_pa, kg_per_m3,
    kilograms, m_per_s, m2_per_s, meter, pascals, square_meter, w_per_m2_k, w_per_m_k,
};
use hamt_core::{
    boundary::{BoundaryCondition, celsius_to_kelvin, kelvin_to_celsius},
    uom::units::seconds,
};
use hamt_core::{
    cell::CellType,
    hamt_manager::{AdaptiveState, HamtManager, HamtSettings, SolverMethod},
};
use hamt_core::{surface::Surface};
use std::time::{Duration, Instant};
use std::{
    borrow::Cow,
    io::{self, IsTerminal, Stdout, Write},
};
use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;

const DIAGNOSTIC_SAMPLE: bool = true;

fn run_benchmark_test(time_step: Time<f64>) -> Result<()> {
    run_benchmark_test_with_solver(time_step, SolverMethod::BlockJacobi)
}

fn run_benchmark_test_with_solver(time_step: Time<f64>, method: SolverMethod) -> Result<()> {
    let start_time = Instant::now();

    init_tracing();

    let ui = Rc::new(RefCell::new(Ui::new("DIN EN 15026 Benchmark")?));

    let result = (|| -> Result<()> {
        // Use 3 layers to optimize grid:
        // 1. Surface (0.1m): Fine grid for moisture profile
        // 2. Transition (1.0m): Intermediate grid
        // 3. Bulk (48.9m): Coarse grid for semi-infinite thermal mass
        //
        // The FIRST (0.1 m surface) layer is heavily subdivided (480 divisions) and,
        // together with `div_distribution_factor = 0.9` below, clusters cells tightly
        // at the wetted exterior face so the FIRST near-surface cell is ~22 µm. The EN
        // 15026 wetting front is concentrated in the first ~1 mm; resolving it requires
        // a sub-0.1 mm near-surface cell. The previously coarse mesh (60 divisions,
        // ~228 µm first cell) only passed because a spurious surface-LIQUID flux through
        // the air film — since removed by the EnergyPlus-style product guard in
        // calculate_resistance_triple (no liquid water crosses an unsaturated air film) —
        // was MASKING the under-resolution at depth 0.01 m. With that fix the mesh must
        // actually resolve the front. See doc/flux_offset_investigation.md.
        //
        // NOTE: the grid's safety clamp reduces divisions while the first cell is below
        // `settings.adj_length`; `adj_length` is lowered to 1e-5 m below so all 480
        // divisions survive (otherwise they collapse back to ~118 / ~100 µm).
        let library = create_library(
            20.0,
            0.5,
            vec![0.1, 0.9, 4.0, 10.0, 20.0],
            vec![Some(480), Some(60), Some(60), Some(60), Some(60)],
        );
        let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

        let mut settings = HamtSettings::default();
        settings.timestep = time_step;
        settings.max_iterations = 3000;
        settings.temp_convergence_tolerance = 1e-6 * kelvin;
        settings.rh_convergence_tolerance = 1e-4 * dimensionless;
        settings.max_div = 500;
        settings.min_div = 20;
        settings.div_length = 1e-5 * meter;
        // Cluster cells at the surface so the wetting front (first ~1 mm) is resolved.
        settings.div_distribution_factor = 0.9;
        // Lower the min-cell safety clamp so the requested 480 surface divisions are
        // NOT reduced (the clamp shrinks divs while the first cell < adj_length). With
        // factor 0.9 the first cell is ~22 µm, so adj_length must sit below that.
        settings.adj_length = 1e-5 * meter;
        settings.min_relaxation_factor = 1.0;
        settings.solver_method = method;

        let boundary_coeff = BoundaryCondition {
            heat_transfer_coeff_ext: 9e9 * w_per_m2_k,
            heat_transfer_coeff_int: 9e-9 * w_per_m2_k,
            mass_convection_coeff_ext: 9e9 * m_per_s,
            mass_convection_coeff_int: 9e-9 * m_per_s,
            vapor_transport_coeff_ext: 9e9 * kg_per_m2_s_pa,
            vapor_transport_coeff_int: 9e-9 * kg_per_m2_s_pa,
            ..Default::default()
        };

        // DIN EN 15026 specifies a single ambient state per side (heat and moisture
        // match). Set both *_air and *_air_moisture explicitly so the test doesn't
        // depend on Default::default() leaving a 20 °C moisture-side fallback.
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

        let mut mgr = HamtManager::new(
            Some(settings.clone()),
            Some(boundary_condition_init),
            construction,
            "BenchmarkSurface".to_string(),
            1.0 * square_meter,
        )
        .unwrap();

        ui.borrow_mut()
            .log(format!("Cell Count: {}", mgr.surface_curr.cells.len()))?;

        let mut simulation_durations_days: Cow<Vec<Time<f64>>> = Cow::Owned(
            REF_TEMPERATURE
                .iter()
                .chain(REF_WATER_CONTENT.iter())
                .map(|(d, _, _)| d)
                .collect::<HashSet<_>>()
                .into_iter()
                .map(|d| (*d as f64) * days)
                .collect(),
        );
        simulation_durations_days
            .to_mut()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        {
            ui.borrow_mut()
                .log(format!("Running benchmark with time step: {:?}", time_step))?;
            ui.borrow_mut()
                .log("Preheating to reach stable...".to_string())?;

            let mut prev_delta_t: Temperature<f64> = 0.0 * kelvin;
            let mut prev_delta_rh: Dimensionless<f64> = 0.0 * dimensionless;
            let mut constant_time: Time<f64> = 0.0 * seconds;
            let allowed_err: Dimensionless<f64> = 1e-4 * dimensionless;
            let mut adap_state = AdaptiveState::default();

            let preheat_speed_tracker = Rc::new(RefCell::new(SpeedTracker::new(10)));
            let preheat_end_hours: f64 = (7.0_f64 * days).value_in(hours);

            let make_progress_bar_preheat = {
                let ui_clone = ui.clone();
                let start_time_for_ui = Rc::new(start_time);
                let orig_timestep_for_ui = Rc::new(mgr.settings.timestep.clone());
                let tracker = preheat_speed_tracker.clone();
                move || {
                    let ui_clone = ui_clone.clone();
                    let start_time_for_ui = start_time_for_ui.clone();
                    let orig_timestep_for_ui = orig_timestep_for_ui.clone();
                    let tracker = tracker.clone();
                    move |state: &AdaptiveState<Vec<(Time<f64>, u32)>>| -> Result<()> {
                        let orig_timestep = (*orig_timestep_for_ui).clone();
                        let current_step =
                            state.proposed_dt.unwrap_or_else(|| orig_timestep.clone());
                        let sim_hours = state.current_time.value_in(hours);
                        let remaining_hours = preheat_end_hours - sim_hours;
                        tracker.borrow_mut().record(sim_hours);
                        let (speed, eta) = tracker.borrow().speed_and_eta(remaining_hours);
                        let dt_ratio = current_step.value_in(hours) / orig_timestep.value_in(hours);
                        ui_clone.borrow_mut().set_status(
                            format!(
                                "Preheat ΔT={:.2e} K, Δrh={:.2}% | Time={:.2} hours | Run time: {} seconds | {:.2} hr/s, ETA: {:.1} s | timestep: {:?} (orig: {:?})",
                                prev_delta_t.value_in(kelvin),
                                prev_delta_rh.value()*100.0,
                                sim_hours,
                                start_time_for_ui.elapsed().as_secs(),
                                speed,
                                eta,
                                current_step,
                                orig_timestep
                            ),
                            StatusStyle::from_dt_ratio(dt_ratio),
                        )
                    }
                }
            };
            while constant_time < 7.0 * days {
                // if !mgr.solve_timestep()? {
                //     return Err(hamt_core::error::Error::Other(
                //         "Preheating failed: timestep did not converge".to_string(),
                //     )
                //     .into());
                // }

                let set_progress_bar = make_progress_bar_preheat();
                mgr.solve_timestep_adaptive(&mut adap_state, Some(Box::new(set_progress_bar)))?;
                mgr.tick_timestep()?;
                // adaptive_state.current_time += mgr.settings.timestep;

                let min_rh = mgr
                    .surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, hamt_core::cell::CellType::Material(_)))
                    .map(|c| c.cell_rh)
                    .reduce(|a, b| a.min(b))
                    .unwrap_or_default();
                let max_rh = mgr
                    .surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, hamt_core::cell::CellType::Material(_)))
                    .map(|c| c.cell_rh)
                    .reduce(|a, b| a.max(b))
                    .unwrap_or_default();

                let min_t = mgr
                    .surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, hamt_core::cell::CellType::Material(_)))
                    .map(|c| c.cell_temp)
                    .reduce(|a, b| a.min(b))
                    .unwrap_or_default();
                let max_t = mgr
                    .surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, hamt_core::cell::CellType::Material(_)))
                    .map(|c| c.cell_temp)
                    .reduce(|a, b| a.max(b))
                    .unwrap_or_default();
                prev_delta_t = (max_t - min_t).abs();
                prev_delta_rh = (max_rh - min_rh).abs();

                if prev_delta_t.value_in(kelvin) <= allowed_err.value()
                    && prev_delta_rh.value() <= allowed_err.value()
                {
                    constant_time += mgr.settings.timestep;
                } else {
                    constant_time = 0.0 * hours;
                }
            }

            ui.borrow_mut().log(format!(
                "Preheat completed in {:.2} hours",
                adap_state.current_time.value_in(hours)
            ))?;
        }

        ui.borrow_mut().log("Starting simulation...".to_string())?;
        ui.borrow_mut()
            .log(format!("Boundary Condition:\n{:#?}", boundary_condition))?;

        {
            mgr.boundary_condition = boundary_condition;

            let end_time: Time<f64> = *simulation_durations_days.last().unwrap() + 1.0 * hours;

            ui.borrow_mut().log(format!(
                "Simulation duration: {:.2} days",
                (end_time / days).value()
            ))?;

            let mut adap_state = AdaptiveState::default();

            let mut fail_details: Vec<(Time<f64>, u32)> = Vec::new();
            let mut current_time: Time<f64> = 0.0 * hours;
            let orig_timestep = mgr.settings.timestep;

            let speed_tracker = Rc::new(RefCell::new(SpeedTracker::new(10)));

            let make_progress_bar = {
                let ui_clone = ui.clone();
                let start_time_for_ui = Rc::new(start_time);
                let simulation_durations_for_ui = simulation_durations_days.clone();
                let orig_timestep_for_ui = Rc::new(orig_timestep);
                let tracker = speed_tracker.clone();
                move || {
                    let ui_clone = ui_clone.clone();
                    let start_time_for_ui = start_time_for_ui.clone();
                    let simulation_durations_for_ui = simulation_durations_for_ui.clone();
                    let orig_timestep_for_ui = orig_timestep_for_ui.clone();
                    let tracker = tracker.clone();
                    move |state: &AdaptiveState<Vec<(Time<f64>, u32)>>| -> Result<()> {
                        let orig_timestep = (*orig_timestep_for_ui).clone();
                        let current_step =
                            state.proposed_dt.unwrap_or_else(|| orig_timestep.clone());
                        let progress = progress_bar(
                            state.current_time,
                            simulation_durations_for_ui.as_ref(),
                            24,
                        );
                        let sim_hours = state.current_time.value_in(hours);
                        let remaining_hours = (end_time - state.current_time).value_in(hours);
                        tracker.borrow_mut().record(sim_hours);
                        let (speed, eta) = tracker.borrow().speed_and_eta(remaining_hours);
                        let realworld_elapsed = start_time_for_ui.elapsed().as_secs();
                        let dt_ratio = current_step.value_in(hours) / orig_timestep.value_in(hours);
                        ui_clone.borrow_mut().set_status(
                            format!(
                                "{} Time={:.2} hours | Run time: {} seconds | {:.2} hr/s, ETA: {:.1} s | timestep: {:?} (orig: {:?})",
                                progress,
                                sim_hours,
                                realworld_elapsed,
                                speed,
                                eta,
                                current_step,
                                orig_timestep
                            ),
                            StatusStyle::from_dt_ratio(dt_ratio),
                        )
                    }
                }
            };

            while current_time <= end_time {
                let set_progress_bar = make_progress_bar();
                mgr.solve_timestep_adaptive(&mut adap_state, Some(Box::new(set_progress_bar)))?;
                mgr.tick_timestep()?;
                current_time = adap_state.current_time;

                if let Some(sim_day) = simulation_durations_days
                    .iter()
                    .filter(|duration| {
                        (duration.value_in(hours) - &current_time.value_in(hours)).abs()
                            <= mgr.settings.timestep.value_in(hours) * 0.5
                    })
                    .next()
                {
                    let sim_day: i32 = sim_day.value_in(days) as i32;
                    ui.borrow_mut()
                        .log(format!("Checkpoint reached: {} days", sim_day))?;
                    let fail_count =
                        compare_results(&mgr.surface_next, sim_day, &mut *ui.borrow_mut())?;
                    if fail_count > 0 {
                        fail_details.push((current_time, fail_count));

                        // TO DEBUG
                        return Err(anyhow::format_err!(
                            "Diff larger than allowed. {} failures in total.",
                            fail_details.len()
                        ));
                    }
                }
            }

            if !fail_details.is_empty() {
                eprintln!("Test failed in {} benchmarks", fail_details.len());
                for (time, count) in fail_details.iter() {
                    eprintln!(
                        "Failed at time: {} hours, Total failures: {}",
                        time.value_in(hours),
                        count
                    );
                }
                return Err(anyhow::format_err!(
                    "Diff larger than allowed. {} failures in total.",
                    fail_details.len()
                ));
            }
        }

        Ok(())
    })();

    ui.borrow_mut().finish()?;
    result
}

const TABLE_PADDING: usize = 15;

fn log_resistances(surface: &Surface, duration_day: i32) -> Option<String> {
    let mut mats = surface
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)));
    if let Some(c0) = mats.next() {
        Some(format!(
            "Diag day {}: c0 size={:.2e}m temp={:.3}K, rh={:.3e}, w={:.3e}kg/m3, mu={:.3}, dw={:.3e}m2/s, dwdphi={:.3e}, k={:.3}W/mK, density={:.2}kg/m3, spech={:.2}J/kgK\nvp={:.2} Pa, vp_sat={:.2} Pa vtc={:.2e} kg/m2sPa, htc={:.2e} W/m2K",
            duration_day,
            c0.size.x().value_in(meter),
            c0.cell_temp.value_in(kelvin),
            c0.cell_rh.value(),
            c0.volumetric_water.value_in(kg_per_m3),
            c0.mu.value(),
            c0.dw.value_in(m2_per_s),
            c0.dwd_phi.value_in(kg_per_m3),
            c0.w_thermal_c.value_in(w_per_m_k),
            c0.density.value_in(kg_per_m3),
            c0.spech.value_in(j_per_kg_k),
            c0.cell_vp.value_in(pascals),
            c0.vp_sat.value_in(pascals),
            c0.vtc.value_in(kg_per_m2_s_pa),
            c0.htc.value_in(w_per_m2_k)
        ))
    } else {
        None
    }
}

fn compare_results(surface: &Surface, duration_day: i32, ui: &mut Ui) -> Result<u32> {
    let temp_ref = REF_TEMPERATURE
        .iter()
        .filter(|(d, _, _)| *d == duration_day)
        .map(|(_, depth, t)| (depth, t));
    let wc_ref = REF_WATER_CONTENT
        .iter()
        .filter(|(d, _, _)| *d == duration_day)
        .map(|(_, depth, wc)| (depth, wc));

    let mut lines = Vec::new();

    if DIAGNOSTIC_SAMPLE {
        if let Some(line) = log_resistances(surface, duration_day) {
            let mut i = 0;
            let line_size = 100;
            let max_i = line.len() / line_size;
            while i <= max_i {
                lines.push(UiLine::plain(
                    line.chars().skip(i * line_size).take(line_size).collect(),
                ));
                i += 1;
            }
        }
    }
    lines.push(UiLine::plain(format!("Comparing at day {}", duration_day)));
    let temp_header = format_table_row(
        &[
            "Depth (m)".to_string(),
            "Temp (C)".to_string(),
            "Ref Temp (C)".to_string(),
            "Diff (%)".to_string(),
            "Result".to_string(),
            "Cell Index".to_string(),
        ],
        TABLE_PADDING,
    );
    lines.push(UiLine::plain(temp_header.clone()));
    lines.push(UiLine::plain("-".repeat(temp_header.len())));
    let mut fail_count = 0u32;
    let mut t_passed = true;
    for (depth, ref_t) in temp_ref {
        let target = *depth * meter;
        let cell_t = kelvin_to_celsius(surface.interpolate_at_depth(target, |c| c.cell_temp));
        let cell_idx = surface.find_cell_idx_at_depth(target).unwrap_or([0, 0, 0]);
        let diviation = (cell_t - ref_t).abs() / ref_t;

        let turn_passed = diviation <= PERCENTAGE_RANGE_T;
        if !turn_passed {
            fail_count += 1;
        }
        t_passed &= turn_passed;
        let line = format_table_row(
            &[
                format!("{:.2}", depth),
                format!("{:.2}", cell_t),
                format!("{:.2}", ref_t),
                format!("{:.1}%", diviation * 100.0),
                if turn_passed { "OK✔️" } else { "FAIL💥" }.to_string(),
                format!("{:?}", cell_idx),
            ],
            TABLE_PADDING,
        );
        lines.push(UiLine::with_color(
            line,
            if turn_passed {
                Color::DarkGreen
            } else {
                Color::Red
            },
        ));
    }

    lines.push(UiLine::plain(String::new()));
    let wc_header = format_table_row(
        &[
            "Depth (m)".to_string(),
            "Water[kg/m3]".to_string(),
            "Ref Water[kg/m3]".to_string(),
            "Diff (%)".to_string(),
            "Status".to_string(),
            "Cell Index".to_string(),
        ],
        TABLE_PADDING,
    );
    lines.push(UiLine::plain(wc_header.clone()));
    lines.push(UiLine::plain("-".repeat(wc_header.len())));

    let mut rh_passed = true;
    for (depth, ref_wc) in wc_ref {
        let target = *depth * meter;
        let cell_wc = surface
            .interpolate_at_depth(target, |c| c.volumetric_water)
            .value_in(kg_per_m3);
        let cell_idx = surface.find_cell_idx_at_depth(target).unwrap_or([0, 0, 0]);
        let diviation = (cell_wc - ref_wc).abs() / ref_wc;

        let turn_passed = diviation <= PERCENTAGE_RANGE_RH;
        if !turn_passed {
            fail_count += 1;
        }
        rh_passed &= turn_passed;
        let line = format_table_row(
            &[
                format!("{:.2}", depth),
                format!("{:.2}", cell_wc),
                format!("{:.2}", ref_wc),
                format!("{:.1}", diviation * 100.0),
                if turn_passed { "OK✔️" } else { "FAIL💥" }.to_string(),
                format!("{:?}", cell_idx),
            ],
            TABLE_PADDING,
        );
        lines.push(UiLine::with_color(
            line,
            if turn_passed {
                Color::DarkGreen
            } else {
                Color::Red
            },
        ));
    }
    if !rh_passed || !t_passed {
        lines.push(UiLine::plain("-".repeat(TABLE_PADDING * 6)));
        lines.push(UiLine::with_color(
            "Something's wrong!".to_string(),
            Color::Red,
        ));
        lines.push(UiLine::with_color(
            format!("Failed count: {}", fail_count),
            Color::Red,
        ));
    }

    if ui.enabled {
        ui.set_result_lines(lines)?;
    } else {
        for line in lines {
            println!("{}", line.text);
        }
    }

    Ok(fail_count)
}

#[test]
#[ignore = "Prefer Block-Tridiagonal solver."]
fn test_din_en_15026_timestep_2h() -> Result<()> {
    run_benchmark_test(2.0 * hours)
}

#[test]
#[ignore = "Prefer Block-Tridiagonal solver."]
fn test_din_en_15026_timestep_6h() -> Result<()> {
    run_benchmark_test(6.0 * hours)
}

#[test]
#[ignore = "Prefer Block-Tridiagonal solver."]
fn test_din_en_15026_timestep_0_5h() -> Result<()> {
    run_benchmark_test(0.5 * hours)
}

#[test]
#[ignore = "Prefer 2h timestep when dynamic timestep is enabled."]
fn test_din_en_15026_block_tridiag_0_5h() -> Result<()> {
    run_benchmark_test_with_solver(0.5 * hours, SolverMethod::BlockTridiagonal)
}

#[test]
#[ignore = "Prefer 2h timestep when dynamic timestep is enabled."]
// #[ignore = "Please manually test the DIN EN 15026 benchmark"]
fn test_din_en_15026_timestep_0_1h() -> Result<()> {
    run_benchmark_test(0.1 * hours)
}

#[test]
fn test_din_en_15026_block_tridiag_2h() -> Result<()> {
    run_benchmark_test_with_solver(2.0 * hours, SolverMethod::BlockTridiagonal)
}

#[test]
fn test_interpolation() {
    let start = 0.0;
    let end = 1.0;
    let d_point_len = 15;
    let x_values: Vec<_> = interpolate_data_points(start, end, d_point_len, 0.4).collect();

    println!("X: {:?}", x_values);
    assert_eq!(x_values.len(), d_point_len, "Length mismatch");
    assert!(x_values[0] == 0.0, "First value should be 0.0");
    assert!(*x_values.last().unwrap() == 1.0, "Last value should be 1.0");
}

// =====================================================================
// Timestep-invariance test for the total interior moisture flux integral.
//
// Purpose: with no TRNSYS-side air↔wall coupling and a constant interior
// boundary RH held throughout the window, HAMT's per-cell mass-conservative
// implicit Euler must yield the same TOTAL water mass exchanged at the
// interior face for any choice of macro timestep, modulo iteration tolerance.
//
// • Test passes  → HAMT itself is timestep-invariant on this scenario; the
//   sensitivity observed in the coupled TRNSYS run must come from the
//   implicit-Euler-with-piecewise-constant-input air↔wall coupling, not HAMT.
// • Test fails   → HAMT does NOT preserve total mass under timestep changes;
//   a real bug somewhere in the conservation chain (warm-start, substep
//   accumulator, or the cell mass balance).
// =====================================================================

fn run_total_release_with_timestep(timestep: Time<f64>) -> Result<f64> {
    run_total_release_with_timestep_and_solver(timestep, SolverMethod::BlockJacobi)
}

fn run_total_release_with_timestep_and_solver(
    timestep: Time<f64>,
    method: SolverMethod,
) -> Result<f64> {
    init_tracing();

    // Thin single layer — keeps the diffusive timescale comparable to the
    // simulation window so meaningful release happens within minutes/hours
    // rather than days.
    let library = create_library(20.0, 0.5, vec![0.005], vec![Some(20)]);
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = timestep;
    settings.max_iterations = 1500;
    // Use the project default convergence tolerances. Tighter (1e-6) tolerances
    // amplify cell-level oscillation under sharp boundary gradients and force
    // the stall-detection branch to fire spuriously, defeating the substep
    // splitter even with very small sub-dt.
    settings.temp_convergence_tolerance = 1e-3 * kelvin;
    settings.rh_convergence_tolerance = 1e-4 * dimensionless;
    settings.min_div = 15;
    settings.max_div = 30;
    settings.min_relaxation_factor = 0.5;
    settings.solver_method = method;

    // Effectively one-sided: the exterior face is sealed (≈0 transport
    // coefficients) so all interior-face flux corresponds to wall storage
    // change. This makes conservation straightforward to assert.
    let bc_init = BoundaryCondition {
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        heat_transfer_coeff_int: 5.0 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e-8 * kg_per_m2_s_pa,
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
        "TimestepInvarianceTest".to_string(),
        1.0 * square_meter,
    )?;

    // CRITICAL: `create_library` hardcodes `SurfaceVaporCoef { is_internal_const:
    // true, internal_value: 1.0e10, .. }` (line 328-333). `HamtManager::new`
    // imprints that onto `boundary_condition.vapor_transport_coeff_int`,
    // overriding our chosen β with a near-infinite coefficient that makes the
    // surface pathologically stiff (instant VP enforcement → spurious huge
    // fluxes). Same on the exterior, which gets pinned to 0.0. Override both
    // back to realistic values now that the manager has been built. β = 1e-8
    // kg/(m²·s·Pa) is in the same range the deck produces from the Lewis
    // analogy at typical HCONVI ≈ 1.4 W/(m²·K).
    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * kg_per_m2_s_pa;

    // Step change: drop interior boundary RH from 0.5 to 0.4. Held constant
    // for the full window so any cross-timestep total-mass difference must
    // come from HAMT's discretisation, not from input-trajectory differences.
    // A 10-pt RH drop is gentle enough that all timesteps in the test sweep
    // converge without exhausting the substep budget.
    mgr.boundary_condition.rh_inside_air = 0.4 * dimensionless;

    let total_duration: Time<f64> = 6.0 * hours;
    let mut total_release_kg: f64 = 0.0;
    let mut adap_state: AdaptiveState<()> = AdaptiveState::default();
    let mut elapsed: Time<f64> = 0.0 * hours;

    while elapsed < total_duration {
        mgr.solve_timestep_adaptive(&mut adap_state, None)?;
        mgr.tick_timestep()?;
        let area = mgr.surface_curr.area;
        // m_int_dt = ∫ m_int dt over this macro-step  [kg/m²]; the m_int
        // sign convention is flux INTO the wall, so a wall releasing
        // moisture produces a negative integral. Flip the sign so the
        // running total reads as released-mass.
        // strip: extracting raw kg total from a typed integral for sum
        let mass_in_kg = (adap_state.flux_accumulator.m_int_dt * area).value_in(kilograms);
        total_release_kg += -mass_in_kg;
        elapsed += mgr.settings.timestep;
    }

    println!(
        "  Δt = {:>9.4e} h: total released = {:>11.4e} kg",
        timestep.value_in(hours),
        total_release_kg,
    );

    Ok(total_release_kg)
}

#[test]
fn test_total_release_invariance_across_timesteps() -> Result<()> {
    // Span ~30× — finest resolves transient, coarsest forces a near-quasi-
    // -steady jump per step (and may trigger substepping internally).
    let timesteps = [0.03 * hours, 0.25 * hours, 1.0 * hours];

    println!("=== Timestep-invariance of total interior moisture release ===");
    let mut totals: Vec<f64> = Vec::with_capacity(timesteps.len());
    for ts in &timesteps {
        totals.push(run_total_release_with_timestep(*ts)?);
    }

    // Treat the finest Δt as the reference truth.
    let reference = totals[0];
    let allowed_rel_err = 0.05; // 5 %

    println!("\n=== Cross-timestep comparison (reference = finest Δt) ===");
    for (ts, total) in timesteps.iter().zip(totals.iter()) {
        let rel_err = ((total - reference) / reference).abs();
        println!(
            "  Δt = {:>9.4e} h: release = {:>11.4e} kg | rel err vs ref = {:>6.2}%",
            ts.value_in(hours),
            total,
            rel_err * 100.0,
        );
    }

    for (ts, total) in timesteps.iter().zip(totals.iter()) {
        let rel_err = ((total - reference) / reference).abs();
        assert!(
            rel_err < allowed_rel_err,
            "Δt={:.4e} h: total release {:.4e} kg differs from reference (Δt={:.4e} h, {:.4e} kg) by {:.2}% — \
             exceeds {:.1}% tolerance. HAMT does NOT preserve total moisture mass under timestep changes — \
             possible bug in implicit-Euler conservation, warm-start, or the substep flux accumulator.",
            ts.value_in(hours),
            total,
            timesteps[0].value_in(hours),
            reference,
            rel_err * 100.0,
            allowed_rel_err * 100.0,
        );
    }
    Ok(())
}

#[test]
fn test_total_release_invariance_block_tridiag() -> Result<()> {
    // Same timestep sweep as the BlockJacobi variant — span ~30× to verify
    // the BlockTridiagonal solver also preserves total release across
    // a wide range of macro-step choices.
    let timesteps = [0.03 * hours, 0.25 * hours, 1.0 * hours];

    println!("=== Timestep-invariance with BlockTridiagonal solver ===");
    let mut totals: Vec<f64> = Vec::with_capacity(timesteps.len());
    for ts in &timesteps {
        totals.push(run_total_release_with_timestep_and_solver(
            *ts,
            SolverMethod::BlockTridiagonal,
        )?);
    }

    let reference = totals[0];
    let allowed_rel_err = 0.05; // 5 %

    println!("\n=== Cross-timestep comparison (reference = finest Δt) ===");
    for (ts, total) in timesteps.iter().zip(totals.iter()) {
        let rel_err = ((total - reference) / reference).abs();
        println!(
            "  Δt = {:>9.4e} h: release = {:>11.4e} kg | rel err vs ref = {:>6.2}%",
            ts.value_in(hours),
            total,
            rel_err * 100.0,
        );
    }

    for (ts, total) in timesteps.iter().zip(totals.iter()) {
        let rel_err = ((total - reference) / reference).abs();
        assert!(
            rel_err < allowed_rel_err,
            "Δt={:.4e} h: total release {:.4e} kg differs from reference (Δt={:.4e} h, {:.4e} kg) by {:.2}% — \
             exceeds {:.1}% tolerance.",
            ts.value_in(hours),
            total,
            timesteps[0].value_in(hours),
            reference,
            rel_err * 100.0,
            allowed_rel_err * 100.0,
        );
    }
    Ok(())
}

// =====================================================================
// Grid-convergence probe for moisture buffering (CORE SOLVER, no TRNSYS).
//
// Question: does increasing the number of material cells slow / damp the
// moisture-release response (apparent "more inertia"), and does the response
// CONVERGE as N grows? Direction of coarse-grid deviation tells us whether the
// cell-count sensitivity lives in the core solver (a discretisation/lumping
// effect) or must come from the TRNSYS coupling.
//
// Setup: single thin hygroscopic layer (EN 15026 material), interior face with
// a finite β, exterior face effectively sealed. Equilibrate at RH 0.5, then
// STEP the interior boundary RH to 0.3 and watch the wall dry out.
//
// Metrics per cell count N:
//   - peak interior surface moisture flux [kg/m2/s] (Surface::moisture_flux)
//   - total water released over 24 h [kg/m2]  (∫ flux dt at the interior face)
//   - time-to-half-release [h]  (time to reach 50% of the 24 h release)
//   - net wall water-content drop over 24 h [kg/m2] (water_total start-end)
// =====================================================================

struct ProbeResult {
    n_cells_material: usize,
    peak_flux_kg_m2_s: f64,
    total_release_24h_kg_m2: f64,
    time_to_half_h: f64,
    water_drop_kg_m2: f64,
}

fn run_cell_count_probe(num_div: usize) -> Result<ProbeResult> {
    use hamt_core::cell::{BoundaryLocation, CellType};

    init_tracing();

    // Single thin layer so the drying transient lives within the 24 h window.
    // num_divs = Some(N) pins the material-cell count to exactly N regardless of
    // div_length / min_div / max_div (see grid.rs::build_x_column: num_div takes
    // precedence over the div_length-derived guess).
    let library = create_library(20.0, 0.5, vec![0.02], vec![Some(num_div)]);
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = 0.1 * hours; // small macro-step: resolve the transient
    settings.max_iterations = 1500;
    settings.temp_convergence_tolerance = 1e-3 * kelvin;
    settings.rh_convergence_tolerance = 1e-4 * dimensionless;
    // num_div on the material wins; keep min/max wide so they never clamp it.
    settings.min_div = 1;
    settings.max_div = 200;
    settings.min_relaxation_factor = 0.5;
    settings.solver_method = SolverMethod::BlockTridiagonal;

    let bc_init = BoundaryCondition {
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        heat_transfer_coeff_int: 5.0 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e-8 * kg_per_m2_s_pa,
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
        "CellCountProbe".to_string(),
        1.0 * square_meter,
    )?;

    // create_library imprints a near-infinite internal β; reset to realistic.
    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * kg_per_m2_s_pa;

    let n_material = mgr
        .surface_curr
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)))
        .count();

    // --- Equilibrate at RH 0.5 so we start from a clean fixed point. ---
    let mut adap_state: AdaptiveState<()> = AdaptiveState::default();
    let mut eq_time: Time<f64> = 0.0 * hours;
    while eq_time < 48.0 * hours {
        mgr.solve_timestep_adaptive(&mut adap_state, None)?;
        mgr.tick_timestep()?;
        eq_time += mgr.settings.timestep;
    }

    let area = mgr.surface_curr.area;
    let water_start = mgr.surface_curr.report(&mgr.boundary_condition)?.water_total;

    // --- STEP: drop interior boundary RH 0.5 -> 0.3, hold for 24 h. ---
    mgr.boundary_condition.rh_inside_air = 0.3 * dimensionless;

    let total_duration: Time<f64> = 24.0 * hours;
    let mut elapsed: Time<f64> = 0.0 * hours;
    let mut adap_state: AdaptiveState<()> = AdaptiveState::default();

    let mut peak_flux: f64 = 0.0;
    let mut cum_release_kg_m2: f64 = 0.0;
    // (time_h, cumulative_release_kg_m2) samples for time-to-half interpolation.
    let mut samples: Vec<(f64, f64)> = vec![(0.0, 0.0)];

    while elapsed < total_duration {
        mgr.solve_timestep_adaptive(&mut adap_state, None)?;
        mgr.tick_timestep()?;
        elapsed += mgr.settings.timestep;

        // Interior surface moisture flux at end of this macro-step. Flux INTO
        // the wall is positive; drying gives negative, so flip sign to read as
        // released-per-area-per-second.
        // strip: external-boundary scalar for table + comparison
        let m_int = mgr
            .surface_curr
            .moisture_flux(BoundaryLocation::Interior, &mgr.boundary_condition)?
            .value_in(kg_per_m2_s);
        let release_rate = -m_int;
        if release_rate > peak_flux {
            peak_flux = release_rate;
        }

        // Time-weighted cumulative release via the substep accumulator (mass-
        // conservative ∫q dt for this macro-step), independent of substep count.
        // strip: raw kg from typed integral
        let mass_in_kg = (adap_state.flux_accumulator.m_int_dt * area).value_in(kilograms);
        // per unit area, released (positive when drying).
        // strip: scalar area for per-m2 normalisation
        let area_m2 = area.value_in(square_meter);
        cum_release_kg_m2 += -mass_in_kg / area_m2;
        samples.push((elapsed.value_in(hours), cum_release_kg_m2));
    }

    let water_end = mgr.surface_curr.report(&mgr.boundary_condition)?.water_total;
    // strip: scalar masses for table
    let water_drop_kg_m2 =
        (water_start - water_end).value_in(kilograms) / area.value_in(square_meter);

    // time-to-half: first sample crossing 50 % of total cumulative release.
    let half = 0.5 * cum_release_kg_m2;
    let mut time_to_half = f64::NAN;
    for w in samples.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if c0 < half && c1 >= half {
            time_to_half = if (c1 - c0).abs() > 0.0 {
                t0 + (half - c0) / (c1 - c0) * (t1 - t0)
            } else {
                t1
            };
            break;
        }
    }

    println!(
        "  N(mat)={:>3} | peak flux={:>11.4e} kg/m2/s | release 24h={:>11.4e} kg/m2 | t_half={:>6.2} h | water drop={:>11.4e} kg/m2",
        n_material, peak_flux, cum_release_kg_m2, time_to_half, water_drop_kg_m2
    );

    Ok(ProbeResult {
        n_cells_material: n_material,
        peak_flux_kg_m2_s: peak_flux,
        total_release_24h_kg_m2: cum_release_kg_m2,
        time_to_half_h: time_to_half,
        water_drop_kg_m2,
    })
}

#[test]
fn test_zz_cell_count_convergence_probe() -> Result<()> {
    println!("=== Grid-convergence probe: moisture release vs cell count (BlockTridiagonal) ===");
    println!("    Single 0.02 m EN-15026 layer, interior RH step 0.5 -> 0.3, 24 h window");

    let cell_counts = [3usize, 6, 12, 24, 48];
    let mut results: Vec<ProbeResult> = Vec::new();
    for &n in &cell_counts {
        results.push(run_cell_count_probe(n)?);
    }

    // Treat the FINEST grid as the converged reference.
    let reference = results.last().unwrap();
    println!("\n=== Deviation vs finest grid (N={} material cells) ===", reference.n_cells_material);
    println!(
        "{:>5} | {:>14} | {:>14} | {:>10} | {:>14}",
        "N", "peak flux %", "release24h %", "t_half %", "water drop %"
    );
    for r in &results {
        let pf = (r.peak_flux_kg_m2_s - reference.peak_flux_kg_m2_s) / reference.peak_flux_kg_m2_s
            * 100.0;
        let rl = (r.total_release_24h_kg_m2 - reference.total_release_24h_kg_m2)
            / reference.total_release_24h_kg_m2
            * 100.0;
        let th = (r.time_to_half_h - reference.time_to_half_h) / reference.time_to_half_h * 100.0;
        let wd = (r.water_drop_kg_m2 - reference.water_drop_kg_m2) / reference.water_drop_kg_m2
            * 100.0;
        println!(
            "{:>5} | {:>+13.2}% | {:>+13.2}% | {:>+9.2}% | {:>+13.2}%",
            r.n_cells_material, pf, rl, th, wd
        );
    }

    Ok(())
}

pub struct Ui {
    enabled: bool,
    stdout: Stdout,
    title: String,
    log_lines: Vec<String>,
    result_lines: Vec<UiLine>,
    result_history: Vec<UiLine>,
    status_line: String,
    status_style: StatusStyle,
    last_render: Instant,
    min_frame_time: Duration,
    last_render_height: usize,
}

impl Ui {
    fn new(title: &str) -> Result<Self> {
        let mut stdout = io::stdout();
        let enabled = stdout.is_terminal();

        if enabled {
            ctrlc::set_handler(move || {
                let mut stdout = io::stdout();
                let _ = stdout.execute(style::SetAttribute(Attribute::Reset));
                let _ = stdout.execute(cursor::Show);
                let _ = stdout.execute(terminal::LeaveAlternateScreen);
                // 退出进程
                std::process::exit(0);
            })
            .expect("Error setting Ctrl-C handler");

            stdout.execute(terminal::EnterAlternateScreen)?;
            stdout.execute(cursor::Hide)?;
        }

        Ok(Self {
            enabled,
            stdout,
            title: title.to_string(),
            log_lines: Vec::new(),
            result_lines: Vec::new(),
            result_history: Vec::new(),
            status_line: String::new(),
            status_style: StatusStyle::Normal,
            last_render: Instant::now(),
            min_frame_time: Duration::from_millis(1000 / 2),
            last_render_height: 0,
        })
    }

    // ... (log, set_status, set_result_lines 方法保持不变) ...
    // 为了节省篇幅，这里省略，请直接保留你原来的实现

    fn log(&mut self, line: String) -> Result<()> {
        if !self.enabled {
            println!("{}", line);
            return Ok(());
        }
        self.log_lines.push(line);
        self.render(true)
    }

    fn set_status(&mut self, line: String, style: StatusStyle) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.status_line = line;
        self.status_style = style;
        self.render(false)
    }

    fn set_result_lines(&mut self, lines: Vec<UiLine>) -> Result<()> {
        if !self.enabled {
            for line in lines {
                println!("{}", line.text);
            }
            return Ok(());
        }
        if !self.result_history.is_empty() {
            self.result_history.push(UiLine::plain(String::new()));
        }
        self.result_history.extend(lines.iter().cloned());
        self.result_lines = lines;
        self.render(true)
    }

    fn finish(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.enabled = false;

        // 恢复终端状态
        self.stdout.execute(style::SetAttribute(Attribute::Reset))?;
        self.stdout.execute(cursor::Show)?;
        self.stdout.execute(terminal::LeaveAlternateScreen)?;
        let _ = self.stdout.flush();

        // 打印最终结果到标准输出（此时已退出 Alternate Screen）
        println!("{}", self.title);
        println!("{}", "-".repeat(40));
        for line in &self.log_lines {
            println!("{}", line);
        }
        if !self.result_history.is_empty() {
            println!("{}", "-".repeat(40));
            for line in &self.result_history {
                println!("{}", line.text);
            }
        }
        Ok(())
    }

    fn render(&mut self, force: bool) -> Result<()> {
        if !force && self.last_render.elapsed() < self.min_frame_time {
            return Ok(());
        }

        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let cols = cols as usize;
        let rows = rows as usize;

        let mut lines: Vec<UiLine> = Vec::new();
        lines.push(UiLine::plain(self.title.clone()));
        lines.push(UiLine::plain("-".repeat(cols.min(80))));

        for log in &self.log_lines {
            // 假设 textwrap::wrap 返回 Vec<Cow<str>>
            // max_cols 是当前终端宽度
            let wrapped = textwrap::wrap(log, cols);
            for part in wrapped {
                lines.push(UiLine::plain(part.to_string()));
            }
        }

        if !self.result_lines.is_empty() {
            lines.push(UiLine::plain("-".repeat(cols.min(80))));
            lines.extend(self.result_lines.iter().cloned());
        }

        // 截断多余的行，为状态栏留一行
        let max_content_lines = rows.saturating_sub(1);
        if lines.len() > max_content_lines {
            lines = lines.split_off(lines.len() - max_content_lines);
        }

        // [FIX 2] 解决闪烁问题
        // 1. 将光标移回左上角，而不是清空全屏
        self.stdout.queue(cursor::MoveTo(0, 0))?;

        // 2. 逐行覆盖打印，并清除该行右侧残留内容
        for line in &lines {
            let text = truncate_line(&line.text, cols);

            if let Some(color) = line.color {
                self.stdout.queue(style::SetForegroundColor(color))?;
            }

            self.stdout.queue(Print(text))?;

            // 关键：重置颜色并清除当前行直到行尾
            self.stdout.queue(style::SetAttribute(Attribute::Reset))?;
            self.stdout
                .queue(terminal::Clear(ClearType::UntilNewLine))?;
            self.stdout.queue(Print("\n"))?;
        }

        let current_height = lines.len();
        if current_height < self.last_render_height {
            for y in current_height..self.last_render_height.min(max_content_lines) {
                self.stdout.queue(cursor::MoveTo(0, y as u16))?;
                self.stdout.queue(terminal::Clear(ClearType::CurrentLine))?;
            }
        }
        self.last_render_height = current_height;

        let status = truncate_line(&self.status_line, cols);
        self.stdout
            .queue(cursor::MoveTo(0, rows.saturating_sub(1) as u16))?
            .queue(style::SetAttribute(Attribute::Reset))?;

        match self.status_style {
            StatusStyle::Normal => {
                self.stdout.queue(style::SetAttribute(Attribute::Reverse))?;
            }
            StatusStyle::Warning => {
                self.stdout
                    .queue(style::SetBackgroundColor(Color::Yellow))?
                    .queue(style::SetForegroundColor(Color::Black))?;
            }
        }

        self.stdout
            .queue(Print(status))?
            .queue(style::SetAttribute(Attribute::Reset))?
            .queue(terminal::Clear(ClearType::UntilNewLine))?; // 确保状态栏右侧也被清理

        self.stdout.flush()?;
        self.last_render = Instant::now();
        Ok(())
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let _ = self.stdout.execute(style::SetAttribute(Attribute::Reset));
        let _ = self.stdout.execute(cursor::Show);
        let _ = self.stdout.execute(terminal::LeaveAlternateScreen);
    }
}
fn format_table_row(values: &[String], padding: usize) -> String {
    let mut line = String::new();
    for value in values {
        line.push_str(&format!("{:^width$}", value, width = padding));
    }
    line
}

#[derive(Clone)]
struct UiLine {
    text: String,
    color: Option<Color>,
}

impl UiLine {
    fn plain(text: String) -> Self {
        Self { text, color: None }
    }

    fn with_color(text: String, color: Color) -> Self {
        Self {
            text,
            color: Some(color),
        }
    }
}

#[derive(Copy, Clone)]
enum StatusStyle {
    Normal,
    Warning,
}

impl StatusStyle {
    /// Maps a PI-controller dt ratio (current_step / orig_timestep) to a display style.
    /// Ratio < 0.5 means the controller has shortened the step significantly — show Warning.
    fn from_dt_ratio(ratio: f64) -> Self {
        if ratio < 0.5 {
            StatusStyle::Warning
        } else {
            StatusStyle::Normal
        }
    }
}

/// Tracks recent simulation speed using a sliding window over real time.
/// Only considers samples from the last `window` seconds for speed/ETA calculations.
struct SpeedTracker {
    samples: Vec<(Instant, f64)>, // (real_time, sim_hours)
    window: Duration,
}

impl SpeedTracker {
    fn new(window_secs: u64) -> Self {
        Self {
            samples: Vec::new(),
            window: Duration::from_secs(window_secs),
        }
    }

    fn record(&mut self, sim_hours: f64) {
        let now = Instant::now();
        self.samples.push((now, sim_hours));
        // Evict samples older than the window
        let cutoff = now.checked_sub(self.window).unwrap_or(now);
        // Keep at least 1 sample (the earliest in window) for delta computation
        let first_in_window = self.samples.partition_point(|(t, _)| *t < cutoff);
        if first_in_window > 1 {
            self.samples.drain(..first_in_window - 1);
        }
    }

    /// Returns (speed_hr_per_s, eta_seconds). Speed is based on the sliding window.
    fn speed_and_eta(&self, remaining_sim_hours: f64) -> (f64, f64) {
        if self.samples.len() < 2 {
            return (0.0, 0.0);
        }
        let first = &self.samples[0];
        let last = &self.samples[self.samples.len() - 1];
        let dt_real = last.0.duration_since(first.0).as_secs_f64();
        let dt_sim = last.1 - first.1;
        if dt_real < 0.01 {
            return (0.0, 0.0);
        }
        let speed = dt_sim / dt_real;
        let eta = if speed > 0.0 {
            remaining_sim_hours / speed
        } else {
            0.0
        };
        (speed, eta)
    }
}

fn progress_bar(current_time: Time<f64>, checkpoints: &[Time<f64>], width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let current_hours = current_time.value_in(hours);
    let (next_index, next_target) = checkpoints
        .iter()
        .enumerate()
        .find(|(_, target)| target.value_in(hours) >= current_hours)
        .map(|(idx, target)| (idx + 1, target.value_in(hours)))
        .unwrap_or_else(|| {
            (
                checkpoints.len().max(1),
                checkpoints.last().map(|t| t.value_in(hours)).unwrap_or(1.0),
            )
        });

    let ratio = if next_target <= 0.0 {
        0.0
    } else {
        (current_hours / next_target).clamp(0.0, 1.0)
    };
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "[{}{}] {:>3.0}% ({}/{})",
        "#".repeat(filled),
        "-".repeat(empty),
        ratio * 100.0,
        next_index,
        checkpoints.len().max(1)
    )
}

fn truncate_line(line: &str, max_cols: usize) -> String {
    if max_cols == 0 {
        return String::new();
    }
    if line.len() <= max_cols {
        return line.to_string();
    }
    line.chars().take(max_cols).collect()
}

#[test]
fn test_isotherm() {
    let curve = get_sorption_isotherm_curve();
    let target = 92.684517;
    let calculated = curve
        .interpolate(0.8540 * dimensionless)
        .0
        .value_in(kg_per_m3);

    assert!(
        (calculated - target).abs() / target < 0.005,
        "Calculated value: {}",
        calculated
    );
}

#[test]
fn test_porosity() {
    assert_eq!((water_sat() / rho_w()).value(), 0.146);
}
