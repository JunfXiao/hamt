#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Test suite for the DIN EN 15026 analytical benchmark (Annex B).
use anyhow::{Ok, Result};
use crossterm::{
    ExecutableCommand, QueueableCommand, cursor,
    style::{self, Attribute, Color, Print},
    terminal::{self, ClearType},
};
use hamt_core::surface::Surface;
use hamt_core::uom::dimensions::{
    Density, Dimensionless, HeatCapacity, Length, SpecificHeatCapacity, Temperature, Time,
    VolumetricHeatCapacity, WaterContent,
};
use hamt_core::uom::units::{
    self, days, dimensionless, hours, j_per_kg_k, j_per_m3_k, kelvin, kg_per_m2_s, kg_per_m2_s_pa,
    kg_per_m3, m_per_s, m2_per_s, meter, square_meter, w_per_m_k, w_per_m2_k,
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
    cell::{BoundaryLocation, CellType},
    hamt_manager::{AdaptiveState, HamtManager, HamtSettings},
};
use hamt_core::{
    material::{Curve, Material, MaterialLibrary, SurfaceVaporCoef},
    uom::dimensions::Mass,
};
use hamt_core::{physics::ECLIPSE, uom::units::kilograms};
use std::sync::Once;
use std::time::{Duration, Instant};
use std::{
    borrow::Cow,
    io::{self, IsTerminal, Stdout, Write},
    ops::Deref,
};
use std::{cell::RefCell, ops::Div, rc::Rc};
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use tracing_test::traced_test;

// Reference results from Table B.1 and B.2
const REF_WATER_CONTENT: &[(i32, f64, f64)] = &[
    // (days, depth_m, wc_kg_m3)
    // Source: Tabelle B.1 (Page 37)

    // --- 7 Days ---
    (7, 0.01, 52.4),
    (7, 0.02, 43.4),
    (7, 0.03, 42.9),
    (7, 0.04, 42.9),
    (7, 0.05, 42.9),
    (7, 0.06, 42.9),
    (7, 0.08, 42.9),
    (7, 0.10, 42.9),
    // --- 30 Days ---
    (30, 0.01, 83.1),
    (30, 0.02, 53.3),
    (30, 0.03, 45.7),
    (30, 0.04, 43.6),
    (30, 0.05, 43.0),
    (30, 0.06, 42.9),
    (30, 0.08, 42.9),
    (30, 0.10, 42.9),
    // --- 365 Days ---
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
    // Source: Tabelle B.2 (Page 38)

    // --- 7 Days ---
    (7, 0.5, 26.64),
    (7, 1.0, 23.84),
    (7, 1.5, 21.91),
    (7, 2.0, 20.82),
    (7, 2.5, 20.30),
    (7, 3.0, 20.09),
    (7, 4.0, 20.01),
    (7, 5.0, 20.00),
    // --- 30 Days ---
    (30, 0.5, 28.35),
    (30, 1.0, 26.75),
    (30, 1.5, 25.29),
    (30, 2.0, 24.01),
    (30, 2.5, 22.93),
    (30, 3.0, 22.07),
    (30, 4.0, 20.92),
    (30, 5.0, 20.35),
    // --- 365 Days ---
    (365, 0.5, 29.54),
    (365, 1.0, 29.06),
    (365, 1.5, 28.58),
    (365, 2.0, 28.11),
    (365, 2.5, 27.64),
    (365, 3.0, 27.19),
    (365, 4.0, 26.30),
    (365, 5.0, 25.47),
];

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

fn curve_by_func<const DX: Dimension, const DY: Dimension>(
    func: fn(Quantity<f64, DX>) -> Quantity<f64, DY>,
    x_range: (Quantity<f64, DX>, Quantity<f64, DX>),
    delta: Quantity<f64, DX>,
) -> Curve<DX, DY>
where
    Quantity<f64, DX>: Div<Quantity<f64, DX>, Output = Dimensionless<f64>>,
{
    let data_len = ((x_range.1 - x_range.0) / delta) + 1.0f64;
    let x_series: Vec<Quantity<f64, DX>> = (0..data_len.value() as usize)
        .map(|i| x_range.0 + i as f64 * delta)
        .collect();
    curve_by_func_and_x_values(func, x_series)
}

fn curve_by_func_and_x_values<const DX: Dimension, const DY: Dimension>(
    func: fn(Quantity<f64, DX>) -> Quantity<f64, DY>,
    x_values: impl IntoIterator<Item = Quantity<f64, DX>>,
) -> Curve<DX, DY>
where
    Quantity<f64, DX>: Div<Quantity<f64, DX>, Output = Dimensionless<f64>>,
{
    let x_values: Vec<_> = x_values.into_iter().collect();
    let y_series: Vec<Quantity<f64, DY>> = x_values.iter().map(|x| func(*x)).collect();
    Curve::new(x_values.into_iter().collect(), y_series)
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

    // Geometric Progression
    let q = 1.0 - proportion;
    let n = d_point_len as i32 - 1; // No 0.0
    let a_1 = (1.0 - q) / (1.0 - q.powi(n));

    let s_q = a_1 * (1.0 - q.powi(n)) / (1.0 - q);

    debug_assert_eq!(s_q, 1.0);

    let mut x_values: Vec<f64> = Vec::with_capacity(d_point_len);

    for i in 0..n + 1 {
        x_values.push(a_1 * (1.0 - q.powi(i as i32)) / (1.0 - q));
    }

    x_values.into_iter().map(move |v| start + v * (end - start))
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

// The RH - $mu$ curve
fn get_vapor_diffusion_curve() -> Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> {
    let x_values: Vec<_> =
        interpolate_data_points(0.0, 1.0, INTERPOLATION_POINTS, INTERPOLATION_DELTA)
            .map(|v| v * dimensionless)
            .collect();

    curve_by_func_and_x_values(
        |x: Dimensionless<f64>| {
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

fn get_liquid_transport_coeff_curve() -> Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF> {
    let x_values: Vec<WaterContent<f64>> = interpolate_data_points(
        0.0,
        water_sat().value_in(kg_per_m3),
        INTERPOLATION_POINTS,
        INTERPOLATION_DELTA,
    )
    .map(|x| x * kg_per_m3)
    .collect();

    curve_by_func_and_x_values(
        |x: WaterContent<f64>| {
            let w = x.value_in(kg_per_m3);
            let w_sat = water_sat().value_in(kg_per_m3);
            let k_w = (-39.2619 + 0.0704 * (w - 73.0)
                - 1.7420e-4 * (w - 73.0).powi(2)
                - 2.7953e-6 * (w - 73.0).powi(3)
                - 1.1566e-7 * (w - 73.0).powi(4)
                + 2.5969e-9 * (w - 73.0).powi(5))
            .exp();
            let dpc_dw = 0.125e8 * 0.375 * w_sat.powf(0.375) / w.powf(1.375);
            return k_w * dpc_dw * m2_per_s;
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

// TODO: Vapor Diffusion

const MAT_NAME: &str = "DIN EN 15026 Mat";
const CONSTRUCTION_NAME: &str = "DIN EN 15026 Construction";

fn create_library(init_temp_celsius: f64, init_rh: f64, thicknesses: Vec<f64>) -> MaterialLibrary {
    let thermal_conductivity = get_thermal_conductivity_curve();
    let sorption_isotherm = get_sorption_isotherm_curve();
    let base_material = Material {
        name: MAT_NAME.to_string(),
        thickness: 0.0 * meter, // Placeholder, will be overridden
        density: rho_mat(),
        thermal_conductivity: thermal_conductivity,
        spec_heat: spec_heat_capacity(),
        porosity: water_sat() / rho_w(),
        initial_water_content: sorption_isotherm.interpolate(init_rh * dimensionless).0 / rho_mat(),
        suction: get_liquid_transport_coeff_curve(),
        redistribution: get_liquid_transport_coeff_curve(),
        sorption_isotherm: sorption_isotherm,
        vapor_diffusion_resistance: get_vapor_diffusion_curve(),
        initial_temp: celsius_to_kelvin(init_temp_celsius),
    };

    let lib = MaterialLibrary::new();
    let mut layer_names = Vec::new();

    for (i, d) in thicknesses.iter().enumerate() {
        let mut mat = base_material.clone();
        mat.name = format!("{}_{}", MAT_NAME, i);
        mat.thickness = *d * meter;
        layer_names.push(mat.name.clone());
        lib.add_material(mat).expect("Cannot add material.");
    }

    let surface_coeff = SurfaceVaporCoef {
        is_external_const: false,
        external_value: 0.0,
        internal_value: 0.0,
        is_internal_const: false,
    };
    lib.create_construction(CONSTRUCTION_NAME.to_string(), layer_names, surface_coeff)
        .expect("Cannot create construction");

    lib
}

const INTERPOLATION_POINTS: usize = 120;
const INTERPOLATION_DELTA: f64 = 0.6;
const DIAGNOSTIC_SAMPLE: bool = true;

fn run_benchmark_test(time_step: Time<f64>) -> Result<()> {
    let start_time = Instant::now();

    init_tracing();

    let mut ui = Rc::new(RefCell::new(Ui::new("DIN EN 15026 Benchmark")?));

    let result = (|| -> Result<()> {
        // Use 3 layers to optimize grid:
        // 1. Surface (0.1m): Fine grid for moisture profile
        // 2. Transition (1.0m): Intermediate grid
        // 3. Bulk (48.9m): Coarse grid for semi-infinite thermal mass
        let library = create_library(20.0, 0.5, vec![15.0]);
        let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

        let mut settings = HamtSettings::default();
        settings.timestep = time_step;
        settings.max_iterations = 500;
        settings.temp_convergence_tolerance = 1e-6 * kelvin;
        settings.rh_convergence_tolerance = 1e-5;
        settings.max_div = 200;
        settings.min_div = 20;
        settings.div_length = 1e-5 * meter;
        settings.adj_length = 1e-4 * meter;

        let boundary_coeff = BoundaryCondition {
            heat_transfer_coeff_ext: 1e100 * w_per_m2_k,
            heat_transfer_coeff_int: 1e100 * w_per_m2_k,
            mass_convection_coeff_ext: 1e100 * m_per_s,
            mass_convection_coeff_int: 1e100 * m_per_s,
            vapor_transport_coeff_ext: 1e100 * kg_per_m2_s_pa,
            vapor_transport_coeff_int: 1e100 * kg_per_m2_s_pa,
            ..Default::default()
        };

        let boundary_condition_init = BoundaryCondition {
            temp_outside_air: celsius_to_kelvin(20.0),
            temp_inside_air: celsius_to_kelvin(20.0),
            rh_outside_air: 0.5 * dimensionless,
            rh_inside_air: 0.5 * dimensionless,
            ..boundary_coeff
        };

        let boundary_condition = BoundaryCondition {
            temp_outside_air: celsius_to_kelvin(30.0),
            temp_inside_air: celsius_to_kelvin(30.0),
            rh_outside_air: 0.95 * dimensionless,
            rh_inside_air: 0.95 * dimensionless,
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

        let simulation_durations_days: Cow<[Time<f64>; 3]> =
            Cow::Owned([7.0 * days, 30.0 * days, 365.0 * days]);

        {
            ui.borrow_mut()
                .log(format!("Running benchmark with time step: {:?}", time_step))?;
            ui.borrow_mut()
                .log("Preheating to reach stable...".to_string())?;

            let mut prev_t: Temperature<f64> = 0.0 * kelvin;
            let mut prev_water: Mass<f64> = 0.0 * kilograms;
            let mut constant_time: Time<f64> = 0.0 * seconds;
            let mut allowed_err: Dimensionless<f64> = 0.001 * dimensionless;
            let mut adap_state = AdaptiveState::default();

            let make_progress_bar_preheat = {
                let ui_clone = ui.clone();
                let start_time_for_ui = Rc::new(start_time);
                let t_avg = prev_t.clone();
                let water_total = prev_water.clone();
                let orig_timestep_for_ui = Rc::new(mgr.settings.timestep.clone());
                move || {
                    let ui_clone = ui_clone.clone();
                    let start_time_for_ui = start_time_for_ui.clone();
                    let orig_timestep_for_ui = orig_timestep_for_ui.clone();
                    move |state: &AdaptiveState| -> Result<()> {
                        let orig_timestep = (*orig_timestep_for_ui).clone();
                        let current_step = if state.split_factor > 0 {
                            orig_timestep / 2.0_f64.powi(state.split_factor - 1)
                        } else {
                            orig_timestep.clone()
                        };
                        ui_clone.borrow_mut().set_status(
                            format!(
                                "Preheat T_avg={:.2} deg, Water={:.2} kg | Time={:.2} hours | Run time: {} seconds | timestep: {:?} (orig: {:?})",
                                kelvin_to_celsius(t_avg),
                                water_total.value_in(kilograms),
                                (state.current_time / hours).value(),
                                start_time_for_ui.elapsed().as_secs(),
                                current_step,
                                orig_timestep
                            ),
                            StatusStyle::from_split_factor(state.split_factor),
                        )
                    }
                }
            };
            while constant_time < 240.0 * hours {
                // if !mgr.solve_timestep()? {
                //     return Err(hamt_core::error::Error::Other(
                //         "Preheating failed: timestep did not converge".to_string(),
                //     )
                //     .into());
                // }

                let set_progress_bar = make_progress_bar_preheat();
                mgr.solve_timestep_adaptive(&mut adap_state, Some(Box::new(set_progress_bar)))?;
                // adaptive_state.current_time += mgr.settings.timestep;

                let curr_t = mgr
                    .surface_prev
                    .avg_virtual_cell(BoundaryLocation::Exterior, |c| c.cell_temp)?;
                let curr_water = mgr
                    .surface_prev
                    .cells
                    .iter()
                    .filter(|c| matches!(c.cell_type, hamt_core::cell::CellType::Material(_)))
                    .map(|c| c.volumetric_water * c.volume)
                    .sum::<Mass<f64>>();

                if ((curr_water - prev_water) / prev_water).value() < allowed_err
                    && ((curr_t - prev_t) / prev_t).value() < allowed_err
                {
                    constant_time += mgr.settings.timestep;
                } else {
                    constant_time = 0.0 * hours;
                }

                prev_t = curr_t;
                prev_water = curr_water;

                if adap_state.current_time > 24.0 * 30.0 * hours {
                    return Err(hamt_core::error::Error::Other(format!(
                        "Preheat failed: Exceeded maximum preheat time {:?}",
                        adap_state.current_time
                    ))
                    .into());
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

            let make_progress_bar = {
                let ui_clone = ui.clone();
                let start_time_for_ui = Rc::new(start_time);
                let simulation_durations_for_ui = simulation_durations_days.clone();
                let orig_timestep_for_ui = Rc::new(orig_timestep);
                move || {
                    let ui_clone = ui_clone.clone();
                    let start_time_for_ui = start_time_for_ui.clone();
                    let simulation_durations_for_ui = simulation_durations_for_ui.clone();
                    let orig_timestep_for_ui = orig_timestep_for_ui.clone();
                    move |state: &AdaptiveState| -> Result<()> {
                        let orig_timestep = (*orig_timestep_for_ui).clone();
                        let current_step = if state.split_factor > 0 {
                            orig_timestep / 2.0_f64.powi(state.split_factor - 1)
                        } else {
                            orig_timestep.clone()
                        };
                        let progress = progress_bar(
                            state.current_time,
                            simulation_durations_for_ui.as_ref(),
                            24,
                        );
                        ui_clone.borrow_mut().set_status(
                            format!(
                                "{} Time={:.2} hours | Run time: {} seconds | timestep: {:?} (orig: {:?})",
                                progress,
                                (state.current_time / hours).value(),
                                start_time_for_ui.elapsed().as_secs(),
                                current_step,
                                orig_timestep
                            ),
                            StatusStyle::from_split_factor(state.split_factor),
                        )
                    }
                }
            };

            while current_time <= end_time {
                let set_progress_bar = make_progress_bar();
                mgr.solve_timestep_adaptive(&mut adap_state, Some(Box::new(set_progress_bar)))?;

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
            "Diag day {}: c0 temp={:.2}K, rh={:.3}, w={:.3e}kg/m3, mu={:.3}, dw={:.3e}m2/s, dwdphi={:.3e}, k={:.3}W/mK, density={:.3}kg/m3, spech={:.3}J/kgK",
            duration_day,
            c0.cell_temp.value_in(kelvin),
            c0.cell_rh.value(),
            c0.volumetric_water.value_in(kg_per_m3),
            c0.mu.value(),
            c0.dw.value_in(m2_per_s),
            c0.dwd_phi.value_in(kg_per_m3),
            c0.w_thermal_c.value_in(w_per_m_k),
            c0.density.value_in(kg_per_m3),
            c0.spech.value_in(j_per_kg_k),
        ))
    } else {
        None
    }
}

fn compare_results(surface: &Surface, duration_day: i32, ui: &mut Ui) -> Result<u32> {
    const PERCENTAGE_RANGE_RH: f64 = 0.025;
    const PERCENTAGE_RANGE_T: f64 = 0.01;
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
            lines.push(UiLine::plain(line));
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
        let cell_idx = surface
            .find_cell_idx_at_depth(*depth * meter)
            .expect(&format!("Searching cell at depth {}", *depth));
        let cell = surface.cells.get(cell_idx).expect("Cannot find the cell");

        // println!("Cell at {:.2} m: {}", depth, cell.cell_type);

        let cell_t = kelvin_to_celsius(cell.cell_temp);
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
        let cell_idx = surface
            .find_cell_idx_at_depth(*depth * meter)
            .expect(&format!("Searching cell at depth {}", *depth));
        let cell = surface.cells.get(cell_idx).expect("Cannot find the cell");

        let cell_wc = cell.volumetric_water.value_in(kg_per_m3);
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
fn test_din_en_15026_timestep_1h() -> Result<()> {
    run_benchmark_test(1.0 * hours)
}

#[test]
#[ignore = "Prefer 1h timestep when dynamic timestep is enabled."]
fn test_din_en_15026_timestep_0_5h() -> Result<()> {
    run_benchmark_test(0.5 * hours)
}

#[test]
#[ignore = "Prefer 1h timestep when dynamic timestep is enabled."]
// #[ignore = "Please manually test the DIN EN 15026 benchmark"]
fn test_din_en_15026_timestep_0_1h() -> Result<()> {
    run_benchmark_test(0.1 * hours)
}

#[test]
fn test_interpolation() {
    let start = 0.0;
    let end = 1.0;
    let d_point_len = 15;
    let mut x_values: Vec<_> = interpolate_data_points(start, end, d_point_len, 0.4).collect();

    println!("X: {:?}", x_values);
    assert_eq!(x_values.len(), d_point_len, "Length mismatch");
    assert!(x_values[0] == 0.0, "First value should be 0.0");
    assert!(*x_values.last().unwrap() == 1.0, "Last value should be 1.0");
}

static INIT: Once = Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    // 如果没有环境变量，使用这个默认值：
                    // "info" -> 显示所有 info 级别
                    // "my_cool_app=debug" -> 特别将你的包开启到 debug 级别
                    .unwrap_or_else(|_| EnvFilter::new("info,hamt_core=info")),
            )
            .with_test_writer() // 关键：使用 TestWriter 以便 cargo test 正确捕获输出
            .init();
    });
}

struct Ui {
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
}

impl Ui {
    fn new(title: &str) -> Result<Self> {
        let mut stdout = io::stdout();
        let enabled = stdout.is_terminal();
        if enabled {
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
            min_frame_time: Duration::from_millis(1000 / 10),
        })
    }

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
        self.stdout.execute(style::SetAttribute(Attribute::Reset))?;
        self.stdout.execute(cursor::Show)?;
        self.stdout.execute(terminal::LeaveAlternateScreen)?;
        let _ = self.stdout.flush();

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
        let mut lines: Vec<UiLine> = Vec::new();
        lines.push(UiLine::plain(self.title.clone()));
        lines.push(UiLine::plain("-".repeat(cols.min(80))));
        lines.extend(self.log_lines.iter().cloned().map(UiLine::plain));
        if !self.result_lines.is_empty() {
            lines.push(UiLine::plain("-".repeat(cols.min(80))));
            lines.extend(self.result_lines.iter().cloned());
        }

        let max_lines = rows.saturating_sub(1) as usize;
        if lines.len() > max_lines {
            lines = lines.split_off(lines.len() - max_lines);
        }

        self.stdout
            .queue(cursor::MoveTo(0, 0))?
            .queue(terminal::Clear(ClearType::All))?;

        for line in lines {
            let text = truncate_line(&line.text, cols);
            if let Some(color) = line.color {
                self.stdout.queue(style::SetForegroundColor(color))?;
            }
            self.stdout.queue(Print(text))?;
            if line.color.is_some() {
                self.stdout.queue(style::SetAttribute(Attribute::Reset))?;
            }
            self.stdout.queue(Print("\n"))?;
        }

        let status = truncate_line(&self.status_line, cols);
        self.stdout
            .queue(cursor::MoveTo(0, rows.saturating_sub(1)))?
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
            .queue(style::SetAttribute(Attribute::Reset))?;

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
    fn from_split_factor(split_factor: i32) -> Self {
        if split_factor > 1 {
            StatusStyle::Warning
        } else {
            StatusStyle::Normal
        }
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
fn test_isotherm(){
    let curve = get_sorption_isotherm_curve();


}