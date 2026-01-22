#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Test suite for the DIN EN 15026 analytical benchmark (Annex B).

use anyhow::{Ok, Result};
use hamt_core::boundary::{BoundaryCondition, celsius_to_kelvin, kelvin_to_celsius};
use hamt_core::cell::{BoundaryLocation, Cell, CellType};
use hamt_core::hamt_manager::{HamtManager, HamtSettings};
use hamt_core::io::load_hamt_by_path;
use hamt_core::material::{Curve, Material, MaterialLibrary, SurfaceVaporCoef};
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
use std::fs::File;
use std::ops::Div;
use std::path::PathBuf;
use tempfile;

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
    let mut x_values: Vec<_> = interpolate_data_points(0.0, 1.0, 15, 0.4)
        .map(|v| v * dimensionless)
        .collect();

    if let Some(last) = x_values.last_mut() {
        *last = 1.0 * dimensionless;
    }

    curve_by_func_and_x_values(|x: Dimensionless<f64>| sorption_isotherm(x), x_values)
}

// The RH - $mu$ curve
fn get_vapor_diffusion_curve() -> Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> {
    let x_values: Vec<_> = interpolate_data_points(0.0, 1.0, 15, 0.4)
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
    let delta = 0.05 * water_sat();
    curve_by_func(
        |x: Density<f64>| (1.5 + 15.8e-3 * x.value_in(kg_per_m3)) * w_per_m_k,
        x_range,
        delta,
    )
}

fn get_liquid_transport_coeff_curve() -> Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF> {
    let x_values: Vec<WaterContent<f64>> =
        interpolate_data_points(0.0, water_sat().value_in(kg_per_m3), 15, 0.3)
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
            let dpc_dw = -0.125e8 / 0.375 * w_sat.powf(1.0 / 0.375) / w.powf(1.0 / 0.375 + 1.0);
            return k_w * dpc_dw.abs() * m2_per_s;
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

fn create_library() -> MaterialLibrary {
    let thermal_conductivity = get_thermal_conductivity_curve();
    let sorption_isotherm = get_sorption_isotherm_curve();
    let material = Material {
        name: MAT_NAME.to_string(),
        thickness: 10.0 * meter,
        density: rho_mat(),
        thermal_conductivity: thermal_conductivity,
        spec_heat: spec_heat_capacity(),
        porosity: water_sat() / rho_w(),
        initial_water_content: sorption_isotherm.interpolate(0.5 * dimensionless).0 / rho_mat(), // Initial RH: 50%
        suction: get_liquid_transport_coeff_curve(),
        redistribution: get_liquid_transport_coeff_curve(),
        sorption_isotherm: sorption_isotherm,
        vapor_diffusion_resistance: get_vapor_diffusion_curve(),
        initial_temp: celsius_to_kelvin(20.0), // Initial temperature: 20°C
    };
    let lib = MaterialLibrary::new();
    let mat = lib.add_material(material).expect("Cannot add material.");
    let surface_coeff = SurfaceVaporCoef {
        is_external_const: true,
        external_value: 1e10,
        internal_value: 1e10,
        is_internal_const: true,
    };
    lib.create_construction(
        CONSTRUCTION_NAME.to_string(),
        vec![MAT_NAME.to_string()],
        surface_coeff,
    )
    .expect("Cannot create construction");

    lib
}

fn run_benchmark_test(time_step: Time<f64>) -> Result<()> {
    let library = create_library();
    let construction = library.get_construction(CONSTRUCTION_NAME).unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = time_step;
    settings.max_iterations = 50;
    settings.temp_convergence_tolerance = 1e-6 * kelvin;
    settings.rh_convergence_tolerance = 1e-7;
    settings.max_div = 80;
    settings.min_div = 50;
    settings.div_length = 1e-6 * meter;

    let boundary_coeff = BoundaryCondition {
        heat_transfer_coeff_ext: 1e10 * w_per_m2_k,
        heat_transfer_coeff_int: 1e10 * w_per_m2_k,
        mass_convection_coeff_ext: 1e10 * m_per_s,
        mass_convection_coeff_int: 1e10 * m_per_s,
        vapor_transport_coeff_ext: 1e10 * kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e10 * kg_per_m2_s_pa,
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

    println!("Cell Count: {}", mgr.surface_curr.cells.len());

    let initial_temp = celsius_to_kelvin(20.0);
    let initial_rh = 0.50 * dimensionless;
    for cell in mgr.surface_curr.cells.iter_mut() {
        cell.cell_temp = initial_temp;
        cell.cell_rh = initial_rh;
    }
    for cell in mgr.surface_prev.cells.iter_mut() {
        cell.cell_temp = initial_temp;
        cell.cell_rh = initial_rh;
    }

    let simulation_durations_days: [Time<f64>; 3] = [7.0 * days, 30.0 * days, 365.0 * days];

    {
        println!("\nRunning benchmark with time step: {:?}", time_step);
        println!("=================================================");
        println!("Preheating to reach stable...");

        let mut preheat_time: Time<f64> = 0.0 * hours;
        while !mgr.solve_timestep()? || preheat_time < 10.0 * hours {
            preheat_time += time_step;
            if preheat_time > 24.0 * 30.0 * hours {
                return Err(hamt_core::error::Error::Other(
                    "Preheating failed: Exceeded maximum preheat time".to_string(),
                )
                .into());
            }
            preheat_time += settings.timestep;
        }

        println!(
            "\n\nPreheating completed in {:.2} hours",
            preheat_time.value_in(hours)
        );
    }

    println!("\n=================================================");

    println!("\nStarting simulation...");

    println!("\n Boundary Condition:\n {:#?}", boundary_condition);

    {
        mgr.boundary_condition = boundary_condition;

        let end_time: Time<f64> = *simulation_durations_days.last().unwrap() + 1.0 * hours;

        println!("Simulation duration: {:.2} days", (end_time / days).value());

        let mut write_time = true;
        let mut current_time: Time<f64> = 0.0 * hours;
        while current_time <= end_time {
            if write_time {
                print!("\nTime={:.2} hours", (current_time / hours).value());
                write_time = false;
            }
            print!("\rTime={:.2} hours", (current_time / hours).value());
            let mut solve_count = 0;
            // Solving the timestep
            while !mgr.solve_timestep()? {
                solve_count += 1;

                print!(
                    "\rTime={:.2} hours, Iteration {}",
                    (current_time / hours).value(),
                    solve_count * mgr.settings.max_iterations
                );

                if solve_count > 30 {
                    return Err(hamt_core::error::Error::Other(format!(
                        "Convergence failed after {} iterations at {:.2} hours",
                        solve_count,
                        (current_time / hours).value()
                    ))
                    .into());
                }
            }

            if let Some(sim_day) = simulation_durations_days
                .iter()
                .filter(|duration| {
                    (duration.value_in(hours) - &current_time.value_in(hours)).abs()
                        <= settings.timestep.value_in(hours) * 0.5
                })
                .next()
            {
                let sim_day: i32 = sim_day.value_in(days) as i32;
                println!("\nCheckpoint reached: {} days\n", sim_day);
                compare_results(&mgr.surface_curr, sim_day);
                write_time = true;
            }

            current_time += settings.timestep;

            mgr.tick_timestep()?;
        }
    }

    Ok(())
}

fn print_table_row(values: impl IntoIterator<Item = String>, padding: usize) {
    for value in values {
        print!("{:^width$}", value, width = padding);
    }
    println!("");
}

const TABLE_PADDING: usize = 15;

fn compare_results(surface: &Surface, duration_day: i32) {
    let sep = "=".repeat(30);
    println!("{}", sep);
    println!("Comparing at day {}", duration_day);
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

    println!("{}", sep);

    print_table_row(
        vec![
            "Depth (m)",
            "Temp (C)",
            "Ref Temp (C)",
            "Diff (%)",
            "Result",
            "Cell Index",
        ]
        .iter()
        .map(|s| s.to_string()),
        TABLE_PADDING,
    );
    println!("------------------");

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
        t_passed &= turn_passed;
        print_table_row(
            vec![
                format!("{:.2}", depth),
                format!("{:.2}", cell_t),
                format!("{:.2}", ref_t),
                format!("{:.1}%", diviation * 100.0),
                format!("{}", if turn_passed { "🆗" } else { "💥" }),
                format!("{:?}", cell_idx),
            ],
            TABLE_PADDING,
        );
    }

    println!("{}", sep);
    print_table_row(
        vec![
            "Depth (m)",
            "Water[kg/m3]",
            "Ref Water[kg/m3]",
            "Diff (%)",
            "Status",
            "Result",
            "Cell Index",
        ]
        .iter()
        .map(|s| s.to_string()),
        TABLE_PADDING,
    );
    println!("------------------");

    let mut rh_passed = true;
    for (depth, ref_wc) in wc_ref {
        let cell_idx = surface
            .find_cell_idx_at_depth(*depth * meter)
            .expect(&format!("Searching cell at depth {}", *depth));
        let cell = surface.cells.get(cell_idx).expect("Cannot find the cell");

        let cell_wc = cell.volumetric_water.value_in(kg_per_m3);
        let diviation = (cell_wc - ref_wc).abs() / ref_wc;

        let turn_passed = diviation <= PERCENTAGE_RANGE_RH;
        rh_passed &= turn_passed;
        print_table_row(
            vec![
                format!("{:.2}", depth),
                format!("{:.2}", cell_wc),
                format!("{:.2}", ref_wc),
                format!("{:.1}", diviation * 100.0),
                format!("{}", if turn_passed { "🆗" } else { "💥" }),
                format!("{:?}", cell_idx),
            ],
            TABLE_PADDING,
        );
    }
    if !rh_passed || !t_passed {
        println!("------------------");
        println!("Something's wrong!");
        panic!("Test failed.")
    }
}

#[test]
fn test_din_en_15026_timestep_1h() -> Result<()> {
    run_benchmark_test(1.0 * hours)
}

#[test]

fn test_din_en_15026_timestep_0_5h() -> Result<()> {
    run_benchmark_test(0.5 * hours)
}

#[test]
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
