#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Test suite for the DIN EN 15026 analytical benchmark (Annex B).

use std::ops::Div;

use hamt_core::boundary::{BoundaryCondition, celsius_to_kelvin, kelvin_to_celsius};
use hamt_core::cell::{Cell, CellType};
use hamt_core::hamt_manager::{HamtManager, HamtSettings};
use hamt_core::io::load_hamt_data;
use hamt_core::material::Curve;
use hamt_core::surface::Surface;
use hamt_core::uom::dimensions::{
    Density, Dimensionless, Length, SpecificHeatCapacity, Temperature,
};
use hamt_core::uom::units::{
    self, dimensionless, hours, j_per_kg_k, kelvin, kg_per_m3, meter, square_meter,
};
use hamt_core::uom::{DIM_DIMENSIONLESS, DIM_WATER_CONTENT, Dimension, Quantity};

// Reference results from Table B.1 and B.2
const REF_WATER_CONTENT: &[(i32, f64, f64)] = &[
    // (days, depth_m, wc_kg_m3)
    (7, 0.01, 52.4),
    (7, 0.02, 43.4),
    (7, 0.03, 42.9),
    (30, 0.01, 83.1),
    (30, 0.02, 53.3),
    (30, 0.03, 45.7),
    (365, 0.01, 119.7),
    (365, 0.02, 106.6),
    (365, 0.03, 90.8),
];

const REF_TEMPERATURE: &[(i32, f64, f64)] = &[
    // (days, depth_m, temp_c)
    (7, 0.01, 26.64),
    (7, 0.02, 23.84),
    (7, 0.03, 21.91),
    (30, 0.01, 28.35),
    (30, 0.02, 26.75),
    (30, 0.03, 25.29),
    (365, 0.01, 29.54),
    (365, 0.02, 29.06),
    (365, 0.03, 28.58),
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
    let y_series: Vec<Quantity<f64, DY>> = x_series.iter().map(|x| func(*x)).collect();
    Curve::new(x_series, y_series)
}

fn get_sorption_isotherm_curve() -> Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> {
    let x_range = (0.0 * dimensionless, 1.0 * dimensionless);
    let delta = 0.05 * dimensionless;

    curve_by_func(
        |x: Dimensionless<f64>| {
            let w_sat = 146.0 * kg_per_m3;
            let r_h2o = r_h2o().value_in(j_per_kg_k);
            let t_ref = t_ref().value_in(kelvin);
            let rho_w = rho_w().value_in(kg_per_m3);
            w_sat
                * (1.0f64 + (8e-8 * r_h2o * t_ref * rho_w * x.value().ln()).powf(1.6)).powf(-0.375)
        },
        x_range,
        delta,
    )
}

// TODO: Vapor Diffusion

fn run_benchmark_test(time_step_h: f64) {
    let idf_path = "fixtures/din_en_15026_benchmark.idf";
    let library = load_hamt_data(idf_path).unwrap();
    let construction = library.get_construction("EN15026_Construction").unwrap();

    let mut settings = HamtSettings::default();
    settings.timestep = time_step_h * hours;
    settings.max_iterations = 100;
    settings.temp_convergence_tolerance = 1e-4 * kelvin;
    settings.rh_convergence_tolerance = 1e-5;

    let boundary_condition = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(30.0),
        temp_inside_air: celsius_to_kelvin(30.0),
        rh_outside_air: 0.95 * dimensionless,
        rh_inside_air: 0.95 * dimensionless,
        ..Default::default()
    };

    let mut hamt_manager = HamtManager::new(
        Some(settings.clone()),
        Some(boundary_condition),
        construction,
        "BenchmarkSurface".to_string(),
        1.0 * square_meter,
    )
    .unwrap();

    let initial_temp = celsius_to_kelvin(20.0);
    let initial_rh = 0.50 * dimensionless;
    for cell in hamt_manager.surface_curr.cells.iter_mut() {
        cell.cell_temp = initial_temp;
        cell.cell_rh = initial_rh;
    }
    for cell in hamt_manager.surface_prev.cells.iter_mut() {
        cell.cell_temp = initial_temp;
        cell.cell_rh = initial_rh;
    }

    let simulation_durations_days = [7, 30, 365];
    let mut current_time = 0.0 * hours;

    println!("\nRunning benchmark with time step: {} hours", time_step_h);
    println!("=================================================");

    for &duration_days in &simulation_durations_days {
        let end_time = hours * 24.0 * duration_days as f64;
        while current_time < end_time {
            if !hamt_manager.solve_timestep().unwrap() {
                panic!("Convergence failed!");
            }
            hamt_manager.tick_timestep().unwrap();
            current_time += settings.timestep;
        }

        println!("\nResults after {} days:", duration_days);
        compare_results(&hamt_manager.surface_curr, duration_days);
    }
}

fn compare_results(surface: &Surface, duration_days: i32) {
    println!("Depth (m)\tTemp (C)\tRef Temp (C)\tDiff (%)\tWC (kg/m³)\tRef WC (kg/m³)\tDiff (%)");
    println!(
        "----------------------------------------------------------------------------------------------------"
    );

    let mut cumulative_depth: Length<f64> = 0.0 * meter;
    for cell in surface.cells.iter() {
        if let CellType::Material(_) = cell.cell_type {
            cumulative_depth += cell.size.x() / 2.0;
            let depth_m = cumulative_depth.value_in(meter);

            let temp_c = kelvin_to_celsius(cell.cell_temp);
            let wc = cell.volumetric_water.value_in(kg_per_m3);

            if let Some((ref_wc, wc_tol_ok)) =
                find_and_check_ref(REF_WATER_CONTENT, duration_days, depth_m, wc, 0.025)
            {
                if let Some((ref_temp, temp_tol_ok)) =
                    find_and_check_ref(REF_TEMPERATURE, duration_days, depth_m, temp_c, 0.01)
                {
                    println!(
                        "{:.4}\t\t{:.2}\t\t{:.2}\t\t{:.2} {}\t{:.2}\t\t{:.2}\t\t{:.2} {}",
                        depth_m,
                        temp_c,
                        ref_temp,
                        (temp_c - ref_temp) / ref_temp * 100.0,
                        if temp_tol_ok { "✅" } else { "❌" },
                        wc,
                        ref_wc,
                        (wc - ref_wc) / ref_wc * 100.0,
                        if wc_tol_ok { "✅" } else { "❌" },
                    );
                }
            }
            cumulative_depth += cell.size.x() / 2.0;
        }
    }
}

fn find_and_check_ref(
    ref_data: &[(i32, f64, f64)],
    days: i32,
    depth: f64,
    value: f64,
    tolerance: f64,
) -> Option<(f64, bool)> {
    ref_data
        .iter()
        .find(|&&(d, dep, _)| d == days && (dep - depth).abs() < 1e-3)
        .map(|&(_, _, ref_val)| {
            let diff = (value - ref_val).abs() / ref_val;
            (ref_val, diff <= tolerance)
        })
}

#[test]
#[ignore]
fn test_din_en_15026_timestep_1h() {
    run_benchmark_test(1.0);
}

#[test]
#[ignore]
fn test_din_en_15026_timestep_0_5h() {
    run_benchmark_test(0.5);
}

#[test]
#[ignore]
fn test_din_en_15026_timestep_0_1h() {
    run_benchmark_test(0.1);
}
