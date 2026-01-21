//! Test suite for the DIN EN 15026 benchmark.
//
// This test suite implements the analytical benchmark case from DIN EN 15026, Annex B.
// The benchmark case is a semi-infinite slab of a specified material, initially in
// equilibrium with a certain climate, and then exposed to a different climate.
//
// The test runs the simulation for different time steps and compares the results
// with the analytical solution.

use hamt_core::boundary::BoundaryCondition;
use hamt_core::hamt_manager::{HamtManager, HamtSettings};
use hamt_core::io::load_hamt_data;
use hamt_core::uom::dimensions::{Length, Temperature};
use hamt_core::uom::units::{self, celsius, dimensionless, hours, kg_per_m3, meter, square_meter};
use hamt_core::uom::{Quantity, Time};
use std::sync::Arc;

// TODO: Replace with actual benchmark data from DIN EN 15026.
const EXPECTED_RESULTS: &str = r#"
    // Placeholder for expected results.
    // The format should be something like:
    // time_days, depth_m, temperature_c, water_content_kg_m3
    7, 0.01, 25.0, 50.0
    30, 0.01, 28.0, 60.0
    365, 0.01, 29.0, 65.0
"#;

/// Runs the DIN EN 15026 benchmark test for a given time step.
///
/// # Arguments
///
/// * `time_step` - The time step for the simulation, in hours.
/// * `idf_path` - The path to the IDF file with the material properties.
fn run_benchmark_test(time_step_h: f64, idf_path: &str) {
    // 1. Set up initial and boundary conditions from the standard.
    // Initial equilibrium state: 20°C, 50% RH
    // Step change to boundary conditions: 30°C, 95% RH
    let initial_temp = 20.0 * celsius;
    let initial_rh = 0.50 * dimensionless;
    let boundary_temp = 30.0 * celsius;
    let boundary_rh = 0.95 * dimensionless;

    // 2. Load the material properties from the IDF file.
    let library = load_hamt_data(idf_path).unwrap();
    let construction = library.get_construction("20241205_LP-LB").unwrap();

    // 3. Set up simulation settings.
    let mut settings = HamtSettings::default();
    settings.timestep = time_step_h * hours;
    // For a semi-infinite slab, we need a thick-enough construction.
    // The benchmark should define this. For now, we use the thickness
    // from the IDF file.
    let total_thickness = construction
        .layers
        .iter()
        .fold(Length::zero(), |acc, m| acc + m.thickness);

    // 4. Set up the boundary conditions.
    // The benchmark assumes a step change, so the outside and inside conditions are the same.
    let boundary_condition = BoundaryCondition {
        temp_outside_air: boundary_temp,
        temp_inside_air: boundary_temp,
        rh_outside_air: boundary_rh,
        rh_inside_air: boundary_rh,
        ..Default::default()
    };

    // 5. Create the HamtManager.
    // The initial state of the cells will be based on the material properties,
    // which should correspond to the initial equilibrium state.
    // We need to verify how the initial state is set in HamtManager::new.
    // For now, we assume it's initialized to a default state and we might need to
    // adjust it.
    let mut hamt_manager = HamtManager::new(
        Some(settings),
        Some(boundary_condition),
        construction,
        "TestSurface".to_string(),
        1.0 * square_meter,
    )
    .unwrap();

    // 6. Define simulation durations.
    let simulation_durations_days = [7, 30, 365];
    let mut current_time = Time::from_seconds(0.0);
    let mut duration_idx = 0;

    println!("\nRunning benchmark with time step: {} hours", time_step_h);
    println!("-------------------------------------------------");
    println!("Time (days)\tDepth (m)\tTemperature (C)\tWater Content (kg/m^3)");

    // 7. Run the simulation loop.
    while duration_idx < simulation_durations_days.len() {
        let end_time = Time::from_days(simulation_durations_days[duration_idx] as f64);

        while current_time < end_time {
            hamt_manager.solve_timestep().unwrap();
            hamt_manager.tick_timestep().unwrap();
            current_time += hamt_manager.settings.timestep;
        }

        // 8. Output results at each specified duration.
        println!(
            "Results after {} days:",
            simulation_durations_days[duration_idx]
        );
        let mut depth = Length::zero();
        for cell in hamt_manager.surface_curr.cells.iter() {
            let temp_c = cell.cell_temp.value_in(celsius);
            let water_content = cell.volumetric_water.value_in(kg_per_m3);
            depth += cell.size.x() / 2.0;
            println!(
                "{}\t\t{:.4}\t\t{:.2}\t\t{:.2}",
                simulation_durations_days[duration_idx],
                depth.value_in(meter),
                temp_c,
                water_content
            );
            depth += cell.size.x() / 2.0;
        }

        // 9. Compare with expected results (placeholder).
        // let expected = parse_expected_results(EXPECTED_RESULTS, simulation_durations_days[duration_idx]);
        // compare_results(&hamt_manager.surface_curr, &expected);

        duration_idx += 1;
    }
}

#[test]
#[ignore] // Ignoring until we have the correct material properties and expected results.
fn test_din_en_15026_timestep_1h() {
    run_benchmark_test(1.0, "fixtures/idf_example.idf");
}

#[test]
#[ignore] // Ignoring until we have the correct material properties and expected results.
fn test_din_en_15026_timestep_0_5h() {
    run_benchmark_test(0.5, "fixtures/idf_example.idf");
}

#[test]
#[ignore] // Ignoring until we have the correct material properties and expected results.
fn test_din_en_15026_timestep_0_1h() {
    run_benchmark_test(0.1, "fixtures/idf_example.idf");
}

// Helper functions for parsing expected results and comparing them.
// These are placeholders and need to be implemented.
/*
fn parse_expected_results(data: &str, duration_days: i32) -> Vec<(f64, f64, f64)> {
    // Parses the expected results string for a given duration.
    // ...
    vec![]
}

fn compare_results(surface: &Surface, expected: &Vec<(f64, f64, f64)>) {
    // Compares the simulation results with the expected values, with a 2.5% tolerance.
    // ...
    const TOLERANCE: f64 = 0.025;
    // assert!(...);
}
*/
