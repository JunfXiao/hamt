//! Shared physics harness and EN 15026 reference tables, used by both
//! `din_en_15026_benchmark` and `adi_2d`.
#![allow(dead_code)]

use hamt_core::material::{Material, MaterialLibrary, SurfaceVaporCoef};
use hamt_core::physics::{Curve, CurveScale, LogY};
use hamt_core::uom::dimensions::{
    Density, Dimensionless, SpecificHeatCapacity, Temperature, VolumetricHeatCapacity, WaterContent,
};
use hamt_core::uom::units::{
    dimensionless, j_per_kg_k, j_per_m3_k, kelvin, kg_per_m3, m2_per_s, meter, w_per_m_k,
};
use hamt_core::uom::{
    DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
    Dimension, Quantity,
};
use hamt_core::boundary::celsius_to_kelvin;
use std::ops::Div;
use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt};

// ---------------------------------------------------------------------------
// EN 15026 reference tables
// Source: DIN EN 15026 Annex B (normative)
// ---------------------------------------------------------------------------

/// Reference water-content profiles (days, depth_m, wc_kg_m3).
// Source: Tabelle B.1 (Page 37)
pub const REF_WATER_CONTENT: &[(i32, f64, f64)] = &[
    // (days, depth_m, wc_kg_m3)
    // Source: Tabelle B.1 (Page 37)
    // (1, 0.01, 26.64),
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

/// Reference temperature profiles (days, depth_m, temp_c).
// Source: Tabelle B.2 (Page 38)
pub const REF_TEMPERATURE: &[(i32, f64, f64)] = &[
    // (days, depth_m, temp_c)
    // Source: Tabelle B.2 (Page 38)
    // (1, 0.5, 26.64),
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

// ---------------------------------------------------------------------------
// Comparison tolerances (same values used by compare_results and compare_2d).
// ---------------------------------------------------------------------------

/// Allowed relative deviation for temperature comparisons (1 %).
pub const PERCENTAGE_RANGE_T: f64 = 0.01;
/// Allowed relative deviation for water-content comparisons (2.5 %).
pub const PERCENTAGE_RANGE_RH: f64 = 0.025;

// ---------------------------------------------------------------------------
// Material constants
// ---------------------------------------------------------------------------

pub const MAT_NAME: &str = "DIN EN 15026 Mat";
pub const CONSTRUCTION_NAME: &str = "DIN EN 15026 Construction";
pub const INTERPOLATION_POINTS: usize = 200;
pub const INTERPOLATION_DELTA: f64 = 0.6;

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------

pub fn t_ref() -> Temperature<f64> {
    293.15 * kelvin
}

pub fn rho_w() -> Density<f64> {
    1000.0 * kg_per_m3
}

pub fn r_h2o() -> SpecificHeatCapacity<f64> {
    462.0 * j_per_kg_k
}

pub fn water_sat() -> Quantity<f64, DIM_WATER_CONTENT> {
    146.0 * kg_per_m3
}

pub fn volumetric_heat_capacity() -> VolumetricHeatCapacity<f64> {
    1.824e6 * j_per_m3_k
}

pub fn rho_mat() -> Density<f64> {
    200.0 * kg_per_m3
}

pub fn spec_heat_capacity() -> SpecificHeatCapacity<f64> {
    volumetric_heat_capacity() / rho_mat()
}

// ---------------------------------------------------------------------------
// Curve builders
// ---------------------------------------------------------------------------

pub fn curve_by_func<const DX: Dimension, const DY: Dimension, S: CurveScale>(
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

pub fn curve_by_func_and_x_values<const DX: Dimension, const DY: Dimension, S: CurveScale>(
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

pub fn sorption_isotherm(rh: Dimensionless<f64>) -> Quantity<f64, DIM_WATER_CONTENT> {
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

pub fn interpolate_data_points(
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

pub fn get_sorption_isotherm_curve() -> Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> {
    let mut x_values: Vec<_> =
        interpolate_data_points(0.0, 1.0, INTERPOLATION_POINTS, INTERPOLATION_DELTA)
            .map(|v| v * dimensionless)
            .collect();

    if let Some(last) = x_values.last_mut() {
        *last = 1.0 * dimensionless;
    }

    curve_by_func_and_x_values(|x: Dimensionless<f64>| sorption_isotherm(x), x_values)
}

// The RH - μ curve
pub fn get_vapor_diffusion_curve() -> Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> {
    // RH
    let x_values: Vec<_> =
        interpolate_data_points(0.0, 1.0, INTERPOLATION_POINTS, INTERPOLATION_DELTA)
            .map(|v| v * dimensionless)
            .collect();

    curve_by_func_and_x_values(
        |x: Dimensionless<f64>| {
            let x = x.value().clamp(0.0, 1.0) * dimensionless;
            let water_content_ratio = sorption_isotherm(x) / water_sat();
            let rest_water_ratio = 1.0 - water_content_ratio.value();
            // μ = μ_dry / f(w), where f(w) = (1-w/w_f) / (0.503*(1-w/w_f)² + 0.497)
            // This is the direct form that avoids any δ₀ temperature conversion.
            // It is consistent with vapor_diffusivity_model = Linear { coeff: 26.1e-6 },
            // which uses δ_a = 26.1e-6 / (VAPOR_GAS_CONSTANT × T) — the temperature factors cancel in μ.
            200.0 * (0.503 * rest_water_ratio.powf(2.0) + 0.497) / rest_water_ratio * dimensionless
        },
        x_values,
    )
}

pub fn get_thermal_conductivity_curve() -> Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> {
    let x_range = (0.0 * water_sat(), 1.0 * water_sat());
    let delta = 0.01 * water_sat();
    curve_by_func(
        |x: Density<f64>| (1.5 + 15.8e-3 * x.value_in(kg_per_m3)) * w_per_m_k,
        x_range,
        delta,
    )
}

pub fn get_liquid_transport_coeff_curve(
) -> Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> {
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
            // let dpc_dw = 0.125e8 * 0.375 * w_sat.powf(0.375) / w.powf(1.375);
            let dpc_dw = p_suc * 0.625f64 / (1.0f64 - w_ratio.powf(1.0 / 0.375)) / 0.375 / w;
            k_w * dpc_dw * m2_per_s
        },
        x_values,
    )
}

// ---------------------------------------------------------------------------
// Library factory
// ---------------------------------------------------------------------------

pub fn create_library(
    init_temp_celsius: f64,
    init_rh: f64,
    thicknesses: Vec<f64>,
    num_divs: Vec<Option<usize>>,
) -> MaterialLibrary {
    let thermal_conductivity = get_thermal_conductivity_curve();
    let sorption_isotherm = get_sorption_isotherm_curve();
    let base_material = Material {
        name: MAT_NAME.to_string(),
        thickness: 0.0 * meter, // Placeholder, will be overridden
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

// ---------------------------------------------------------------------------
// Tracing initialiser (shared Once guard)
// ---------------------------------------------------------------------------

static INIT: Once = Once::new();

pub fn init_tracing() {
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
// Face energy/mass balance helpers (shared by adi_2d and adi_3d balance tests)
// ---------------------------------------------------------------------------

/// Compute the total stored water mass [kg] over all material cells in the surface.
/// Works for any grid shape (1-D, 2-D, 3-D).
pub fn total_stored_water(
    surface: &hamt_core::surface::Surface,
) -> hamt_core::uom::dimensions::Mass<f64> {
    use hamt_core::cell::CellType;
    use hamt_core::uom::units::kilograms;
    surface
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)))
        .map(|c| c.volumetric_water * c.volume)
        .fold(0.0 * kilograms, |acc, x| acc + x)
}

/// Compute a heat-content proxy E = Σ (ρ·c_dry + w·c_w) · V · T  [J]
/// over all material cells. Used only for the ENERGY balance where a few-
/// percent tolerance is applied. Works for any grid shape.
pub fn total_stored_energy(
    surface: &hamt_core::surface::Surface,
) -> hamt_core::uom::dimensions::Energy<f64> {
    use hamt_core::cell::CellType;
    use hamt_core::uom::units::{j_per_kg_k, joules, kg_per_m3};
    // WATER_SPECIFIC_HEAT = 4180 J/(kg·K): use the raw constant value directly.
    const C_WATER: f64 = 4180.0; // J/(kg·K)

    surface
        .cells
        .iter()
        .filter(|c| matches!(c.cell_type, CellType::Material(_)))
        .map(|c| {
            // strip: Diman cannot canonicalise (kg/m³·J/kg·K)·m³·K in one step;
            // we do the scalar multiplication and tag back as Energy = joules.
            let rho_c_dry = c.density.value_in(kg_per_m3) * c.spech.value_in(j_per_kg_k);
            let rho_c_water = c.volumetric_water.value_in(kg_per_m3) * C_WATER;
            let rho_c_eff = rho_c_dry + rho_c_water; // J/(m³·K)
            let e_scalar =
                rho_c_eff * c.volume.value_unchecked() * c.cell_temp.value_unchecked(); // J
            e_scalar * joules
        })
        .fold(0.0 * joules, |acc, x| acc + x)
}

/// Build a minimal single-layer construction for balance tests.
///
/// Uses the DIN EN 15026 material curves but with non-const surface VTCs so
/// that the BC values are not overridden. `init_rh` sets the initial
/// equilibrium moisture content.
pub fn build_balance_construction(
    init_rh: f64,
) -> std::sync::Arc<hamt_core::material::Construction> {
    use hamt_core::material::{MaterialLibrary, SurfaceVaporCoef};
    const BALANCE_MAT_NAME: &str = "BalanceMat";
    const BALANCE_CONST_NAME: &str = "BalanceConst";

    // Build a base library to access the EN 15026 material curves, then
    // re-wrap them in a construction with NON-CONST surface VTCs so the BC
    // vapor_transport_coeff values are not overridden at manager construction.
    let base_lib = create_library(20.0, init_rh, vec![0.1], vec![Some(6)]);
    let base_const = base_lib.get_construction(CONSTRUCTION_NAME).unwrap();

    let mat = base_const.layers[0].as_ref().clone();
    let mut new_mat = mat.clone();
    new_mat.name = BALANCE_MAT_NAME.to_string();

    let lib = MaterialLibrary::new();
    lib.add_material(new_mat).expect("add material");
    // NON-CONST surface VTCs: BC values flow through unchanged.
    let sv = SurfaceVaporCoef {
        is_external_const: false,
        external_value: 0.0,
        is_internal_const: false,
        internal_value: 0.0,
    };
    lib.create_construction(
        BALANCE_CONST_NAME.to_string(),
        vec![BALANCE_MAT_NAME.to_string()],
        sv,
    )
    .expect("create balance construction");
    lib.get_construction(BALANCE_CONST_NAME).unwrap()
}
