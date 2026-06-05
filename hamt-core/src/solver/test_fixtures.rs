//! Shared test fixtures for solver unit tests.
//!
//! Gated `#[cfg(test)]` — not compiled into the release binary.

use crate::manager::HamtManager;

/// Build a tiny test surface — single layer, `n_div` cells, fixed
/// material (DIN-EN-15026-style), gentle symmetric BCs.
///
/// Returns the manager so tests can read `surface_curr`, `surface_prev`,
/// `boundary_condition`, `settings`. The fixture is intentionally identical
/// to what `solver::assembly::tests::fixture` used to inline, so all
/// solver-level tests start from the same wall.
pub(crate) fn build_fixture_manager(n_div: usize) -> HamtManager {
    use crate::boundary::{BoundaryCondition, celsius_to_kelvin};
    use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
    use crate::physics::{Curve, LogY};
    use crate::uom::units::{
        dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, meter, square_meter,
        w_per_m_k, w_per_m2_k,
    };
    use crate::uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY,
        DIM_WATER_CONTENT, Quantity,
    };

    // ── Simple material curves ─────────────────────────────────────────
    // Sorption isotherm: w = w_sat * RH^0.5 (simple power-law, monotone).
    let w_sat = 146.0 * kg_per_m3;
    let n_pts = 30usize;
    let si_x: Vec<_> = (0..=n_pts)
        .map(|k| (k as f64 / n_pts as f64) * dimensionless)
        .collect();
    let si_y: Vec<_> = si_x
        .iter()
        .map(|&rh| w_sat * rh.value().max(0.0).sqrt())
        .collect();
    let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
        Curve::new(si_x.clone(), si_y);

    // Vapor diffusion resistance mu (constant = 10).
    let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
    let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
        Curve::new(si_x.clone(), mu_y);

    // Thermal conductivity (constant = 0.5 W/mK vs water content).
    let wc_pts = 20usize;
    let wc_x: Vec<_> = (0..=wc_pts)
        .map(|k| (k as f64 / wc_pts as f64) * w_sat)
        .collect();
    let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
    let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
        Curve::new(wc_x.clone(), k_y);

    // Liquid transport coefficient Dw (constant = 1e-10 m²/s).
    let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
    let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
        Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid fixture Dw curve");

    let mat = Material {
        name: "FixtureMat".to_string(),
        thickness: 0.01 * meter,
        density: 200.0 * kg_per_m3,
        spec_heat: 1000.0 * j_per_kg_k,
        porosity: Quantity(0.146),
        initial_water_content: Quantity(0.5 * 0.146), // 50% of porosity
        sorption_isotherm,
        suction: liquid_tc.clone(),
        redistribution: liquid_tc,
        vapor_diffusion_resistance,
        thermal_conductivity,
        sorption_enthalpy: None,
        initial_temp: celsius_to_kelvin(20.0),
        num_div: Some(n_div),
        div_distribution_factor: 0.5,
    };

    let lib = MaterialLibrary::new();
    lib.add_material(mat).expect("add material");
    // Set surface vapor coefficients to NOT override our BC values.
    let surface_coeff = SurfaceVaporCoef {
        is_external_const: false,
        external_value: 1e-12,
        is_internal_const: false,
        internal_value: 1e-8,
    };
    lib.create_construction(
        "FixtureConstruction".to_string(),
        vec!["FixtureMat".to_string()],
        surface_coeff,
    )
    .expect("create construction");
    let construction = lib.get_construction("FixtureConstruction").unwrap();

    let bc = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_mean_radiant: celsius_to_kelvin(20.0),
        // Symmetric RH so the equilibrium test's fixed-point property holds.
        rh_outside_air: 0.4 * dimensionless,
        rh_inside_air: 0.4 * dimensionless,
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        heat_transfer_coeff_int: 5.0 * w_per_m2_k,
        heat_transfer_coeff_conv_int: 5.0 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * crate::uom::units::kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e-8 * crate::uom::units::kg_per_m2_s_pa,
        ..BoundaryCondition::default()
    };

    let settings = {
        let mut s = crate::config::HamtSettings::default();
        s.min_div = n_div as i32;
        s.max_div = n_div as i32;
        s.coupled_heat_moisture_enabled = true;
        s
    };

    let mut mgr = HamtManager::new(
        Some(settings),
        Some(bc),
        construction,
        "FixtureSurface".to_string(),
        1.0 * square_meter,
    )
    .expect("HamtManager::new");

    // Override any vapor coefficients that HamtManager::new may have
    // imprinted from the SurfaceVaporCoef.
    mgr.boundary_condition.vapor_transport_coeff_int =
        1e-8 * crate::uom::units::kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext =
        1e-12 * crate::uom::units::kg_per_m2_s_pa;

    // Prime: run apply_boundary_coefficients + update_cell_properties.
    // Seed virtual cells to the inside BC so the assembler sees a consistent
    // state from the very first call.
    for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
        surf.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .expect("apply BC");
        let t_seed = mgr.boundary_condition.temp_inside_air_moisture;
        let rh_seed = mgr.boundary_condition.rh_inside_air;
        for cell in surf.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Virtual(_)) {
                cell.cell_temp = t_seed;
                cell.cell_rh = rh_seed;
            }
        }
        surf.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .expect("update props");
    }

    mgr
}

/// Build a tiny 3-D `(n_div_x, ny, nz)` test manager — same single material and
/// curves as [`build_fixture_manager`], constructed via `HamtManager::new_3d`.
///
/// `y_extent` and `z_extent` set the physical size of the grid in the y and z
/// directions respectively. `col_y = y_extent / ny`, `col_z = z_extent / nz`.
/// Use `1.0 * meter` for each to reproduce the previous default. Shrinking
/// these extents makes the off-axis cells smaller relative to the x-cells,
/// increasing the y/z conductance and creating strong 3-D coupling — needed for
/// the z-sweep gate test that proves the sweep is load-bearing.
///
/// `y_faces_active = true` flips the y-min/y-max faces from the default
/// Adiabatic to Active on ALL three surfaces, enabling inter-row (y-axis)
/// conduction. `z_faces_active = true` passes `[FaceBc::Active; 2]` as `z_faces`
/// to `new_3d`, enabling cross-z material coupling.
pub(crate) fn build_fixture_manager_3d(
    n_div_x: usize,
    ny: usize,
    nz: usize,
    y_extent: crate::uom::dimensions::Length<f64>,
    z_extent: crate::uom::dimensions::Length<f64>,
    y_faces_active: bool,
    z_faces_active: bool,
) -> HamtManager {
    use crate::boundary::{BoundaryCondition, celsius_to_kelvin};
    use crate::config::{FACE_Y_MAX, FACE_Y_MIN, FaceBc};
    use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
    use crate::physics::{Curve, LogY};
    use crate::uom::units::{
        dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, meter, square_meter, w_per_m_k, w_per_m2_k,
    };
    use crate::uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
        Quantity,
    };

    // Same curves as build_fixture_manager.
    let w_sat = 146.0 * kg_per_m3;
    let n_pts = 30usize;
    let si_x: Vec<_> = (0..=n_pts)
        .map(|k| (k as f64 / n_pts as f64) * dimensionless)
        .collect();
    let si_y: Vec<_> = si_x
        .iter()
        .map(|&rh| w_sat * rh.value().max(0.0).sqrt())
        .collect();
    let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
        Curve::new(si_x.clone(), si_y);
    let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
    let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
        Curve::new(si_x.clone(), mu_y);
    let wc_pts = 20usize;
    let wc_x: Vec<_> = (0..=wc_pts)
        .map(|k| (k as f64 / wc_pts as f64) * w_sat)
        .collect();
    let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
    let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
        Curve::new(wc_x.clone(), k_y);
    let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
    let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
        Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid fixture Dw curve");

    let mat = Material {
        name: "FixtureMat3D".to_string(),
        thickness: 0.01 * meter,
        density: 200.0 * kg_per_m3,
        spec_heat: 1000.0 * j_per_kg_k,
        porosity: Quantity(0.146),
        initial_water_content: Quantity(0.5 * 0.146),
        sorption_isotherm,
        suction: liquid_tc.clone(),
        redistribution: liquid_tc,
        vapor_diffusion_resistance,
        thermal_conductivity,
        sorption_enthalpy: None,
        initial_temp: celsius_to_kelvin(20.0),
        num_div: Some(n_div_x),
        div_distribution_factor: 0.5,
    };

    let lib = MaterialLibrary::new();
    lib.add_material(mat).expect("add material");
    let surface_coeff = SurfaceVaporCoef {
        is_external_const: false,
        external_value: 1e-12,
        is_internal_const: false,
        internal_value: 1e-8,
    };
    lib.create_construction(
        "FixtureConstruction3D".to_string(),
        vec!["FixtureMat3D".to_string()],
        surface_coeff,
    )
    .expect("create construction");
    let construction = lib.get_construction("FixtureConstruction3D").unwrap();

    let bc = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_mean_radiant: celsius_to_kelvin(20.0),
        rh_outside_air: 0.4 * dimensionless,
        rh_inside_air: 0.4 * dimensionless,
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        heat_transfer_coeff_int: 5.0 * w_per_m2_k,
        heat_transfer_coeff_conv_int: 5.0 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * crate::uom::units::kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e-8 * crate::uom::units::kg_per_m2_s_pa,
        ..BoundaryCondition::default()
    };

    let settings = crate::config::HamtSettings {
        min_div: n_div_x as i32,
        max_div: n_div_x as i32,
        coupled_heat_moisture_enabled: true,
        ..crate::config::HamtSettings::default()
    };

    let z_faces = if z_faces_active {
        [FaceBc::Active; 2]
    } else {
        [FaceBc::Adiabatic; 2]
    };

    let mut mgr = HamtManager::new_3d(
        Some(settings),
        Some(bc),
        construction,
        "FixtureSurface3D".to_string(),
        1.0 * square_meter,
        ny,
        y_extent,
        nz,
        z_extent,
        z_faces,
    )
    .expect("HamtManager::new_3d");

    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * crate::uom::units::kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * crate::uom::units::kg_per_m2_s_pa;

    if y_faces_active {
        for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev, &mut mgr.surface_next] {
            surf.face_bcs[FACE_Y_MIN] = FaceBc::Active;
            surf.face_bcs[FACE_Y_MAX] = FaceBc::Active;
        }
    }

    for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
        surf.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .expect("apply BC");
        let t_seed = mgr.boundary_condition.temp_inside_air_moisture;
        let rh_seed = mgr.boundary_condition.rh_inside_air;
        for cell in surf.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Virtual(_)) {
                cell.cell_temp = t_seed;
                cell.cell_rh = rh_seed;
            }
        }
        surf.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .expect("update props");
    }

    mgr
}

/// Build a tiny 2-D `(n_div_x, ny, 1)` test manager — same single material and
/// curves as [`build_fixture_manager`], constructed via `HamtManager::new_2d`.
/// Used by the ADI-vs-direct strong-coupling gate (Task 10). KEEP `n_div_x` and
/// `ny` small: the direct oracle solves a dense `2*Nmat` system.
///
/// `y_faces_active = true` flips the y-min/y-max faces from the default
/// Adiabatic to Active on BOTH surfaces, enabling inter-row (cross-axis)
/// conduction — the off-axis coupling a thermal bridge exercises.
pub(crate) fn build_fixture_manager_2d(
    n_div_x: usize,
    ny: usize,
    y_faces_active: bool,
) -> HamtManager {
    use crate::boundary::{BoundaryCondition, celsius_to_kelvin};
    use crate::config::{FACE_Y_MAX, FACE_Y_MIN, FaceBc};
    use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
    use crate::physics::{Curve, LogY};
    use crate::uom::units::{
        dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, meter, square_meter, w_per_m_k, w_per_m2_k,
    };
    use crate::uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
        Quantity,
    };

    // Same curves as build_fixture_manager.
    let w_sat = 146.0 * kg_per_m3;
    let n_pts = 30usize;
    let si_x: Vec<_> = (0..=n_pts)
        .map(|k| (k as f64 / n_pts as f64) * dimensionless)
        .collect();
    let si_y: Vec<_> = si_x
        .iter()
        .map(|&rh| w_sat * rh.value().max(0.0).sqrt())
        .collect();
    let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
        Curve::new(si_x.clone(), si_y);
    let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
    let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
        Curve::new(si_x.clone(), mu_y);
    let wc_pts = 20usize;
    let wc_x: Vec<_> = (0..=wc_pts)
        .map(|k| (k as f64 / wc_pts as f64) * w_sat)
        .collect();
    let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
    let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
        Curve::new(wc_x.clone(), k_y);
    let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
    let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
        Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid fixture Dw curve");

    let mat = Material {
        name: "FixtureMat2D".to_string(),
        thickness: 0.01 * meter,
        density: 200.0 * kg_per_m3,
        spec_heat: 1000.0 * j_per_kg_k,
        porosity: Quantity(0.146),
        initial_water_content: Quantity(0.5 * 0.146),
        sorption_isotherm,
        suction: liquid_tc.clone(),
        redistribution: liquid_tc,
        vapor_diffusion_resistance,
        thermal_conductivity,
        sorption_enthalpy: None,
        initial_temp: celsius_to_kelvin(20.0),
        num_div: Some(n_div_x),
        div_distribution_factor: 0.5,
    };

    let lib = MaterialLibrary::new();
    lib.add_material(mat).expect("add material");
    let surface_coeff = SurfaceVaporCoef {
        is_external_const: false,
        external_value: 1e-12,
        is_internal_const: false,
        internal_value: 1e-8,
    };
    lib.create_construction(
        "FixtureConstruction2D".to_string(),
        vec!["FixtureMat2D".to_string()],
        surface_coeff,
    )
    .expect("create construction");
    let construction = lib.get_construction("FixtureConstruction2D").unwrap();

    let bc = BoundaryCondition {
        temp_outside_air: celsius_to_kelvin(20.0),
        temp_inside_air: celsius_to_kelvin(20.0),
        temp_inside_air_moisture: celsius_to_kelvin(20.0),
        temp_outside_air_moisture: celsius_to_kelvin(20.0),
        temp_inside_mean_radiant: celsius_to_kelvin(20.0),
        rh_outside_air: 0.4 * dimensionless,
        rh_inside_air: 0.4 * dimensionless,
        heat_transfer_coeff_ext: 1e-9 * w_per_m2_k,
        heat_transfer_coeff_int: 5.0 * w_per_m2_k,
        heat_transfer_coeff_conv_int: 5.0 * w_per_m2_k,
        vapor_transport_coeff_ext: 1e-12 * crate::uom::units::kg_per_m2_s_pa,
        vapor_transport_coeff_int: 1e-8 * crate::uom::units::kg_per_m2_s_pa,
        ..BoundaryCondition::default()
    };

    let settings = crate::config::HamtSettings {
        min_div: n_div_x as i32,
        max_div: n_div_x as i32,
        coupled_heat_moisture_enabled: true,
        ..crate::config::HamtSettings::default()
    };

    let mut mgr = HamtManager::new_2d(
        Some(settings),
        Some(bc),
        construction,
        "FixtureSurface2D".to_string(),
        1.0 * square_meter,
        ny,
        1.0 * meter,
    )
    .expect("HamtManager::new_2d");

    mgr.boundary_condition.vapor_transport_coeff_int = 1e-8 * crate::uom::units::kg_per_m2_s_pa;
    mgr.boundary_condition.vapor_transport_coeff_ext = 1e-12 * crate::uom::units::kg_per_m2_s_pa;

    if y_faces_active {
        for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev, &mut mgr.surface_next] {
            surf.face_bcs[FACE_Y_MIN] = FaceBc::Active;
            surf.face_bcs[FACE_Y_MAX] = FaceBc::Active;
        }
    }

    for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
        surf.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .expect("apply BC");
        let t_seed = mgr.boundary_condition.temp_inside_air_moisture;
        let rh_seed = mgr.boundary_condition.rh_inside_air;
        for cell in surf.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Virtual(_)) {
                cell.cell_temp = t_seed;
                cell.cell_rh = rh_seed;
            }
        }
        surf.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .expect("update props");
    }

    mgr
}
