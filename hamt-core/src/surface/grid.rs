
use super::*;
use crate::uom::dimensions::{Area, Length};
use crate::uom::units::meter;
use crate::{
    cell::Boundary,
    material::Construction,
    physics::geometry::{Pt3D, Size3D},
};

/// Build ONE x-column of cells (exterior boundary stack -> material layers ->
/// interior boundary stack), advancing `base_pt.x` as it goes. `col_y`/`col_z`
/// are the cross-section extents (y,z) used for EVERY cell in this column
/// (boundary and material). Returns the column cells in x-order.
pub(super) fn build_x_column(
    settings: &HamtSettings,
    construction: &Construction,
    base_pt: &mut Pt3D,
    col_y: Length<f64>,
    col_z: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<Vec<Cell>> {
    let boundary_cell_thickness = settings.base_cell_size.x();
    let boundary_cell_size = Size3D::new(boundary_cell_thickness, col_y, col_z);

    let get_size = |thickness: Length<f64>| Pt3D(thickness, col_y, col_z);
    let get_origin = |bp: Pt3D, size: Size3D| bp + size / 2.0;

    let mut cells: Vec<Cell> = vec![];

    // --- Exterior Boundary Cells (4) ---
    // All four share the same base_pt; no x-advance between them.
    let ext_con_cell = Cell::new(
        CellType::Boundary(Boundary::Convection, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_con_cell);

    let ext_rad_cell = Cell::new(
        CellType::Boundary(Boundary::Radiation, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_rad_cell);

    let ext_sky_cell = Cell::new(
        CellType::Boundary(Boundary::Sky, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_sky_cell);

    let ext_grn_cell = Cell::new(
        CellType::Boundary(Boundary::Ground, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_grn_cell);

    // --- Exterior Virtual Surface Cell ---
    base_pt.0 += boundary_cell_thickness;
    let ext_virtual_cell = Cell::new(
        CellType::Virtual(BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_virtual_cell);
    base_pt.0 += boundary_cell_thickness;

    // --- Material Cells ---
    let dist_factor = settings.div_distribution_factor;
    if dist_factor > 1.0 || dist_factor < 0.0 {
        return Err(crate::error::Error::InvalidInput(
            "DivDistributionFactor".to_string(),
            dist_factor.to_string(),
            crate::error::InvalidInputReason::OutOfRange,
        )
        .into());
    }

    for mat_arc in construction.layers.iter() {
        let mat_thickness_m = mat_arc.thickness;
        let div_len_m = settings.div_length;

        // 1. Initial guess for divisions
        let mut divs = mat_arc.num_div.unwrap_or(
            ((mat_thickness_m / div_len_m).value().ceil() as i32)
                .max(settings.min_div)
                .min(settings.max_div) as usize,
        );

        // Helper closure to calculate the position of the i-th boundary (0..=divs)
        // proportional to thickness (range 0.0 to 1.0)
        let get_norm_pos = |i: usize, n: usize, factor: f64| -> f64 {
            let frac = i as f64 / n as f64;
            if factor <= 1e-6 {
                return frac; // Pure Uniform
            }

            // Sine distribution formula (clusters points at 0.0 and 1.0)
            // Range of inside term: -pi/2 to -3pi/2
            // sin map: -1.0 to 1.0
            // result map: 0.0 to 1.0
            let sin_pos = ((std::f64::consts::PI * (-frac) - std::f64::consts::PI / 2.0).sin()
                / 2.0)
                - (-1.0 / 2.0); // sin(-pi/2) is -1

            // Linear interpolation between Uniform and Sine
            frac * (1.0 - factor) + sin_pos * factor
        };

        // 2. Safety Check: Ensure the smallest cell is not too small
        // The smallest cell in Sine distribution is at the boundary (i=1)
        while divs > 1 {
            // Calculate the normalized length of the first cell (usually the smallest)
            let norm_len_first =
                get_norm_pos(1, divs, dist_factor) - get_norm_pos(0, divs, dist_factor);
            let test_len = mat_thickness_m * norm_len_first;

            if test_len > settings.adj_length {
                break;
            }
            divs -= 1;
        }

        // 3. Generate Cells
        let mut prev_pos_m = 0.0 * meter; // Start at 0 relative to material layer

        for did in 1..=divs {
            // Calculate current boundary position
            let norm_pos = get_norm_pos(did, divs, dist_factor);
            let curr_pos_m = mat_thickness_m * norm_pos;

            // Length is difference between current boundary and previous boundary
            let length = curr_pos_m - prev_pos_m;
            prev_pos_m = curr_pos_m;

            let size = get_size(length);

            let cell = Cell::new(
                CellType::Material(mat_arc.clone()),
                get_origin(*base_pt, size),
                size,
            );

            cells.push(cell);

            // Update base pointer for the NEXT cell
            base_pt.0 += length;
        }
    }

    // --- Interior Virtual Surface Cell ---
    let int_virtual_cell = Cell::new(
        CellType::Virtual(BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_virtual_cell);

    base_pt.0 += boundary_cell_size.0;

    // --- Interior Boundary Cells ---
    // Interior Convection and Radiation cells share the same geometric position
    // (same base_pt, no x-advance between them) — mirroring the exterior side which
    // stacks Convection/Radiation/Sky/Ground at the same base_pt. Both are reachable
    // from the interior Virtual cell because `neighbor_indices` for Virtual(Interior)
    // collects ALL cells to its right, so the Virtual acts as a hub and the 1-D
    // neighbor walk doesn't need Radiation to be at a distinct x slot.
    let int_con_cell = Cell::new(
        CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_con_cell);

    let int_rad_cell = Cell::new(
        CellType::Boundary(Boundary::Radiation, BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_rad_cell);

    // calc dependent properties
    cells
        .iter_mut()
        .for_each(|c| c.calc_dependent_state(init_bc, settings));

    Ok(cells)
}

/// Build a (nx, ny, 1) cell lattice: `ny` x-columns stacked along y, each with
/// real cross-section (col_y = y_extent/ny, col_z). Column j sits at y-origin
/// j*col_y. Returns the ArcArray3 ready for Surface assembly.
///
/// The 1-D path is the `ny = 1` case: one column at y-origin 0 with
/// `col_y = col_z = sqrt(area)` — byte-identical to the legacy `(N,1,1)` build.
///
/// `build_x_column` advances only `base_pt.x`; the y/z components of `base_pt`
/// are honoured by `get_origin` (`bp + size/2.0`) for every cell origin, so
/// seeding `base_pt.y = j*col_y` places column j correctly without any change
/// to `build_x_column`. Subdivision is x-only, so every column has the same nx.
pub(super) fn build_grid_2d(
    settings: &HamtSettings,
    construction: &Construction,
    surface_area: Area<f64>,
    ny: usize,
    y_extent: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<ndarray::ArcArray3<Cell>> {
    let boundary_cell_thickness = settings.base_cell_size.x();
    let col_y = y_extent / ny as f64;
    let col_z = surface_area.sqrt(); // 2-D out-of-plane extent (z); unchanged convention
    let mut columns: Vec<Vec<Cell>> = Vec::with_capacity(ny);
    for j in 0..ny {
        // Each column starts at the same x offset as the 1-D path (two exterior
        // boundary layers worth of negative x), and y-origin j*col_y.
        let mut base_pt = Pt3D::new(-boundary_cell_thickness * 2.0, col_y * j as f64, 0.0 * meter);
        let col = build_x_column(settings, construction, &mut base_pt, col_y, col_z, init_bc)?;
        columns.push(col);
    }
    let nx = columns[0].len();
    // All columns must have identical nx (subdivision is x-only).
    assert!(
        columns.iter().all(|c| c.len() == nx),
        "non-uniform column length"
    );
    // Flatten C-order for shape (nx, ny, 1): flat[i*ny + j] = columns[j][i] = cell[[i,j,0]].
    let mut flat: Vec<Cell> = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for col in columns.iter() {
            flat.push(col[i].clone());
        }
    }
    Ok(ndarray::ArcArray3::from_shape_vec((nx, ny, 1), flat)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::BoundaryLocation;
    use crate::config::{FACE_Y_MIN, FaceBc};
    use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
    use crate::physics::{Curve, LogY};
    use crate::uom::units::{
        dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, square_meter, w_per_m_k,
    };
    use crate::uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY,
        DIM_WATER_CONTENT, Quantity,
    };

    /// Build a minimal single-layer construction with simple monotone curves,
    /// suitable for exercising the grid topology (no solve is performed).
    fn build_test_construction() -> Arc<Construction> {
        let w_sat = 146.0 * kg_per_m3;
        let n_pts = 10usize;
        let si_x: Vec<_> = (0..=n_pts)
            .map(|k| (k as f64 / n_pts as f64) * dimensionless)
            .collect();
        let si_y: Vec<_> = si_x.iter().map(|&rh| w_sat * rh.value().max(0.0)).collect();
        let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
            Curve::new(si_x.clone(), si_y);

        let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
        let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
            Curve::new(si_x.clone(), mu_y);

        let wc_x: Vec<_> = (0..=n_pts).map(|k| (k as f64 / n_pts as f64) * w_sat).collect();
        let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
        let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
            Curve::new(wc_x.clone(), k_y);

        let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
        let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
            Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid Dw curve");

        let mat = Material {
            name: "GridTestMat".to_string(),
            thickness: 0.1 * meter,
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
            initial_temp: crate::boundary::celsius_to_kelvin(20.0),
            num_div: Some(3),
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
            "GridTestConstruction".to_string(),
            vec!["GridTestMat".to_string()],
            surface_coeff,
        )
        .expect("create construction");
        lib.get_construction("GridTestConstruction").unwrap()
    }

    #[test]
    fn grid_2d_shape_and_sheets() {
        let settings = HamtSettings::default();
        let bc = BoundaryCondition::default();
        let construction = build_test_construction();
        let ny = 4usize;
        let y_extent = 2.0 * meter;

        let surface = Surface::from_construction_2d(
            &settings,
            construction,
            "GridTest".to_string(),
            1.0 * square_meter,
            ny,
            y_extent,
            &bc,
        )
        .expect("build 2-D surface");

        // Shape is (nx, ny, 1) with a genuinely subdivided x-stack.
        let dim = surface.cells.raw_dim();
        let nx = dim[0];
        assert!(nx > 1, "expected nx > 1, got {nx}");
        assert_eq!([dim[0], dim[1], dim[2]], [nx, ny, 1]);

        // Every Material cell spans col_y = y_extent / ny = 0.5 m in y.
        let expected_col_y = (y_extent / ny as f64).value_in(meter); // strip: scalar tolerance compare
        for cell in surface.cells.iter() {
            if matches!(cell.cell_type, CellType::Material(_)) {
                let got = cell.size.y().value_in(meter); // strip: scalar tolerance compare
                assert!(
                    (got - expected_col_y).abs() < 1e-9,
                    "material cell y-size {got} != {expected_col_y}"
                );
            }
        }

        // Per-row boundary sheets: one interior Convection + one interior Virtual
        // per y-row (no global Virtual cell).
        let int_conv = CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior);
        let int_virtual = CellType::Virtual(BoundaryLocation::Interior);
        let count = |ct: &CellType| surface.cells.iter().filter(|c| &c.cell_type == ct).count();
        assert_eq!(count(&int_conv), ny, "interior Convection sheet count");
        assert_eq!(count(&int_virtual), ny, "interior Virtual sheet count");

        // Side (y-min) face stays Adiabatic.
        assert_eq!(surface.face_bcs[FACE_Y_MIN], FaceBc::Adiabatic);
    }
}
