
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

/// Build a `(nx, ny, nz)` cell lattice: `ny * nz` x-columns, each with
/// cross-section `(col_y, col_z) = (y_extent/ny, z_extent/nz)`. Column `(j, k)`
/// sits at y-origin `j * col_y`, z-origin `k * col_z`.
///
/// The flat storage is C-order: `flat[i*ny*nz + j*nz + k] == cell[[i,j,k]]`.
///
/// `build_x_column` advances only `base_pt.x`; seeding `base_pt.(y,z)` for each
/// `(j, k)` pair places the column correctly without any change to `build_x_column`.
/// Every column has the same `nx` (subdivision is x-only).
///
/// The 2-D `(nx, ny, 1)` path is `nz = 1, z_extent = sqrt(area)`.
/// The 1-D `(N, 1, 1)` path is `ny = 1, nz = 1` with `col_y = col_z = sqrt(area)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // `i` indexes column[j,k][i]; not indexing one collection
pub(super) fn build_grid_3d(
    settings: &HamtSettings,
    construction: &Construction,
    _surface_area: Area<f64>,
    ny: usize,
    y_extent: Length<f64>,
    nz: usize,
    z_extent: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<ndarray::ArcArray3<Cell>> {
    let boundary_cell_thickness = settings.base_cell_size.x();
    let col_y = y_extent / ny as f64;
    let col_z = z_extent / nz as f64;

    // Build ny * nz x-columns keyed by (j, k).
    // Use a flat Vec<Vec<Cell>> of length ny*nz, accessed as columns[j*nz + k].
    let mut columns: Vec<Vec<Cell>> = Vec::with_capacity(ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            let mut base_pt = Pt3D::new(
                -boundary_cell_thickness * 2.0,
                col_y * j as f64,
                col_z * k as f64,
            );
            let col =
                build_x_column(settings, construction, &mut base_pt, col_y, col_z, init_bc)?;
            columns.push(col);
        }
    }
    let nx = columns[0].len();
    assert!(
        columns.iter().all(|c| c.len() == nx),
        "non-uniform column length"
    );

    // Flatten C-order for shape (nx, ny, nz):
    //   flat[i*ny*nz + j*nz + k] = cell[[i, j, k]]
    //   The build loop above is k-outer, j-inner, so column (j,k) sits at
    //   columns[k*ny + j].
    let mut flat: Vec<Cell> = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // strip: integer index into column (j,k), no UoM involved.
                flat.push(columns[k * ny + j][i].clone());
            }
        }
    }
    Ok(ndarray::ArcArray3::from_shape_vec((nx, ny, nz), flat)?)
}

/// Build a `(nx, ny, 1)` cell lattice: `ny` x-columns stacked along y, each with
/// real cross-section `(col_y = y_extent/ny, col_z = sqrt(surface_area))`.
///
/// This is the `nz = 1` delegate of [`build_grid_3d`], reproducing the existing
/// 2-D geometry exactly (same `col_z = sqrt(area)` convention).
///
/// The 1-D path is the `ny = 1` case: one column at y-origin 0 with
/// `col_y = col_z = sqrt(area)` — byte-identical to the legacy `(N,1,1)` build.
///
/// Kept as a named API for `from_construction_2d` call sites and tests; the
/// production path calls `build_grid_3d(nz=1)` directly via `from_construction_3d`.
#[allow(dead_code)] // called from tests and retained as a named 2-D API
pub(super) fn build_grid_2d(
    settings: &HamtSettings,
    construction: &Construction,
    surface_area: Area<f64>,
    ny: usize,
    y_extent: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<ndarray::ArcArray3<Cell>> {
    // Reproduce the existing col_z = sqrt(area) convention exactly.
    let z_extent = surface_area.sqrt();
    build_grid_3d(
        settings,
        construction,
        surface_area,
        ny,
        y_extent,
        1,
        z_extent,
        init_bc,
    )
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

    /// Verify build_grid_3d produces the correct (nx, ny, nz) shape and that the
    /// C-order flatten places cell [i,j,k] at z-origin col_z*k.
    #[test]
    fn build_grid_3d_has_correct_dims_and_z_origin() {
        let settings = HamtSettings::default();
        let bc = BoundaryCondition::default();
        let construction = build_test_construction();
        let ny = 2usize;
        let nz = 3usize;
        let y_extent = 1.0 * meter;
        let z_extent = 1.5 * meter;
        let area = y_extent * z_extent;
        let col_z = z_extent / nz as f64;

        let cells = build_grid_3d(
            &settings,
            &construction,
            area,
            ny,
            y_extent,
            nz,
            z_extent,
            &bc,
        )
        .expect("build_grid_3d must succeed");

        let dim = cells.raw_dim();
        let nx = dim[0];
        assert!(nx > 1, "expected nx > 1, got {nx}");
        assert_eq!([dim[0], dim[1], dim[2]], [nx, ny, nz], "shape mismatch");

        // Pick a material cell at each k and verify its z-origin advances by col_z*k.
        // Find the first material x-index (same for all columns).
        let first_mat_x = (0..nx)
            .find(|&x| matches!(cells[[x, 0, 0]].cell_type, CellType::Material(_)))
            .expect("no material cells found");

        let col_z_m = col_z.value_in(meter); // strip: scalar tolerance compare
        for k in 0..nz {
            // z-origin of the cell = col_z * k + col_z/2 (cell is centred within its slab).
            let expected_z = col_z_m * k as f64 + col_z_m / 2.0;
            let got = cells[[first_mat_x, 0, k]].origin.z().value_in(meter); // strip: scalar tolerance compare
            assert!(
                (got - expected_z).abs() < 1e-9,
                "k={k}: expected z-origin {expected_z:.6}, got {got:.6}"
            );
        }
    }

    /// Verify build_grid_2d is byte-identical to build_grid_3d with nz=1 and
    /// z_extent = sqrt(area), at every cell index [i,j,0].
    #[test]
    fn build_grid_2d_equals_build_grid_3d_nz1() {
        use crate::config::FaceBc;
        let settings = HamtSettings::default();
        let bc = BoundaryCondition::default();
        let construction = build_test_construction();
        let ny = 3usize;
        let y_extent = 1.2 * meter;
        let area: crate::uom::dimensions::Area<f64> = 1.0 * square_meter;
        let z_extent = area.sqrt();

        let cells_2d = build_grid_2d(&settings, &construction, area, ny, y_extent, &bc)
            .expect("build_grid_2d must succeed");

        let cells_3d = build_grid_3d(
            &settings,
            &construction,
            area,
            ny,
            y_extent,
            1,
            z_extent,
            &bc,
        )
        .expect("build_grid_3d must succeed");

        let dim2 = cells_2d.raw_dim();
        let dim3 = cells_3d.raw_dim();
        assert_eq!(
            [dim2[0], dim2[1], dim2[2]],
            [dim3[0], dim3[1], dim3[2]],
            "shape mismatch between build_grid_2d and build_grid_3d(nz=1)"
        );

        let (nx, ny_dim, _nz_dim) = (dim2[0], dim2[1], dim2[2]);
        for i in 0..nx {
            for j in 0..ny_dim {
                let c2 = &cells_2d[[i, j, 0]];
                let c3 = &cells_3d[[i, j, 0]];
                assert_eq!(
                    c2.cell_type, c3.cell_type,
                    "cell_type mismatch at [{i},{j},0]"
                );
                let o2 = c2.origin;
                let o3 = c3.origin;
                let dx = (o2.x() - o3.x()).value_in(meter).abs(); // strip: scalar tolerance compare
                let dy = (o2.y() - o3.y()).value_in(meter).abs(); // strip: scalar tolerance compare
                let dz = (o2.z() - o3.z()).value_in(meter).abs(); // strip: scalar tolerance compare
                assert!(dx < 1e-12, "[{i},{j},0] x-origin mismatch: {dx}");
                assert!(dy < 1e-12, "[{i},{j},0] y-origin mismatch: {dy}");
                assert!(dz < 1e-12, "[{i},{j},0] z-origin mismatch: {dz}");
                let sz2 = c2.size;
                let sz3 = c3.size;
                let sdx = (sz2.x() - sz3.x()).value_in(meter).abs(); // strip: scalar tolerance compare
                let sdy = (sz2.y() - sz3.y()).value_in(meter).abs(); // strip: scalar tolerance compare
                let sdz = (sz2.z() - sz3.z()).value_in(meter).abs(); // strip: scalar tolerance compare
                assert!(sdx < 1e-12, "[{i},{j},0] x-size mismatch: {sdx}");
                assert!(sdy < 1e-12, "[{i},{j},0] y-size mismatch: {sdy}");
                assert!(sdz < 1e-12, "[{i},{j},0] z-size mismatch: {sdz}");
            }
        }
        // Also confirm the type is unused here (silence unused import)
        let _ = FaceBc::Adiabatic;
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
