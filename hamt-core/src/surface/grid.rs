
use super::*;
use crate::uom::dimensions::Length;
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
