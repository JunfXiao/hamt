
use super::*;
use crate::uom::dimensions::Length;
use crate::uom::{self};
use crate::{
    cell::Boundary,
    material::Construction,
    physics::geometry::{Pt3D, Size3D},
};
use uom::units::meter;

impl Surface {
    pub fn new(
        name: String,
        dimension: Size3D,
        construction_name: String,
        cells: ArcArray3<Cell>,
        settings: &HamtSettings,
    ) -> Self {
        let mut surface = Surface {
            name,
            construction_name,
            area: dimension.y() * dimension.z(),
            thickness: dimension.x(),
            cells,
            size: dimension,
            cells_idx_map: HashMap::new(),
            interior_surface_material_idx: None,
            exterior_surface_material_idx: None,
            user_defined_depth: settings.user_defined_depth,
            user_defined_cell_idx: None,
            face_bcs: [
                crate::config::FaceBc::Active,    // x-min (exterior)
                crate::config::FaceBc::Active,    // x-max (interior)
                crate::config::FaceBc::Adiabatic, // y-min
                crate::config::FaceBc::Adiabatic, // y-max
                crate::config::FaceBc::Adiabatic, // z-min
                crate::config::FaceBc::Adiabatic, // z-max
            ],
        };
        surface.init_cells_idx_map();
        surface
    }

    /// Dividing the cells and create a surface.
    pub fn from_construction(
        settings: &HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
        init_boundary_condition: &BoundaryCondition,
    ) -> Result<Surface> {
        // Initialize original point with the padding for exterior boundary cells
        let boundary_cell_thickness = settings.base_cell_size.x();
        // Use non-zero y/z to avoid zero-overlap in resistance calculation
        let dummy_length = surface_area.sqrt();
        let dummy_height = surface_area.sqrt();
        let boundary_cell_size = Size3D::new(boundary_cell_thickness, dummy_length, dummy_height);

        // base pt is always the point with the smallest x/y/z
        // Offset for 2 exterior layers. First layer is for Conv/Rad/Sky/Ground Cells
        // Second layer is for Virtual Cell.
        let mut base_pt = Size3D::new(-boundary_cell_thickness * 2.0, 0.0 * meter, 0.0 * meter);
        let get_size = |thickness: Length<f64>| Pt3D(thickness, dummy_length, dummy_height);
        let get_origin = |base_pt: Pt3D, size: Size3D| base_pt + size / 2.0;

        // --- Exterior Boundary Cells (5) ---
        let mut cells = vec![];
        let ext_con_cell = Cell::new(
            CellType::Boundary(Boundary::Convection, BoundaryLocation::Exterior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(ext_con_cell);
        let ext_rad_cell = Cell::new(
            CellType::Boundary(Boundary::Radiation, BoundaryLocation::Exterior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(ext_rad_cell);

        let ext_sky_cell = Cell::new(
            CellType::Boundary(Boundary::Sky, BoundaryLocation::Exterior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(ext_sky_cell);

        let ext_grn_cell = Cell::new(
            CellType::Boundary(Boundary::Ground, BoundaryLocation::Exterior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(ext_grn_cell);

        // --- Exterior Virtual Surface Cell ---
        base_pt.0 += boundary_cell_thickness;
        let ext_virtual_cell = Cell::new(
            CellType::Virtual(BoundaryLocation::Exterior),
            get_origin(base_pt, boundary_cell_size),
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
                    get_origin(base_pt, size),
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
            get_origin(base_pt, boundary_cell_size),
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
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(int_con_cell);
        let int_rad_cell = Cell::new(
            CellType::Boundary(Boundary::Radiation, BoundaryLocation::Interior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(int_rad_cell);

        // calc dependent properties
        cells
            .iter_mut()
            .for_each(|c| c.calc_dependent_state(init_boundary_condition, settings));

        // currently, only 1D
        let cells_3d = ArcArray3::from_shape_vec((cells.len(), 1, 1), cells)?;


        Ok(Surface::new(
            surface_name,
            get_size(construction.layers.iter().map(|l| l.thickness).sum()),
            construction.name.clone(),
            cells_3d,
            settings,
        ))
    }

    fn init_cells_idx_map(&mut self) {
        self.cells_idx_map.clear();
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            let cell_type = cell.cell_type.clone();
            self.cells_idx_map
                .entry(cell_type)
                .or_insert(Vec::with_capacity(self.cells.len())) // For performance
                .push([x, y, z]);
        }

        // Cache boundary-adjacent material cell indices (scan x-axis once).
        let nx = self.cells.raw_dim()[0];
        self.exterior_surface_material_idx = (0..nx).find(|&i| {
            matches!(self.cells[[i, 0, 0]].cell_type, CellType::Material(_))
        });
        self.interior_surface_material_idx = (0..nx).rev().find(|&i| {
            matches!(self.cells[[i, 0, 0]].cell_type, CellType::Material(_))
        });

        // default to mid-depth if not provided
        if self.user_defined_depth.is_none() {
            self.user_defined_depth = Some({
                // default to mid-depth if not provided
                
                // Calculate the thickness of the surface based on the first and last cell positions in the x-direction
                let first_cell_pos = self
                    .cells
                    .get([0, 0, 0])
                    .map(|c| c.origin.x() - c.size.x() / 2.0) // Get the left boundary of the first cell
                    .unwrap_or(0.0 * meter);

                let last_cell_pos = self
                    .cells
                    .get([self.cells.len() - 1, 0, 0])
                    .map(|c| c.origin.x() + c.size.x() / 2.0) // Get the right boundary of the last cell
                    .unwrap_or(0.0 * meter);
            
                (first_cell_pos + last_cell_pos) / 2.0
            })
        }
        self.user_defined_cell_idx = self.user_defined_depth.map(|v| {
            let target = Pt3D::new(v, self.size.y(), self.size.z());
            self.compute_interpolation_weights(&target)
        });
    }
}
