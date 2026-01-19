use std::f64::consts::PI;

use super::*;
use crate::hamt::uom;
use crate::hamt::uom::dimensions::Length;
use crate::hamt::{
    cell::Boundary,
    material::Construction,
    physics::geometry::{Pt3D, Size3D},
};
use uom::units::meter;

impl Surface {
    pub fn new(
        name: String,
        area: Area<f64>,
        construction_name: String,
        cells: ArcArray3<Cell>,
    ) -> Self {
        let mut surface = Surface {
            name,
            construction_name,
            area,
            cells,
            cells_idx_map: HashMap::new(),
            ext_vtc: None,
            int_vtc: None,
            user_defined_depth: None,
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
    ) -> Result<Surface> {
        // Initialize original point with the padding for exterior boundary cells
        let boundary_cell_thickness = settings.base_cell_size.x();
        let boundary_cell_size = Size3D::new(boundary_cell_thickness, 0.0 * meter, 0.0 * meter);

        // base pt is always the point with the smallest x/y/z
        // Offset for 2 exterior layers. First layer is for Conv/Rad/Sky/Ground Cells
        // Second layer is for Virtual Cell.
        let mut base_pt = Size3D::new(-boundary_cell_thickness * 2.0, 0.0 * meter, 0.0 * meter);
        // currently, we use a dummy y and z size for all.
        let dummy_length = surface_area.sqrt();
        let dummy_height = surface_area.sqrt();
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

        // --- Material Cells ---
        for mat_arc in construction.layers.iter() {
            // Simplified division calculation
            let mat_thickness_m = mat_arc.thickness;
            let div_len_m = settings.div_length;

            let mut divs = ((mat_thickness_m / div_len_m).value().ceil() as i32)
                .max(settings.min_div)
                .min(settings.max_div);

            // Ensure cell length is not too small
            let sin_neg_pi_ovr_2 = (PI * -0.5).sin();

            while divs > 1 {
                let test_len = mat_thickness_m
                    * (((PI * (-1.0 / divs as f64) - PI / 2.0).sin() / 2.0)
                        - (sin_neg_pi_ovr_2 / 2.0));
                if test_len > settings.adj_length {
                    break;
                }
                divs -= 1;
            }

            for did in 1..=divs {
                let did_f64 = did as f64;
                let divs_f64 = divs as f64;
                // Make cells smaller near the surface using sine distribution
                let length = mat_thickness_m
                    * (((PI * (-did_f64 / divs_f64) - PI / 2.0).sin() / 2.0)
                        - ((PI * (-(did_f64 - 1.0) / divs_f64) - PI / 2.0).sin() / 2.0));

                base_pt.0 += length;

                let size = get_size(length);

                let cell = Cell::new(
                    CellType::Material(mat_arc.clone()),
                    get_origin(base_pt, size),
                    size,
                );

                cells.push(cell);
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

        // --- Interior Boundary Cell ---
        let int_con_cell = Cell::new(
            CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior),
            get_origin(base_pt, boundary_cell_size),
            boundary_cell_size,
        );
        cells.push(int_con_cell);

        // currently, only 1D
        let cells_3d = ArcArray3::from_shape_vec((cells.len(), 1, 1), cells)?;

        let surface = Surface::new(
            surface_name,
            surface_area,
            construction.name.clone(),
            cells_3d,
        );

        Ok(surface)
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
    }
}
