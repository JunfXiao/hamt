
use super::*;
use crate::uom::dimensions::Length;
use crate::uom::{self};
use crate::{
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
        let boundary_cell_thickness = settings.base_cell_size.x();
        // Use non-zero y/z to avoid zero-overlap in resistance calculation
        let dummy_length = surface_area.sqrt();
        let dummy_height = surface_area.sqrt();

        // base pt is always the point with the smallest x/y/z
        // Offset for 2 exterior layers. First layer is for Conv/Rad/Sky/Ground Cells
        // Second layer is for Virtual Cell.
        let mut base_pt = Size3D::new(-boundary_cell_thickness * 2.0, 0.0 * meter, 0.0 * meter);

        let cells = crate::surface::grid::build_x_column(
            settings,
            &construction,
            &mut base_pt,
            dummy_length,
            dummy_height,
            init_boundary_condition,
        )?;

        // currently, only 1D
        let cells_3d = ArcArray3::from_shape_vec((cells.len(), 1, 1), cells)?;

        let get_size = |thickness: Length<f64>| Pt3D(thickness, dummy_length, dummy_height);
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
