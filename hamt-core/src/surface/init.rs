
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
    ///
    /// The 1-D path: delegates to [`Surface::from_construction_2d`] with `ny = 1`
    /// and `y_extent = sqrt(area)`, which builds a single x-column at y-origin 0
    /// with `col_y = col_z = sqrt(area)` — byte-identical to the legacy
    /// `(N,1,1)` build.
    pub fn from_construction(
        settings: &HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
        init_boundary_condition: &BoundaryCondition,
    ) -> Result<Surface> {
        Self::from_construction_2d(
            settings,
            construction,
            surface_name,
            surface_area,
            1,
            surface_area.sqrt(),
            init_boundary_condition,
        )
    }

    /// Build a 2-D `(nx, ny, 1)` surface: `ny` x-columns stacked along y, each
    /// with a `y_extent/ny`-tall cross-section. The interior/exterior boundary
    /// sheets are per-column (one Conv/Rad/Sky/Ground/Virtual stack per y-row);
    /// there is no single global Virtual cell. Side faces (y-min/y-max) and the
    /// z-faces stay `FaceBc::Adiabatic` (the `Surface::new` default).
    ///
    /// `from_construction` is the `ny = 1` case of this routine.
    pub fn from_construction_2d(
        settings: &HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
        ny: usize,
        y_extent: Length<f64>,
        init_bc: &BoundaryCondition,
    ) -> Result<Surface> {
        let cells_3d = crate::surface::grid::build_grid_2d(
            settings,
            &construction,
            surface_area,
            ny,
            y_extent,
            init_bc,
        )?;

        // Surface bounding box: x = total construction thickness; y = y_extent;
        // z = sqrt(area) (the out-of-plane extent used by every column).
        let col_z = surface_area.sqrt();
        let total_thickness: Length<f64> = construction.layers.iter().map(|l| l.thickness).sum();
        let dimension = Pt3D(total_thickness, y_extent, col_z);

        Ok(Surface::new(
            surface_name,
            dimension,
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
