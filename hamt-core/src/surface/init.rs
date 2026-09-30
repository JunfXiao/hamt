
use super::*;
use crate::config::{FACE_Z_MAX, FACE_Z_MIN, FaceBc};
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
            interior_surface_material_idx_set: Vec::new(),
            exterior_surface_material_idx_set: Vec::new(),
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

    /// Build a 3-D `(nx, ny, nz)` surface: `ny * nz` x-columns keyed by `(j, k)`,
    /// each with cross-section `(y_extent/ny, z_extent/nz)`. Side faces y-min/y-max
    /// stay `FaceBc::Adiabatic`; z-faces are set from the `z_faces` argument.
    ///
    /// `from_construction_2d` is the `nz = 1` delegate of this routine.
    #[allow(clippy::too_many_arguments)]
    pub fn from_construction_3d(
        settings: &HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
        ny: usize,
        y_extent: Length<f64>,
        nz: usize,
        z_extent: Length<f64>,
        z_faces: [FaceBc; 2],
        init_bc: &BoundaryCondition,
    ) -> Result<Surface> {
        let cells_3d = crate::surface::grid::build_grid_3d(
            settings,
            &construction,
            surface_area,
            ny,
            y_extent,
            nz,
            z_extent,
            init_bc,
        )?;

        let total_thickness: Length<f64> = construction.layers.iter().map(|l| l.thickness).sum();
        let dimension = Pt3D(total_thickness, y_extent, z_extent);

        let mut surface = Surface::new(
            surface_name,
            dimension,
            construction.name.clone(),
            cells_3d,
            settings,
        );
        surface.face_bcs[FACE_Z_MIN] = z_faces[0];
        surface.face_bcs[FACE_Z_MAX] = z_faces[1];
        Ok(surface)
    }

    /// Build a 2-D `(nx, ny, 1)` surface: `ny` x-columns stacked along y, each
    /// with a `y_extent/ny`-tall cross-section. The interior/exterior boundary
    /// sheets are per-column (one Conv/Rad/Sky/Ground/Virtual stack per y-row);
    /// there is no single global Virtual cell. Side faces (y-min/y-max) and the
    /// z-faces stay `FaceBc::Adiabatic` (the `Surface::new` default).
    ///
    /// This is the `nz = 1` delegate of [`Surface::from_construction_3d`].
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
        // Reproduce the existing col_z = sqrt(area) convention exactly.
        let z_extent = surface_area.sqrt();
        Self::from_construction_3d(
            settings,
            construction,
            surface_name,
            surface_area,
            ny,
            y_extent,
            1,
            z_extent,
            [FaceBc::Adiabatic; 2],
            init_bc,
        )
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

        // Cache boundary-adjacent material cell indices.
        //
        // Per-column sets: for each (y,z) column, find the exterior-most (min-x)
        // and interior-most (max-x) Material cell and push its [x,y,z] into the
        // corresponding set.  For a (N,1,1) wall each set has exactly one element.
        let dims = self.cells.raw_dim();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        self.interior_surface_material_idx_set.clear();
        self.exterior_surface_material_idx_set.clear();
        for y in 0..ny {
            for z in 0..nz {
                // exterior: first x with Material
                if let Some(x) = (0..nx).find(|&x| {
                    matches!(self.cells[[x, y, z]].cell_type, CellType::Material(_))
                }) {
                    self.exterior_surface_material_idx_set.push([x, y, z]);
                }
                // interior: last x with Material
                if let Some(x) = (0..nx).rev().find(|&x| {
                    matches!(self.cells[[x, y, z]].cell_type, CellType::Material(_))
                }) {
                    self.interior_surface_material_idx_set.push([x, y, z]);
                }
            }
        }
        // Scalar back-compat: first element's x-index (or None if no material cells).
        self.exterior_surface_material_idx =
            self.exterior_surface_material_idx_set.first().map(|t| t[0]);
        self.interior_surface_material_idx =
            self.interior_surface_material_idx_set.first().map(|t| t[0]);

        // default to mid-depth if not provided
        if self.user_defined_depth.is_none() {
            self.user_defined_depth = Some({
                // Calculate the thickness of the surface based on the first and last cell positions
                // in the x-direction.  Use the real x-extent (nx) rather than cells.len() which is
                // the flat product nx*ny*nz and would give an out-of-bounds index on 2-D grids.
                let nx = self.cells.raw_dim()[0];

                let first_cell_pos = self
                    .cells
                    .get([0, 0, 0])
                    .map(|c| c.origin.x() - c.size.x() / 2.0) // left boundary of the first cell
                    .unwrap_or(0.0 * meter);

                let last_cell_pos = self
                    .cells
                    .get([nx - 1, 0, 0])
                    .map(|c| c.origin.x() + c.size.x() / 2.0) // right boundary of the last cell
                    .unwrap_or(0.0 * meter);

                (first_cell_pos + last_cell_pos) / 2.0
            })
        }
        self.user_defined_cell_idx = self.user_defined_depth.map(|v| {
            // Use the geometric mid-point in y and z so the probe falls in the centre
            // of the wall on 2-D/3-D grids.  For a 1-D (N,1,1) grid the y- and
            // z-interpolation fractions are 0.0 regardless of the target value
            // (find_bracket returns fraction=0 when there is only one cell), so
            // size.y()/2 and size.z()/2 produce the same result as any other y/z
            // target — 1-D probe output is unchanged.
            let target = Pt3D::new(v, self.size.y() / 2.0, self.size.z() / 2.0);
            self.compute_interpolation_weights(&target)
        });
    }
}
