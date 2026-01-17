//! The main orchestrator for the HAMT simulation.

use super::boundary::BoundaryCondition;
use super::surface::Surface;
use crate::hamt::cell::{self, Boundary, BoundaryLocation, Cell, CellType, ADJACENT_CELLS_MAX};
use crate::hamt::material::{Construction, Material};
use crate::hamt::physics;
use crate::hamt::physics::geometry::{Pt3D, Size3D};
use crate::trnsys::error::TrnSysError;
use anyhow::{Context, Ok, Result};
use ndarray::{s, ArcArray3, ViewRepr};
use std::f64::consts::PI;
use std::sync::Arc;
use std::{default, mem};

/// Manages the state and execution of the HAMT simulation for a surface.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// Materials registry for this manager.
    pub materials: Vec<Arc<Material>>,

    /// Construction for the surface.
    pub construction: Arc<Construction>,

    pub surface_prev: Surface,
    pub surface_curr: Surface,
    pub surface_next: Surface,

    /// Boundary conditions for the simulation.
    pub boundary_condition: BoundaryCondition,

    /// Simulation-wide settings
    pub settings: HamtSettings,
}

const ADJ_DIST: f64 = 0.00005; // Allowable distance between two cells
const DIV_SIZE: f64 = 0.05; // Simplified default division size [m]
const DIV_MIN: i32 = 1; // Simplified minimum divisions

impl HamtManager {
    /// Creates a new `HamtManager` and initializes the cell structure for a single surface.
    /// This is a Rust translation of the `InitHeatBalHAMT` subroutine logic.
    pub fn new(
        settings: Option<HamtSettings>,
        initial_boundary: Option<BoundaryCondition>,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: f64,
    ) -> Result<Self> {
        let settings = settings.unwrap_or_default();
        let initial_boundary = initial_boundary.unwrap_or_default();

        let surface = HamtManager::initialize_surface(
            &settings,
            construction.clone(),
            surface_name,
            surface_area,
        )?;

        Ok(Self {
            settings,
            construction: construction.clone(),
            surface_curr: surface.clone(),
            surface_prev: surface.clone(),
            surface_next: surface,
            boundary_condition: initial_boundary,
            materials: construction.layers.clone(),
        })
    }

    /// Dividing the cells and create a surface.
    pub fn initialize_surface(
        settings: &HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: f64,
    ) -> Result<Surface> {
        // Initialize original point with the padding for exterior boundary cells
        let boundary_cell_thickness: f64 = settings.base_cell_size.x();
        let boundary_cell_size = Size3D::new(boundary_cell_thickness, 0.0, 0.0);

        // base pt is always the point with the smallest x/y/z
        // Offset for 2 exterior layers. First layer is for Conv/Rad/Sky/Ground Cells
        // Second layer is for Virtual Cell.
        let mut base_pt = Size3D::new(-boundary_cell_thickness * 2.0, 0.0, 0.0);
        // currently, we use a dummy y and z size for all.
        let dummy_length = surface_area.sqrt();
        let dummy_height = surface_area.sqrt();
        let get_size = |thickness: f64| Pt3D(thickness, dummy_length, dummy_height);
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
            let mut divs = ((mat_arc.thickness / settings.div_length).ceil() as i32)
                .max(settings.min_div)
                .min(settings.max_div);

            // Ensure cell length is not too small
            let sin_neg_pi_ovr_2 = (PI * -0.5).sin();

            while divs > 1 {
                let test_len = mat_arc.thickness
                    * (((PI * (-1.0 / divs as f64) - PI / 2.0).sin() / 2.0)
                        - (sin_neg_pi_ovr_2 / 2.0));
                if test_len > ADJ_DIST {
                    break;
                }
                divs -= 1;
            }

            for did in 1..=divs {
                let did_f64 = did as f64;
                let divs_f64 = divs as f64;
                // Make cells smaller near the surface using sine distribution
                let length = mat_arc.thickness
                    * (((PI * (-did_f64 / divs_f64) - PI / 2.0).sin() / 2.0)
                        - ((PI * (-(did_f64 - 1.0) / divs_f64) - PI / 2.0).sin() / 2.0));

                base_pt.0 += length;

                let size = get_size(length);

                let mut cell = Cell::new(
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

    /// Executes one iteration of the timestep by producing `surface_next` from `surface_curr` (with `surface_prev` available if needed).
    /// After this call, `surface_curr` and `surface_next` are swapped (iteration rotation).
    pub fn iter_in_timestep(&mut self) -> Result<()> {
        self.surface_next.calculate_timestep(
            &self.settings,
            &self.boundary_condition,
            &self.surface_prev.cells,
            &self.surface_curr.cells,
        )?;

        // iteration-level rotation: curr <-> next
        mem::swap(&mut self.surface_curr, &mut self.surface_next);
        Ok(())
    }

    /// Executes after the timestep has been calculated.
    pub fn tick_timestep(&mut self) -> Result<()> {
        // Current timestep ends. Now, `surface_next` becomes `surface_prev`.
        mem::swap(&mut self.surface_prev, &mut self.surface_next);
        // `surface_curr` should have the value of the previous timestep's `surface_next`.
        self.surface_curr.override_cell_state(&self.surface_prev);

        Ok(())
    }
}

/// Global settings for the HAMT simulation.
#[derive(Debug, Clone)]
pub struct HamtSettings {
    /// Timestep [s].
    pub timestep: f64,
    /// Maximum nonlinear iterations (count).
    pub max_iterations: i32,
    /// Temperature convergence tolerance [K or °C].
    pub temp_convergence_tolerance: f64,
    /// Relative humidity convergence tolerance [-].
    pub rh_convergence_tolerance: f64,
    /// Base Cell Depth [m]
    pub base_cell_size: Size3D,
    /// Flag to enable latent heat coupling (dimensionless).
    pub latent_heat_enabled: bool,
    /// Flag to indicate if rain is present on the exterior surface.
    pub rain_enabled: bool,
    /// Cell division size [m]
    pub div_length: f64,
    /// Minimum cell amount for a material after division
    pub min_div: i32,
    /// Maximum cell amount for a material after division
    pub max_div: i32,
    /// Allowed length for adjacent check [m]
    pub adj_length: f64,
}

impl Default for HamtSettings {
    fn default() -> Self {
        Self {
            timestep: 3600.0,
            max_iterations: 150,
            temp_convergence_tolerance: 0.0001,
            rh_convergence_tolerance: 0.0001,
            base_cell_size: Pt3D(0.1, 0.0, 0.0),
            latent_heat_enabled: true,
            rain_enabled: false,
            div_length: 0.05,
            min_div: 3,
            max_div: 20,
            adj_length: 0.00005,
        }
    }
}
