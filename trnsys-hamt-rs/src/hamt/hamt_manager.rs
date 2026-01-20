//! The main orchestrator for the HAMT simulation.

use super::boundary::BoundaryCondition;
use super::surface::Surface;
use crate::hamt::material::{Construction, Material};
use crate::hamt::physics::geometry::{Pt3D, Size3D};
use crate::hamt::uom;
use crate::hamt::uom::dimensions::{Temperature, Time};
use anyhow::Result;
use std::mem;
use std::sync::Arc;
use uom::dimensions::Area;
use uom::dimensions::Length;
use uom::units;

/// Manages the state and execution of the HAMT simulation for a surface.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// Materials registry for this manager.
    pub materials: Vec<Arc<Material>>,

    /// Construction for the surface.
    pub construction: Arc<Construction>,

    /// The surface's state of last converged time step (t-1).
    pub surface_prev: Surface,
    /// The surface's state of current time step (t) in **last iteration**.
    pub surface_curr: Surface,
    /// The surface's state of current time step (t) in **current iteration**.
    pub surface_next: Surface,

    /// Boundary conditions for the simulation.
    pub boundary_condition: BoundaryCondition,

    /// Simulation-wide settings
    pub settings: HamtSettings,
}

impl HamtManager {
    /// Creates a new `HamtManager` and initializes the cell structure for a single surface.
    /// This is a Rust translation of the `InitHeatBalHAMT` subroutine logic.
    pub fn new(
        settings: Option<HamtSettings>,
        initial_boundary: Option<BoundaryCondition>,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
    ) -> Result<Self> {
        let settings = settings.unwrap_or_default();
        let mut boundary = initial_boundary.unwrap_or_default();

        // Propagate constant vapor transfer coefficients from construction definition
        if construction.surface_vapor_coef.is_external_const {
            boundary.vapor_transport_coeff_ext =
                construction.surface_vapor_coef.external_value * units::kg_per_m2_s_pa;
        }
        if construction.surface_vapor_coef.is_internal_const {
            boundary.vapor_transport_coeff_int =
                construction.surface_vapor_coef.internal_value * units::kg_per_m2_s_pa;
        }

        let surface = Surface::from_construction(
            &settings,
            construction.clone(),
            surface_name,
            surface_area,
            &boundary,
        )?;

        Ok(Self {
            settings,
            construction: construction.clone(),
            surface_curr: surface.clone(),
            surface_prev: surface.clone(),
            surface_next: surface,
            boundary_condition: boundary,
            materials: construction.layers.clone(),
        })
    }

    /// Solves for the current timestep by iterating until convergence.
    pub fn solve_timestep(&mut self) -> Result<bool> {
        for _ in 0..self.settings.max_iterations {
            self.surface_next.calculate_iteration(
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            if self
                .surface_curr
                .check_convergence(&self.surface_next, &self.settings)
            {
                mem::swap(&mut self.surface_curr, &mut self.surface_next);
                return Ok(true); // Converged
            }

            mem::swap(&mut self.surface_curr, &mut self.surface_next);
        }

        Ok(false) // Not converged
    }

    /// Executes after the timestep has been calculated.
    pub fn tick_timestep(&mut self) -> Result<()> {
        // The converged state is in `surface_curr`. This becomes the previous state for the next timestep.
        self.surface_prev = self.surface_curr.clone();
        // `surface_next` is scratch space, so its contents don't matter between steps.
        Ok(())
    }
}

/// Global settings for the HAMT simulation.
#[derive(Debug, Clone)]
pub struct HamtSettings {
    /// Timestep [s].
    pub timestep: Time<f64>,
    /// Maximum nonlinear iterations (count).
    pub max_iterations: i32,
    /// Temperature convergence tolerance [K or °C].
    pub temp_convergence_tolerance: Temperature<f64>,
    /// Relative humidity convergence tolerance [-].
    pub rh_convergence_tolerance: f64,
    /// Base Cell Depth [m]
    pub base_cell_size: Size3D,
    /// Flag to enable latent heat coupling (dimensionless).
    pub latent_heat_enabled: bool,
    /// Flag to indicate if rain is present on the exterior surface.
    pub rain_enabled: bool,
    /// Cell division size [m]
    pub div_length: Length<f64>,
    /// Minimum cell amount for a material after division
    pub min_div: i32,
    /// Maximum cell amount for a material after division
    pub max_div: i32,
    /// Allowed length for adjacent check [m]
    pub adj_length: Length<f64>,
    /// user defined depth [m]
    pub user_defined_depth: Option<Length<f64>>,
}

impl Default for HamtSettings {
    fn default() -> Self {
        Self {
            timestep: 3600.0 * units::seconds,
            max_iterations: 150,
            temp_convergence_tolerance: 0.0001 * units::kelvin,
            rh_convergence_tolerance: 0.0001,
            base_cell_size: Pt3D(0.1 * units::meter, 0.0 * units::meter, 0.0 * units::meter),
            latent_heat_enabled: true,
            rain_enabled: false,
            div_length: 0.05 * units::meter,
            min_div: 3,
            max_div: 20,
            adj_length: 0.00005 * units::meter,
            user_defined_depth: None,
        }
    }
}
