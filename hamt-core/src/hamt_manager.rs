//! The main orchestrator for the HAMT simulation.

use crate::boundary::BoundaryCondition;
use crate::cell::CellType;
use crate::material::{Construction, Material};
use crate::physics::R_V;
use crate::physics::geometry::{Pt3D, Size3D};
use crate::surface::Surface;
use crate::uom::dimensions::{Power, Temperature, Time};
use crate::uom::units::{dimensionless, hours, kelvin, kg_per_m2_s_pa, seconds};
use crate::{physics, uom};
use anyhow::{Context, Result};
use std::mem;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uom::dimensions::Area;
use uom::dimensions::Length;
use uom::units;

/// Manages the state and execution of the HAMT simulation for a surface.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// Materials registry for this self.
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

    pub last_report_time: Time<f64>,
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
        debug!(
            "Initializing HamtManager for surface {}. Boundary conditions: {:#?}",
            surface_name, boundary
        );
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
            last_report_time: 0.0 * hours,
        })
    }

    /// Solves for the current timestep by iterating until convergence.
    pub fn solve_timestep(&mut self) -> Result<bool> {
        // Reapply surface vapor coefficients if not using constnat value
        if !self.construction.surface_vapor_coef.is_external_const {
            self.boundary_condition.vapor_transport_coeff_ext =
                self.boundary_condition.mass_convection_coeff_ext
                    / (R_V * self.boundary_condition.temp_outside_air);
        }
        if !self.construction.surface_vapor_coef.is_internal_const {
            self.boundary_condition.vapor_transport_coeff_int =
                self.boundary_condition.mass_convection_coeff_int
                    / (R_V * self.boundary_condition.temp_inside_air);
        }
        // Previous stats for adaptive time stepping

        let mut stats = ConvergenceStats::default();

        for _ in 0..self.settings.max_iterations {
            self.surface_next.calculate_iteration(
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            mem::swap(&mut self.surface_curr, &mut self.surface_next);

            if stats.converged {
                if stats.rh_converged {
                    return Ok(true); // Converged
                }
            }
        }

        warn!(
            max_delta_t_k = stats.max_delta_t_k,
            max_delta_rh = stats.max_delta_rh,
            max_temp_cell = ?stats.max_temp_cell,
            max_rh_cell = ?stats.max_rh_cell,
            "HAMT timestep did not converge within max iterations"
        );
        Ok(false) // Not converged
    }

    /// Executes after the timestep has been calculated.
    pub fn tick_timestep(&mut self) -> Result<()> {
        // The converged state is in `surface_curr`. This becomes the previous state for the next timestep.
        mem::swap(&mut self.surface_prev, &mut self.surface_curr);

        // After swapping, the previous becomes the current state.
        self.surface_curr
            .sync_state_from(&self.surface_prev)
            .with_context(|| "Syncing current and previous states")?;
        // `surface_next` is scratch space, so its contents don't matter between steps.
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ConvergenceStats {
    converged: bool,
    rh_converged: bool,
    max_delta_t_k: f64,
    max_delta_rh: f64,
    max_temp_cell: Option<[usize; 3]>,
    max_rh_cell: Option<[usize; 3]>,
}
impl Default for ConvergenceStats {
    fn default() -> Self {
        Self {
            converged: false,
            rh_converged: false,
            max_delta_t_k: 9999.0,
            max_delta_rh: 9999.0,
            max_temp_cell: None,
            max_rh_cell: None,
        }
    }
}

fn convergence_stats(
    surface_curr: &Surface,
    surface_prev: &Surface,
    settings: &HamtSettings,
) -> ConvergenceStats {
    let temp_tol = settings.temp_convergence_tolerance.value_in(kelvin);
    let rh_tol = settings.rh_convergence_tolerance;
    let mut temp_converged = true;
    let mut rh_converged = true;
    let mut max_delta_t_k = 0.0;
    let mut max_delta_rh = 0.0;
    let mut max_temp_cell = None;
    let mut max_rh_cell = None;

    for (((x, y, z), cell_curr), cell_prev) in surface_curr
        .cells
        .indexed_iter()
        .zip(surface_prev.cells.iter())
    {
        if !matches!(cell_curr.cell_type, CellType::Material(_)) {
            continue;
        }

        let delta_t_k = (cell_curr.cell_temp - cell_prev.cell_temp)
            .value_in(kelvin)
            .abs();
        if delta_t_k > max_delta_t_k {
            max_delta_t_k = delta_t_k;
            max_temp_cell = Some([x, y, z]);
        }
        if delta_t_k > temp_tol {
            temp_converged = false;
        }

        let delta_rh = (cell_curr.cell_rh - cell_prev.cell_rh).value().abs();
        if delta_rh > max_delta_rh {
            max_delta_rh = delta_rh;
            max_rh_cell = Some([x, y, z]);
        }
        if delta_rh > rh_tol {
            rh_converged = false;
        }
    }

    ConvergenceStats {
        converged: temp_converged,
        rh_converged,
        max_delta_t_k,
        max_delta_rh,
        max_temp_cell,
        max_rh_cell,
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
    /// Clamp for latent heat contribution [W].
    pub max_latent_heat: Power<f64>,
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
            max_iterations: 1500,
            temp_convergence_tolerance: 0.001 * units::kelvin,
            rh_convergence_tolerance: 0.0001,
            base_cell_size: Pt3D(0.1 * units::meter, 0.0 * units::meter, 0.0 * units::meter),
            latent_heat_enabled: true,
            rain_enabled: false,
            max_latent_heat: 100_000.0 * units::watts,
            div_length: 0.05 * units::meter,
            min_div: 3,
            max_div: 20,
            adj_length: 0.00005 * units::meter,
            user_defined_depth: None,
        }
    }
}
