//! The main orchestrator for the HAMT simulation.

use crate::boundary::BoundaryCondition;
use crate::cell::CellType;
use crate::material::{Construction, Material};

use crate::physics::geometry::{Pt3D, Size3D};
use crate::surface::Surface;
use crate::uom::constants::VAPOR_GAS_CONSTANT;
use crate::uom::dimensions::{Power, Temperature, Time};
use crate::uom::units::{hours, kelvin, m_per_s};
use crate::{error, uom};
use anyhow::{Context, Ok, Result};
use std::mem;
use std::sync::Arc;
use tracing::{debug, warn};
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
        // Reapply surface vapor coefficients if not using constant value.
        // When mass_convection_coeff is non-zero it represents h_mass [m/s] and β is derived
        // via β = h_mass / (R_v·T).  When it is zero the caller has set vapor_transport_coeff
        // directly as β [kg/(m²·s·Pa)] and no conversion is needed.
        if !self.construction.surface_vapor_coef.is_external_const
            && self.boundary_condition.mass_convection_coeff_ext.value_in(m_per_s) != 0.0
        {
            self.boundary_condition.vapor_transport_coeff_ext =
                self.boundary_condition.mass_convection_coeff_ext
                    / (VAPOR_GAS_CONSTANT * self.boundary_condition.temp_outside_air);
        }
        if !self.construction.surface_vapor_coef.is_internal_const
            && self.boundary_condition.mass_convection_coeff_int.value_in(m_per_s) != 0.0
        {
            self.boundary_condition.vapor_transport_coeff_int =
                self.boundary_condition.mass_convection_coeff_int
                    / (VAPOR_GAS_CONSTANT * self.boundary_condition.temp_inside_air);
        }

        let mut stats: ConvergenceStats = ConvergenceStats::default();

        let temp_tol = self.settings.temp_convergence_tolerance.value_in(kelvin);
        let rh_tol = self.settings.rh_convergence_tolerance;
        let min_alpha = self.settings.min_relaxation_factor;

        // For early termination: track best residual to detect stalls/oscillation
        let mut best_max_delta = f64::MAX;
        let mut stall_count = 0_i32;
        const STALL_PATIENCE: i32 = 50; // bail out if no improvement for this many iterations

        for iter in 0..self.settings.max_iterations {
            self.surface_next.calculate_iteration(
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            // Adaptive per-cell relaxation: reduce α for cells with large residuals
            for (cell_next, cell_curr) in self
                .surface_next
                .cells
                .iter_mut()
                .zip(self.surface_curr.cells.iter())
            {
                if !matches!(cell_next.cell_type, CellType::Material(_)) {
                    continue;
                }
                let delta_t = (cell_next.cell_temp - cell_curr.cell_temp)
                    .value_in(kelvin)
                    .abs();
                let delta_rh = (cell_next.cell_rh - cell_curr.cell_rh).value().abs();

                // If residual is large (>10× tolerance), tighten relaxation
                if delta_t > temp_tol * 10.0 || delta_rh > rh_tol * 10.0 {
                    cell_next.relaxation_factor =
                        (cell_next.relaxation_factor * 0.9).max(min_alpha);
                } else if delta_t < temp_tol && delta_rh < rh_tol {
                    // Cell is converged — slowly recover toward 1.0
                    cell_next.relaxation_factor =
                        (cell_next.relaxation_factor * 1.02).min(1.0);
                }
            }

            mem::swap(&mut self.surface_curr, &mut self.surface_next);

            if stats.converged && stats.rh_converged {
                if iter > 100 {
                    debug!(
                        iter,
                        max_delta_t_k = stats.max_delta_t_k,
                        max_delta_rh = stats.max_delta_rh,
                        "solve_timestep converged (slow)"
                    );
                }
                return Ok(true);
            }

            // Early termination: detect stall/oscillation
            let current_max = stats.max_delta_t_k.max(stats.max_delta_rh * 1000.0);
            if current_max < best_max_delta * 0.99 {
                // Meaningful improvement (>1%) — reset stall counter
                best_max_delta = current_max;
                stall_count = 0;
            } else {
                stall_count += 1;
                if stall_count >= STALL_PATIENCE {
                    debug!(
                        iter,
                        max_delta_t_k = stats.max_delta_t_k,
                        max_delta_rh = stats.max_delta_rh,
                        "solve_timestep: early termination — residual stalled"
                    );
                    break;
                }
            }
        }

        debug!(
            max_iter = self.settings.max_iterations,
            max_delta_t_k = stats.max_delta_t_k,
            max_delta_rh = stats.max_delta_rh,
            max_temp_cell = ?stats.max_temp_cell,
            max_rh_cell = ?stats.max_rh_cell,
            dt_hours = self.settings.timestep.value_in(hours),
            "solve_timestep did NOT converge"
        );
        Ok(false) // Not converged
    }

    /// Resets all per-cell relaxation factors to 1.0.
    pub fn reset_relaxation_factors(&mut self) {
        for cell in self.surface_curr.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
        for cell in self.surface_next.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
        for cell in self.surface_prev.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
    }


    /// Solves for the current timestep using adaptive substepping.
    /// 
    /// Warning: This function won't tick timestep after solving.
    pub fn solve_timestep_adaptive<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        mut callback_state_change: Option<Box<dyn FnMut(&AdaptiveState<T>) -> Result<()>>>,
    ) -> Result<()> {
        adaptive_state.substep_time = 0.0 * hours;
        let orig_timestep = self.settings.timestep;

        // Reset relaxation factors once at the start of each macro-timestep
        self.reset_relaxation_factors();

        if adaptive_state.split_factor < 1 {
            adaptive_state.split_factor = 1;
        }
        if adaptive_state.split_factor > adaptive_state.max_split_factor as i32 {
            adaptive_state.split_factor = adaptive_state.max_split_factor as i32;
        }

        let timestep_for_split = |split_factor: i32| orig_timestep / 2.0_f64.powi(split_factor - 1);

        // Track remaining substeps as integer rounds so the sum lands exactly on orig_timestep.
        let mut rounds_remaining = 1_i32 << (adaptive_state.split_factor - 1);
        self.settings.timestep = timestep_for_split(adaptive_state.split_factor);

        debug!(
            orig_dt_hours = orig_timestep.value_in(hours),
            split_factor = adaptive_state.split_factor,
            rounds_remaining,
            sub_dt_hours = self.settings.timestep.value_in(hours),
            "solve_timestep_adaptive: BEGIN"
        );

        if let Some(ref mut prev_surface_state) = adaptive_state.prev_surface_state {
            prev_surface_state.sync_state_from(&self.surface_prev)?;
        } else {
            adaptive_state.prev_surface_state = Some(self.surface_prev.clone());
        }

        let mut total_substeps = 0_u32;

        while rounds_remaining > 0 {
            if adaptive_state.retry_wait_time > 0 {
                adaptive_state.retry_wait_time -= 1;
            }

            adaptive_state.reset_time_in_contineous = true;
            if let Some(ref mut callback) = callback_state_change {
                callback(adaptive_state)?;
            }

            // Solving the timestep
            let mut solve_in_one_timestep = true;
            while !self.solve_timestep()? {
                solve_in_one_timestep = false;
                adaptive_state.reset_time_in_contineous = false;
                adaptive_state.split_factor += 1;
                if adaptive_state.split_factor > adaptive_state.max_split_factor as i32 {
                    debug!(
                        split_factor = adaptive_state.split_factor,
                        max_split = adaptive_state.max_split_factor,
                        rounds_remaining,
                        total_substeps,
                        "solve_timestep_adaptive: CONVERGENCE FAILED — max split exceeded"
                    );
                    self.settings.timestep = orig_timestep;
                    return Err(error::Error::Other(format!(
                        "Convergence failed with splitting factor {} at {:.2} hours",
                        adaptive_state.split_factor,
                        (adaptive_state.current_time / hours).value()
                    ))
                    .into());
                }
                debug!(
                    split_factor = adaptive_state.split_factor,
                    rounds_remaining = rounds_remaining * 2,
                    sub_dt_hours = timestep_for_split(adaptive_state.split_factor).value_in(hours),
                    "solve_timestep_adaptive: splitting further"
                );

                adaptive_state.time_in_contineous_split += 1;
                adaptive_state.retry_wait_time +=
                    adaptive_state.time_in_contineous_split * adaptive_state.split_factor;

                // Split by 2 -> remaining rounds double (k -> 2k).
                rounds_remaining *= 2;
                self.settings.timestep = timestep_for_split(adaptive_state.split_factor);
                if let Some(ref mut callback) = callback_state_change {
                    callback(adaptive_state)?;
                }

                // Preserve learned relaxation factors across the state reset
                let saved_alphas: Vec<f64> = self
                    .surface_curr
                    .cells
                    .iter()
                    .map(|c| c.relaxation_factor)
                    .collect();

                self.surface_curr
                    .sync_state_from(&self.surface_prev.clone())?;
                self.surface_next
                    .sync_state_from(&self.surface_prev.clone())?;

                // Restore learned relaxation factors
                for (cell, &alpha) in self.surface_curr.cells.iter_mut().zip(saved_alphas.iter()) {
                    cell.relaxation_factor = alpha;
                }
                for (cell, &alpha) in self.surface_next.cells.iter_mut().zip(saved_alphas.iter()) {
                    cell.relaxation_factor = alpha;
                }
            }

            let actual_step = self.settings.timestep;
            adaptive_state.substep_time += actual_step;
            adaptive_state.current_time += actual_step;
            rounds_remaining -= 1;
            total_substeps += 1;
            self.tick_timestep()?;
            

            if adaptive_state.reset_time_in_contineous {
                adaptive_state.time_in_contineous_split = 0;
            }

            if solve_in_one_timestep {
                adaptive_state.retry_wait_time -= adaptive_state.split_factor;
            }

            // Backtracking: only merge steps when aligned (even rounds remaining).
            if solve_in_one_timestep
                && adaptive_state.split_factor > 1
                && adaptive_state.retry_wait_time <= 0
                && rounds_remaining % 2 == 0
            {
                adaptive_state.retry_wait_time = 0;
                adaptive_state.split_factor -= 1;
                rounds_remaining /= 2;
                self.settings.timestep = timestep_for_split(adaptive_state.split_factor);
            }
        }

        debug!(
            total_substeps,
            final_split_factor = adaptive_state.split_factor,
            "solve_timestep_adaptive: DONE"
        );

        // Pretend we never substepped by "rewinding" the ticking and substepping.
        // First, swap back the surface_curr and surface_prev
        mem::swap(&mut self.surface_prev, &mut self.surface_curr);
        // Then, restore the previous state from before substepping
        self.surface_prev
            .sync_state_from(adaptive_state.prev_surface_state.as_ref().unwrap())?;
        self.settings.timestep = orig_timestep;
        adaptive_state.substep_time = orig_timestep;
        Ok(())
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
    /// Cell division distribution factor.
    ///
    /// When set to 1.0, the cells are divided by sine distribution.
    /// When set to 0.0, the cells are divided by linear distribution.
    pub div_distribution_factor: f64,
    /// user defined depth [m]
    pub user_defined_depth: Option<Length<f64>>,
    /// Minimum per-cell under-relaxation factor (floor) for nonlinear iteration.
    ///
    /// Each cell starts with α=1.0 (no relaxation). When a cell's iteration
    /// residuals indicate divergence, α is reduced adaptively but never below
    /// this floor. Lower values allow more aggressive damping at the cost of
    /// slower convergence.
    pub min_relaxation_factor: f64,
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
            div_distribution_factor: 0.5,
            user_defined_depth: None,
            min_relaxation_factor: 0.5,
        }
    }
}

pub struct AdaptiveState<T> 
{
    pub substep_time: Time<f64>,
    pub current_time: Time<f64>,
    pub split_factor: i32,
    pub max_split_factor: i32,
    pub data: T,
    pub retry_wait_time: i32,
    pub time_in_contineous_split: i32,
    pub reset_time_in_contineous: bool,
    pub prev_surface_state: Option<Surface>,
}

impl<T> Default for AdaptiveState<T>
where T:Default
{
    fn default() -> Self {
        Self {
            substep_time: 0.0 * units::seconds,
            current_time: 0.0 * units::seconds,
            split_factor: 1,
            max_split_factor: 12,
            data: T::default(),
            retry_wait_time: 0,
            time_in_contineous_split: 0,
            reset_time_in_contineous: false,
            prev_surface_state: None,
        }
    }
}
