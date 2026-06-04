//! The main orchestrator for the HAMT simulation.

use crate::anderson::AndersonAccelerator;
use crate::boundary::BoundaryCondition;
use crate::cell::CellType;
use crate::material::{Construction, Material};

use crate::cell::BoundaryLocation;
use crate::physics::geometry::{Pt3D, Size3D};
use crate::surface::Surface;
use crate::uom::Quantity;
use crate::uom::constants::VAPOR_GAS_CONSTANT;
use crate::uom::dimensions::{
    ArealEnergyDensity, ArealMassDensity, Dimensionless, EnergyFluxDensity, MassFluxDensity, Power,
    Temperature, Time,
};
use crate::uom::units::{
    dimensionless, hours, j_per_m2, kelvin, kg_per_m2, kg_per_m2_s, m_per_s, seconds, w_per_m2,
};
use crate::{error, uom};
use anyhow::{Context, Ok, Result};
use std::ops::Div;
use std::sync::Arc;
use std::{cell, mem};
use tracing::{debug, info, warn};
use uom::dimensions::Area;
use uom::dimensions::Length;
use uom::units;

const MAX_ALPHA: f64 = 1.5;
pub type TempRate<S> = <Temperature<S> as Div<Time<S>>>::Output;
pub type RhRate<S> = <Dimensionless<S> as Div<Time<S>>>::Output;
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

    /// Converged state from two macro-steps ago (t-2). Used together with
    /// `surface_prev` (t-1) to seed the next macro-step's iteration with a
    /// linear time extrapolation instead of the previous-step state.
    /// `None` until two successful macro-steps have completed.
    pub surface_prev_prev_macro: Option<Surface>,

    /// Macro-timestep duration that just finished. Pairs with
    /// `surface_prev_prev_macro` to scale the time-extrapolation guess to the
    /// upcoming macro-step's Δt. `None` until at least one macro-step has been
    /// solved successfully.
    pub dt_prev_macro: Option<Time<f64>>,
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
            surface_prev_prev_macro: None,
            dt_prev_macro: None,
        })
    }

    /// Solves for the current timestep by iterating until convergence.
    /// Dispatches to the configured inner solver.
    pub fn solve_timestep(&mut self) -> Result<bool> {
        match self.settings.solver_method {
            SolverMethod::BlockJacobi => self.solve_timestep_block_jacobi(),
            SolverMethod::BlockTridiagonal => {
                if !self.settings.coupled_heat_moisture_enabled {
                    warn!(
                        "BlockTridiagonal requires coupled_heat_moisture_enabled=true; \
                         falling back to BlockJacobi for this step"
                    );
                    return self.solve_timestep_block_jacobi();
                }
                self.solve_timestep_block_tridiag()
            }
        }
    }

    /// Direct block-tridiagonal solve. See `solver::block_tridiag` for the
    /// kernel and `surface::update::solve_step_block_tridiag` for the 1-D
    /// wall assembly.
    fn solve_timestep_block_tridiag(&mut self) -> Result<bool> {
        // The outer loop is much shorter than BlockJacobi: it only iterates to
        // convergence on the nonlinearity (Psat(T), dwdphi(φ), L_eff(φ)).
        // Cap at a small number; if it doesn't converge in ~20 iters the
        // inputs are pathological.
        const MAX_OUTER: i32 = 200;

        let mut stats = ConvergenceStats::default();

        // Fully utilize the dimension definition of surface_curr.
        // Allocate iteration history buffers outside the loop to avoid reallocation.
        let n_cells = self.surface_curr.cells.len();
        let mut prev_delta_t = vec![0.0f64; n_cells];
        let mut prev_delta_rh = vec![0.0f64; n_cells];

        for iter in 0..MAX_OUTER {
            // 1. Pass surface_prev (physical time t-1) for the inertial term (tcap/dt).
            // 2. Pass surface_curr (iteration step k) to compute nonlinear coefficients and Jacobian.
            // 3. Write computation results to surface_next (iteration step k+1).
            self.surface_next.calculate_iteration_block_tridiag(
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            // Aitken Δ² dynamic relaxation calculation (Component-wise)
            for (i, (cell_next, cell_curr)) in self
                .surface_next
                .cells
                .iter_mut()
                .zip(self.surface_curr.cells.iter())
                .enumerate()
            {
                if !matches!(cell_next.cell_type, CellType::Material(_)) {
                    continue;
                }

                // Extract scalar increments (strip units for pure numerical ratio calculations)
                let delta_t_curr =
                    (cell_next.cell_temp - cell_curr.cell_temp).value_in(crate::uom::units::kelvin);
                let delta_rh_curr = (cell_next.cell_rh - cell_curr.cell_rh).value();

                let mut next_alpha = cell_curr.relaxation_factor;

                if iter > 0 {
                    let delta_t_prev = prev_delta_t[i];
                    let delta_rh_prev = prev_delta_rh[i];

                    let mut alpha_t = next_alpha;
                    let mut alpha_rh = next_alpha;

                    // Calculate the Aitken decay rate for temperature
                    if delta_t_prev.abs() > 1e-12 {
                        let r_t = delta_t_curr / delta_t_prev;
                        if (1.0 - r_t) > 1e-3 {
                            alpha_t = next_alpha / (1.0 - r_t);
                        } else {
                            // r_t >= 1.0 (diverging) or r_t approaches 1 (stagnation)
                            // Strictly enforce minimum damping to brake
                            alpha_t = 0.0;
                        }
                    }

                    // Calculate the Aitken decay rate for relative humidity
                    if delta_rh_prev.abs() > 1e-12 {
                        let r_rh = delta_rh_curr / delta_rh_prev;
                        if (1.0 - r_rh) > 1e-3 {
                            alpha_rh = next_alpha / (1.0 - r_rh);
                        } else {
                            alpha_rh = 0.0;
                        }
                    }

                    // Catch negative overshoot and strictly clamp to the minimum safe value
                    if alpha_t <= 0.0 {
                        alpha_t = self.settings.min_relaxation_factor;
                    }
                    if alpha_rh <= 0.0 {
                        alpha_rh = self.settings.min_relaxation_factor;
                    }

                    // Conservative strategy: take the smaller alpha between temperature and humidity
                    // to ensure the overall stability of the cross-coupled system
                    next_alpha = alpha_t.min(alpha_rh);
                }

                // Apply slight inertial smoothing to alpha itself to prevent excessive
                // acceleration/deceleration jumps at extreme nonlinear breakpoints
                let smoothed_alpha = 0.5 * cell_curr.relaxation_factor + 0.5 * next_alpha;

                // Assign the safely clamped dynamic alpha to the next iteration
                cell_next.relaxation_factor =
                    smoothed_alpha.clamp(self.settings.min_relaxation_factor, MAX_ALPHA);

                // Update history buffers for the next iteration's calculation
                prev_delta_t[i] = delta_t_curr;
                prev_delta_rh[i] = delta_rh_curr;
            }

            // Iteration advancement: overwrite the k state with the k+1 state for the next loop
            std::mem::swap(&mut self.surface_curr, &mut self.surface_next);

            // Tolerance check (only used for the exit condition, no longer involved in relaxation control)
            if stats.converged && stats.rh_converged {
                if iter > 4 {
                    debug!(
                        iter,
                        max_dt_k = stats.max_delta_t_k,
                        max_drh = stats.max_delta_rh,
                        pinned_t_cell = ?stats.max_temp_cell,
                        pinned_rh_cell = ?stats.max_rh_cell,
                        n_cells = self.surface_curr.cells.len(),
                        dt_hours = self.settings.timestep.value_in(hours),
                        exit_reason = "converged",
                        "solve_timestep_block_tridiag exit"
                    );
                }
                return Ok(true);
            }
        }

        debug!(
            iter = MAX_OUTER,
            max_dt_k = stats.max_delta_t_k,
            max_drh = stats.max_delta_rh,
            pinned_t_cell = ?stats.max_temp_cell,
            pinned_rh_cell = ?stats.max_rh_cell,
            n_cells = self.surface_curr.cells.len(),
            dt_hours = self.settings.timestep.value_in(hours),
            exit_reason = "max_outer",
            "solve_timestep_block_tridiag did NOT converge"
        );
        Ok(false)
    }

    fn solve_timestep_block_jacobi(&mut self) -> Result<bool> {
        // Reapply surface vapor coefficients if not using constant value.
        // When mass_convection_coeff is non-zero it represents h_mass [m/s] and β is derived
        // via β = h_mass / (R_v·T).  When it is zero the caller has set vapor_transport_coeff
        // directly as β [kg/(m²·s·Pa)] and no conversion is needed.
        if !self.construction.surface_vapor_coef.is_external_const
            && self
                .boundary_condition
                .mass_convection_coeff_ext
                .value_in(m_per_s)
                != 0.0
        {
            self.boundary_condition.vapor_transport_coeff_ext =
                self.boundary_condition.mass_convection_coeff_ext
                    / (VAPOR_GAS_CONSTANT * self.boundary_condition.temp_outside_air);
        }
        if !self.construction.surface_vapor_coef.is_internal_const
            && self
                .boundary_condition
                .mass_convection_coeff_int
                .value_in(m_per_s)
                != 0.0
        {
            self.boundary_condition.vapor_transport_coeff_int =
                self.boundary_condition.mass_convection_coeff_int
                    / (VAPOR_GAS_CONSTANT * self.boundary_condition.temp_inside_air);
        }

        let mut stats: ConvergenceStats = ConvergenceStats::default();

        let temp_tol = self.settings.temp_convergence_tolerance.value_in(kelvin);
        let rh_tol = self.settings.rh_convergence_tolerance;
        let min_alpha = self.settings.min_relaxation_factor;
        // Assuming MAX_ALPHA is already defined in scope as requested. If not locally defined,
        // it references the outer scope const.

        // For early termination: track best residual to detect stalls/oscillation
        let mut best_max_delta = f64::MAX;
        let mut stall_count = 0_i32;
        const STALL_PATIENCE: i32 = 50; // bail out if no improvement for this many iterations

        // Option-(2) adaptive min-iteration guard. BlockJacobi with neighbour lagging
        // propagates a BC perturbation exactly one material cell per iteration,
        // so the per-iter δ-tolerance test can fire long before the disturbance
        // has reached the back of the wall — leaving the implicit-Euler equations
        // under-converged at deep cells.
        //
        // Three exit gates work together:
        //   * Soft floor `n_mat`     — at least a full forward sweep so the wave
        //                              reaches the back of the wall once.
        //   * Stability check (`consecutive_met >= n_mat`) — criterion has held
        //     uninterrupted for one full additional sweep, indicating the back
        //     wave has propagated forward without breaking the residual.
        //   * Hard cap `2·n_mat`     — guaranteed exit even if stability never
        //     accumulates (e.g. slow oscillation), preserving the previous
        //     forward-plus-backward worst-case behaviour.
        let n_mat: i32 = self
            .surface_curr
            .cells
            .iter()
            .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            .count() as i32;

        // Counter of consecutive iterations where the per-iter δ criterion held.
        let mut consecutive_met: i32 = 0;

        // Track when the per-iter δ criterion FIRST fires.
        let mut premature_iter: Option<i32> = None;
        let mut premature_max_dt_k: f64 = 0.0;
        let mut premature_max_drh: f64 = 0.0;

        // Anderson acceleration on the outer fixed-point loop. Reset per timestep.
        let mut accelerator = AndersonAccelerator::new(self.settings.anderson_history_depth);
        accelerator.reset();

        let mut x_prev_iter: Option<Vec<(f64, f64)>> = None;
        let mut material_indices: Option<Vec<usize>> = None;

        // ---------------------------------------------------------------------
        // Aitken Δ² Dynamic Relaxation Setup
        // Allocate iteration history buffers outside the loop to track increments.
        // ---------------------------------------------------------------------
        let n_cells = self.surface_curr.cells.len();
        let mut prev_delta_t = vec![0.0f64; n_cells];
        let mut prev_delta_rh = vec![0.0f64; n_cells];

        for iter in 0..self.settings.max_iterations {
            self.surface_next.calculate_iteration(
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            // -----------------------------------------------------------------
            // Aitken Δ² dynamic relaxation calculation (Component-wise)
            // Replaces the heuristic 10x tolerance checks.
            // -----------------------------------------------------------------
            for (i, (cell_next, cell_curr)) in self
                .surface_next
                .cells
                .iter_mut()
                .zip(self.surface_curr.cells.iter())
                .enumerate()
            {
                if !matches!(cell_next.cell_type, CellType::Material(_)) {
                    continue;
                }

                // Extract scalar increments for the current iteration
                let delta_t_curr = (cell_next.cell_temp - cell_curr.cell_temp).value_in(kelvin);
                let delta_rh_curr = (cell_next.cell_rh - cell_curr.cell_rh).value();

                let mut next_alpha = cell_curr.relaxation_factor;

                if iter > 0 {
                    let delta_t_prev = prev_delta_t[i];
                    let delta_rh_prev = prev_delta_rh[i];

                    let mut alpha_t = next_alpha;
                    let mut alpha_rh = next_alpha;

                    // Compute Aitken decay rate for Temperature
                    if delta_t_prev.abs() > 1e-12 {
                        let r_t = delta_t_curr / delta_t_prev;
                        if (1.0 - r_t) > 1e-3 {
                            alpha_t = next_alpha / (1.0 - r_t);
                        } else {
                            // r_t >= 1.0 (diverging) or r_t ~ 1.0 (stagnating): emergency brake
                            alpha_t = 0.0;
                        }
                    }

                    // Compute Aitken decay rate for Relative Humidity
                    if delta_rh_prev.abs() > 1e-12 {
                        let r_rh = delta_rh_curr / delta_rh_prev;
                        if (1.0 - r_rh) > 1e-3 {
                            alpha_rh = next_alpha / (1.0 - r_rh);
                        } else {
                            alpha_rh = 0.0;
                        }
                    }

                    // Catch overshoots and fallback to the minimum allowed relaxation factor
                    if alpha_t <= 0.0 {
                        alpha_t = min_alpha;
                    }
                    if alpha_rh <= 0.0 {
                        alpha_rh = min_alpha;
                    }

                    // Conservative approach: take the minimum alpha between T and RH
                    // to keep the coupled physical system stable.
                    next_alpha = alpha_t.min(alpha_rh);
                }

                // Apply mild low-pass filtering to the relaxation factor itself
                // to prevent erratic jumps across highly nonlinear boundaries.
                let smoothed_alpha = 0.5 * cell_curr.relaxation_factor + 0.5 * next_alpha;

                // Write back clamped alpha using the globally defined MAX_ALPHA
                cell_next.relaxation_factor = smoothed_alpha.clamp(min_alpha, MAX_ALPHA);

                // Update history buffers
                prev_delta_t[i] = delta_t_curr;
                prev_delta_rh[i] = delta_rh_curr;
            }

            mem::swap(&mut self.surface_curr, &mut self.surface_next);

            // Anderson acceleration. After the swap, surface_curr holds
            // G(x_prev) — the relaxed iterate produced by `calculate_iteration`
            // plus the per-cell α write-back.
            if self.settings.anderson_enabled {
                if material_indices.is_none() {
                    let mut idx = Vec::with_capacity(self.surface_curr.cells.len());
                    for (linear_i, cell) in self.surface_curr.cells.iter().enumerate() {
                        if matches!(cell.cell_type, CellType::Material(_)) {
                            idx.push(linear_i);
                        }
                    }
                    material_indices = Some(idx);
                }
                let mat_idx = material_indices.as_ref().unwrap();

                let mut g_x: Vec<(f64, f64)> = Vec::with_capacity(mat_idx.len());
                {
                    let cells_slice = self
                        .surface_curr
                        .cells
                        .as_slice()
                        .expect("cells contiguous");
                    for &li in mat_idx.iter() {
                        let c = &cells_slice[li];
                        g_x.push((c.cell_temp.value_in(kelvin), c.cell_rh.value()));
                    }
                }

                if let Some(x_curr_pairs) = x_prev_iter.as_ref() {
                    let x_curr_flat: Vec<f64> =
                        x_curr_pairs.iter().flat_map(|&(t, rh)| [t, rh]).collect();
                    let g_x_flat: Vec<f64> = g_x.iter().flat_map(|&(t, rh)| [t, rh]).collect();

                    if let Some(x_aa) = accelerator.accelerate(&x_curr_flat, &g_x_flat) {
                        let x_aa_pairs: Vec<(f64, f64)> =
                            x_aa.chunks_exact(2).map(|c| (c[0], c[1])).collect();

                        {
                            let cells_slice = self
                                .surface_curr
                                .cells
                                .as_slice_mut()
                                .expect("cells contiguous");
                            for (&li, &(t_new, rh_new_raw)) in mat_idx.iter().zip(x_aa_pairs.iter())
                            {
                                let rh_new = if rh_new_raw.is_finite() {
                                    rh_new_raw.clamp(0.0, 1.01)
                                } else {
                                    cells_slice[li].cell_rh.value()
                                };
                                let cell = &mut cells_slice[li];
                                cell.cell_temp = t_new * kelvin;
                                cell.cell_rh = rh_new * uom::units::dimensionless;
                                cell.calc_dependent_state(&self.boundary_condition, &self.settings);
                            }
                        }

                        let cells_slice = self
                            .surface_curr
                            .cells
                            .as_slice()
                            .expect("cells contiguous");
                        let mut x_aa_actual: Vec<(f64, f64)> = Vec::with_capacity(mat_idx.len());
                        for &li in mat_idx.iter() {
                            let c = &cells_slice[li];
                            x_aa_actual.push((c.cell_temp.value_in(kelvin), c.cell_rh.value()));
                        }
                        x_prev_iter = Some(x_aa_actual);
                    } else {
                        x_prev_iter = Some(g_x);
                    }
                } else {
                    x_prev_iter = Some(g_x);
                }
            }

            let criterion_met = stats.converged && stats.rh_converged;

            if criterion_met && premature_iter.is_none() {
                premature_iter = Some(iter);
                premature_max_dt_k = stats.max_delta_t_k;
                premature_max_drh = stats.max_delta_rh;
            }

            if criterion_met {
                consecutive_met += 1;
            } else {
                consecutive_met = 0;
            }

            let soft_floor_met = iter + 1 >= n_mat;
            let stable_exit = consecutive_met >= n_mat;
            let hard_cap_hit = iter + 1 >= 2 * n_mat;

            if criterion_met && soft_floor_met && (stable_exit || hard_cap_hit) {
                let extension = premature_iter.map_or(0, |pi| iter - pi);
                if iter > 50 || extension > 0 {
                    debug!(
                        iter,
                        max_dt_k = stats.max_delta_t_k,
                        max_drh = stats.max_delta_rh,
                        pinned_t_cell = ?stats.max_temp_cell,
                        pinned_rh_cell = ?stats.max_rh_cell,
                        n_cells = self.surface_curr.cells.len(),
                        n_mat,
                        consecutive_met,
                        premature_iter = ?premature_iter,
                        premature_max_dt_k,
                        premature_max_drh,
                        min_iter_extension = extension,
                        dt_hours = self.settings.timestep.value_in(hours),
                        exit_via = if stable_exit { "stable" } else { "hard_cap" },
                        exit_reason = "converged",
                        "solve_timestep exit"
                    );
                }
                return Ok(true);
            }

            if !criterion_met {
                let current_max = stats.max_delta_t_k.max(stats.max_delta_rh * 1000.0);
                if current_max < best_max_delta * 0.99 {
                    best_max_delta = current_max;
                    stall_count = 0;
                } else {
                    stall_count += 1;
                    if stall_count >= STALL_PATIENCE {
                        debug!(
                            iter,
                            max_dt_k = stats.max_delta_t_k,
                            max_drh = stats.max_delta_rh,
                            pinned_t_cell = ?stats.max_temp_cell,
                            pinned_rh_cell = ?stats.max_rh_cell,
                            n_cells = self.surface_curr.cells.len(),
                            n_mat,
                            consecutive_met,
                            premature_iter = ?premature_iter,
                            premature_max_dt_k,
                            premature_max_drh,
                            dt_hours = self.settings.timestep.value_in(hours),
                            exit_reason = "stalled",
                            "solve_timestep exit"
                        );
                        return Ok(false);
                    }
                }
            }
        }

        debug!(
            iter = self.settings.max_iterations,
            max_dt_k = stats.max_delta_t_k,
            max_drh = stats.max_delta_rh,
            pinned_t_cell = ?stats.max_temp_cell,
            pinned_rh_cell = ?stats.max_rh_cell,
            n_cells = self.surface_curr.cells.len(),
            n_mat,
            consecutive_met,
            premature_iter = ?premature_iter,
            premature_max_dt_k,
            premature_max_drh,
            dt_hours = self.settings.timestep.value_in(hours),
            exit_reason = "max_iter",
            "solve_timestep exit"
        );
        Ok(false)
    }

    /// Linear time-extrapolation warm start for the macro-step initial guess.
    ///
    /// Replaces material-cell `cell_temp` and `cell_rh` in `surface_curr` and
    /// `surface_next` with `value_prev + (value_prev - value_prev_prev) * Δt_ratio`,
    /// where `Δt_ratio = dt_macro_now / dt_prev_macro`. Boundary cells are left
    /// untouched — `apply_boundary_coefficients` overwrites their state at the
    /// start of every iteration. After the extrapolation, dependent state
    /// (vp_sat, d_psat_d_t, volumetric_water, ...) is refreshed via
    /// `update_cell_properties` so it stays consistent with the new (T, φ).
    fn apply_warm_start(
        &mut self,
        prev_prev: &Surface,
        dt_macro_now: Time<f64>,
        dt_prev_macro: Time<f64>,
    ) -> Result<()> {
        use crate::uom::Quantity;
        use crate::uom::dimensions::Dimensionless;
        let ratio: Dimensionless<f64> = dt_macro_now / dt_prev_macro;

        // surface_prev is the t-1 anchor; prev_prev is t-2. surface_curr was
        // just sync'd to surface_prev by tick_timestep, so it carries t-1 too.
        for ((cell_curr, cell_next), (cell_prev, cell_prev_prev)) in self
            .surface_curr
            .cells
            .iter_mut()
            .zip(self.surface_next.cells.iter_mut())
            .zip(self.surface_prev.cells.iter().zip(prev_prev.cells.iter()))
        {
            if !matches!(cell_prev.cell_type, CellType::Material(_)) {
                continue;
            }
            let dt_extrap = (cell_prev.cell_temp - cell_prev_prev.cell_temp) * ratio;
            let drh_extrap = (cell_prev.cell_rh - cell_prev_prev.cell_rh) * ratio;
            let t_init = cell_prev.cell_temp + dt_extrap;
            let rh_init = (cell_prev.cell_rh + drh_extrap).clamp(Quantity(0.0), Quantity(1.01));
            cell_curr.cell_temp = t_init;
            cell_curr.cell_rh = rh_init;
            cell_next.cell_temp = t_init;
            cell_next.cell_rh = rh_init;
        }

        self.surface_curr
            .update_cell_properties(&self.boundary_condition, &self.settings)?;
        self.surface_next
            .update_cell_properties(&self.boundary_condition, &self.settings)?;
        Ok(())
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

    /// Solves for the current timestep using an adaptive stepping strategy based on a PI controller.
/// Solves for the current timestep using an adaptive stepping strategy based on a PI controller.
    pub fn solve_timestep_adaptive<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        mut callback_state_change: Option<Box<dyn FnMut(&AdaptiveState<T>) -> Result<()>>>,
    ) -> Result<()> {
        // --- 1. Macro-level Warm Start ---
        // Uses previous macro-steps to provide a better initial guess for the non-linear solver.
        if let (Some(prev_prev), Some(dt_prev)) =
            (self.surface_prev_prev_macro.clone(), self.dt_prev_macro)
        {
            self.apply_warm_start(&prev_prev, self.settings.timestep, dt_prev)?;
        }

        if let Some(ref mut prev_surface_state) = adaptive_state.prev_surface_state {
            prev_surface_state.sync_state_from(&self.surface_prev)?;
        } else {
            adaptive_state.prev_surface_state = Some(self.surface_prev.clone());
        }

        // --- 2. State & Accumulator Initialization ---
        adaptive_state.flux_accumulator.reset();
        let macro_dt = self.settings.timestep;
        adaptive_state.substep_time = 0.0 * hours;

        // Optimistic Start with the full macro Δt. The PI controller will adjust down if it detects instability.
        let mut current_dt: Time<f64> = adaptive_state.proposed_dt.unwrap_or(macro_dt);

        // Historical change rates for PI linear extrapolation. Cold start with 0.
        if adaptive_state.rate_t.is_none() {
            adaptive_state.rate_t = Some(ndarray::Array3::<Quantity<f64, _>>::from_shape_fn(
                self.surface_curr.cells.dim(),
                |_| 0.0 * kelvin / hours,
            ));
        }
        if adaptive_state.rate_rh.is_none() {
            adaptive_state.rate_rh = Some(ndarray::Array3::<Quantity<f64, _>>::from_shape_fn(
                self.surface_curr.cells.dim(),
                |_| 0.0 * (dimensionless / hours),
            ));
        }

        if adaptive_state.predicted_surface_state.is_none() {
            adaptive_state.predicted_surface_state = Some(self.surface_prev.clone());
        }

        // Use adaptive_state.substep_time as the accumulator condition
        while adaptive_state.substep_time < macro_dt {
            let time_remaining = macro_dt - adaptive_state.substep_time;
            let is_boundary_step = current_dt >= time_remaining;
            let rate_t = adaptive_state.rate_t.as_mut().unwrap();
            let rate_rh = adaptive_state.rate_rh.as_mut().unwrap();
            
            // Boundary Clamping
            let dt_step: Time<f64> = current_dt.min(time_remaining);

            let predicted_surface = adaptive_state
                .predicted_surface_state
                .as_mut()
                .expect("Predicted Surface shouldn't be None");
                
            predicted_surface.sync_state_from(&self.surface_prev)?;
            for (idx, cell) in predicted_surface.cells.indexed_iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp += (*rate_t)[idx] * dt_step;
                    cell.cell_rh += (*rate_rh)[idx] * dt_step;
                    cell.calc_dependent_state(&self.boundary_condition, &self.settings);
                }
            }


            // Predictor Warm-Start
            // Use the predicted surface as the initial guess for the current step. 
            //This can significantly reduce iterations for small steps, especially in the early phase of a macro-step where the state changes rapidly.
            self.surface_curr.sync_state_from(predicted_surface)?;

            self.settings.timestep = dt_step;
            let converged = self.solve_timestep()?;

            if !converged {
                // Failed to converge, rollback to the previous state and reduce the timestep more aggressively for the next try.
                current_dt *= 0.25;

                if current_dt.value_in(seconds) < 1e-3 {
                    return Err(error::Error::Other(format!(
                        "Solver irrecoverably stalled at {:.2} hours with dt < 1ms",
                        adaptive_state.current_time.value_in(hours)
                    ))
                    .into());
                }

                // Clean up any partial updates to avoid contaminating the next iteration
                self.surface_curr.sync_state_from(&self.surface_prev)?;
                self.surface_next.sync_state_from(&self.surface_prev)?;
                continue;
            }


            // Decouple Integration Tolerance
            // Instead of using the global convergence tolerances, we define separate integration tolerances for temperature and relative humidity that are more lenient. 
            // This allows the PI controller to make informed decisions based on the physical changes rather than strict solver convergence, which can be too tight for adaptive stepping.
            let integ_tol_t: Temperature<f64> = self.settings.temp_convergence_tolerance.max(0.25 * kelvin); 
            let integ_tol_rh: Dimensionless<f64> = self.settings.rh_convergence_tolerance.max(0.02 * dimensionless);

            let mut max_lte: f64 = 1e-12;
            for (cell_pred, cell_curr) in predicted_surface
                .cells
                .iter()
                .zip(self.surface_curr.cells.iter())
            {
                if matches!(cell_curr.cell_type, CellType::Material(_)) {
                    let lte_t:Dimensionless<f64> = (cell_pred.cell_temp - cell_curr.cell_temp).abs() / integ_tol_t;
                    let lte_rh:Dimensionless<f64> = (cell_pred.cell_rh - cell_curr.cell_rh).abs() / integ_tol_rh;

                    let cell_lte = lte_t.max(lte_rh).value();
                    if cell_lte > max_lte {
                        max_lte = cell_lte;
                    }
                }
            }

            // Accept/Reject Logic
            if max_lte <= 1.0 {
                // ACCEPT

                // Update the prediction rates.
                for (idx, cell_prev) in self.surface_prev.cells.indexed_iter() {
                    if matches!(cell_prev.cell_type, CellType::Material(_)) {
                        (*rate_t)[idx] = (self.surface_curr.cells[idx].cell_temp - cell_prev.cell_temp) / dt_step;
                        (*rate_rh)[idx] = (self.surface_curr.cells[idx].cell_rh - cell_prev.cell_rh) / dt_step;
                    }
                }

                adaptive_state.substep_time += dt_step;
                adaptive_state.current_time += dt_step;

                // Accumulate fluxes.
                {
                    let q_int = self.surface_curr.heat_flux(BoundaryLocation::Interior)?;
                    let q_ext = self.surface_curr.heat_flux(BoundaryLocation::Exterior)?;
                    let m_int = self
                        .surface_curr
                        .moisture_flux(BoundaryLocation::Interior, &self.boundary_condition)?;
                    let m_ext = self
                        .surface_curr
                        .moisture_flux(BoundaryLocation::Exterior, &self.boundary_condition)?;

                    adaptive_state.flux_accumulator.q_int_dt += q_int * dt_step;
                    adaptive_state.flux_accumulator.q_ext_dt += q_ext * dt_step;
                    adaptive_state.flux_accumulator.m_int_dt += m_int * dt_step;
                    adaptive_state.flux_accumulator.m_ext_dt += m_ext * dt_step;
                    adaptive_state.flux_accumulator.total_time += dt_step;
                }

                std::mem::swap(&mut self.surface_prev, &mut self.surface_curr);
                self.surface_curr.sync_state_from(&self.surface_prev)?;

                if let Some(ref mut cb) = callback_state_change {
                    cb(adaptive_state)?;
                }

                // PI Controller
                if !is_boundary_step {
                    let k_i = 0.15; 
                    let k_p = 0.05; 
                    let safety = 0.8; 

                    let mut amplification = (1.0 / max_lte).powf(k_i);
                    if let Some(pe) = adaptive_state.prev_lte {
                        amplification *= (pe / max_lte).powf(k_p);
                    }
                    
                    // Asymmetric Clamping
                    // Allow more aggressive amplification on the first successful step after a failure to quickly recover, 
                    // but clamp more conservatively after that to ensure stability.
                    let max_amp = if adaptive_state.prev_lte.is_none() { 2.0 } else { 1.5 };
                    amplification = (amplification * safety).clamp(0.25, max_amp);

                    current_dt = dt_step * amplification;
                    adaptive_state.prev_lte = Some(max_lte);
                }
            } else {
                // REJECT 
                
                // In cold start, be more forgiving to allow the controller to gather data. After that, be more conservative to ensure stability.
                let min_shrink = if adaptive_state.prev_lte.is_none() { 0.5 } else { 0.1 };
                let shrink_factor = (0.8 / max_lte.sqrt()).clamp(min_shrink, 0.9);

                current_dt = dt_step * shrink_factor;

                // Clean up any partial updates to avoid contaminating the next iteration
                self.surface_curr.sync_state_from(&self.surface_prev)?;
                self.surface_next.sync_state_from(&self.surface_prev)?;
            }
        }

        // Save the successfully proposed macro Δt for the next step's warm start extrapolation, 
        // but only if we made actual progress (substep_time > 0) to avoid poisoning the next step with a failed attempt.
        adaptive_state.proposed_dt = Some(current_dt);

        mem::swap(&mut self.surface_prev, &mut self.surface_curr);
        self.surface_prev
            .sync_state_from(adaptive_state.prev_surface_state.as_ref().unwrap())?;

        self.settings.timestep = macro_dt;
        self.dt_prev_macro = Some(macro_dt);
        adaptive_state.substep_time = macro_dt;

        Ok(())
    }


    /// Executes after the timestep has been calculated.
    pub fn tick_timestep(&mut self) -> Result<()> {
        // Capture the about-to-be-replaced t-1 state as the warm-start anchor
        // for the *next* macro call (where it becomes t-2). Done here, before
        // the swap, because afterwards `surface_prev` carries the just-converged
        // state and the t-1 snapshot is no longer accessible.
        match self.surface_prev_prev_macro {
            Some(ref mut s) => {
                s.sync_state_from(&self.surface_prev)?;
            }
            None => {
                self.surface_prev_prev_macro = Some(self.surface_prev.clone());
            }
        }

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

/// Selects the inner linear solver used by `solve_timestep`.
///
/// The legacy `BlockJacobi` path runs block-Jacobi sweeps until per-iter δ
/// falls below tolerance (with the option-2 min-iter guard for
/// correctness). The `BlockTridiagonal` path assembles the implicit-Euler
/// step into a block-tridiagonal system and solves it directly via the
/// Thomas algorithm — O(N) per outer iteration and free of the
/// premature-convergence pathology (see `solver::block_tridiag`).
///
/// `BlockTridiagonal` requires `coupled_heat_moisture_enabled = true`.
/// If false, `solve_timestep` falls back to `BlockJacobi` and emits a warn!.
///
/// Future 2-D / 3-D wall extensions will add an `Adi` variant that
/// reuses the same block-Thomas kernel direction-by-direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    BlockJacobi,
    BlockTridiagonal,
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
    pub rh_convergence_tolerance: Dimensionless<f64>,
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

    /// Enable symmetric 2×2 heat–moisture block solve.
    ///
    /// When `true`, the energy equation is linearized to include the T↔φ cross-coupling
    /// term through P_sat, giving symmetric first-order coupling in both directions.
    /// Set to `false` to reproduce the legacy staggered (lagged) scheme exactly.
    pub coupled_heat_moisture_enabled: bool,

    /// Enable differential sorption-enthalpy contribution to latent heat.
    ///
    /// When `true` and a material provides a `sorption_enthalpy` curve, the effective
    /// latent heat used in the energy balance is `L_v + H_diff(φ)` instead of `L_v` alone.
    /// Has no effect if no material in the construction provides the curve.
    pub sorption_enthalpy_enabled: bool,

    /// Enable Type-II Anderson acceleration on the outer fixed-point loop.
    ///
    /// When `true`, the solver wraps the per-iteration BlockJacobi map with an AA
    /// extrapolation step using the last `anderson_history_depth` iterates.
    /// When `false`, the loop is byte-identical to the unaccelerated baseline.
    pub anderson_enabled: bool,

    /// Maximum AA history depth (mixing parameter `m`). Typical values 3..=5.
    pub anderson_history_depth: usize,

    /// Inner linear-solver method. See `SolverMethod`.
    pub solver_method: SolverMethod,
}

impl Default for HamtSettings {
    fn default() -> Self {
        Self {
            timestep: 3600.0 * units::seconds,
            max_iterations: 1500,
            temp_convergence_tolerance: 0.001 * units::kelvin,
            rh_convergence_tolerance: 0.0001 * units::dimensionless,
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
            min_relaxation_factor: 0.1,
            coupled_heat_moisture_enabled: true,
            sorption_enthalpy_enabled: true,
            anderson_enabled: true,
            anderson_history_depth: 3,
            solver_method: SolverMethod::BlockTridiagonal,
        }
    }
}

/// Accumulates time-weighted boundary fluxes across adaptive substeps.
/// After all substeps, dividing each `*_dt` field by `total_time` yields the
/// time-averaged flux density over the macro-timestep.
#[derive(Debug, Clone)]
pub struct SubstepFluxAccumulator {
    /// ∫ q_int dt  [J/m²]
    pub q_int_dt: ArealEnergyDensity<f64>,
    /// ∫ q_ext dt  [J/m²]
    pub q_ext_dt: ArealEnergyDensity<f64>,
    /// ∫ m_int dt  [kg/m²]
    pub m_int_dt: ArealMassDensity<f64>,
    /// ∫ m_ext dt  [kg/m²]
    pub m_ext_dt: ArealMassDensity<f64>,
    /// Total substep time accumulated
    pub total_time: Time<f64>,
}

impl Default for SubstepFluxAccumulator {
    fn default() -> Self {
        Self {
            q_int_dt: 0.0 * j_per_m2,
            q_ext_dt: 0.0 * j_per_m2,
            m_int_dt: 0.0 * kg_per_m2,
            m_ext_dt: 0.0 * kg_per_m2,
            total_time: 0.0 * seconds,
        }
    }
}

impl SubstepFluxAccumulator {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Time-averaged interior heat flux [W/m²].
    pub fn avg_q_int(&self) -> EnergyFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.q_int_dt / self.total_time
        } else {
            0.0 * w_per_m2
        }
    }

    /// Time-averaged exterior heat flux [W/m²].
    pub fn avg_q_ext(&self) -> EnergyFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.q_ext_dt / self.total_time
        } else {
            0.0 * w_per_m2
        }
    }

    /// Time-averaged interior moisture flux [kg/(m²·s)].
    pub fn avg_m_int(&self) -> MassFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.m_int_dt / self.total_time
        } else {
            0.0 * kg_per_m2_s
        }
    }

    /// Time-averaged exterior moisture flux [kg/(m²·s)].
    pub fn avg_m_ext(&self) -> MassFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.m_ext_dt / self.total_time
        } else {
            0.0 * kg_per_m2_s
        }
    }
}

pub struct AdaptiveState<T> {
    pub substep_time: Time<f64>,
    pub current_time: Time<f64>,
    pub split_factor: i32,
    pub max_split_factor: i32,

    /// The current timestep size proposed by the PI controller.
    pub proposed_dt: Option<Time<f64>>,

    /// The local truncation error (LTE) from the previous accepted step, used for the Integral term.
    pub prev_lte: Option<f64>,

    // Temperature per Time
    pub rate_t: Option<ndarray::Array3<TempRate<f64>>>,
    pub rate_rh: Option<ndarray::Array3<RhRate<f64>>>,
    pub data: T,
    pub retry_wait_time: i32,
    pub time_in_contineous_split: i32,
    pub reset_time_in_contineous: bool,
    pub prev_surface_state: Option<Surface>,
    pub predicted_surface_state: Option<Surface>,
    pub flux_accumulator: SubstepFluxAccumulator,
}

impl<T> Default for AdaptiveState<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            substep_time: 0.0 * units::seconds,
            current_time: 0.0 * units::seconds,
            split_factor: 1,
            max_split_factor: 12,
            data: T::default(),
            proposed_dt: None,
            prev_lte: None,
            rate_t: None,
            rate_rh: None,
            retry_wait_time: 0,
            time_in_contineous_split: 0,
            reset_time_in_contineous: false,
            prev_surface_state: None,
            predicted_surface_state: None,
            flux_accumulator: SubstepFluxAccumulator::default(),
        }
    }
}

#[cfg(test)]
mod block_tridiag_dispatch_tests {
    use super::*;
    use crate::cell::CellType;
    use crate::uom::units::kelvin;

    /// Property: with `solver_method = BlockTridiagonal` and a benign
    /// macro-step (BC matches initial wall state), `solve_timestep`
    /// returns `Ok(true)` and produces a small-residual converged state.
    /// Catches dispatch bugs (BlockJacobi path getting hit instead) and
    /// outer-loop loop-condition bugs.
    #[test]
    fn block_tridiag_converges_on_benign_step() {
        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        mgr.settings.solver_method = SolverMethod::BlockTridiagonal;
        // Drive both states to BCs so the step is a no-op (benign).
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp = bc_temp;
                    cell.cell_rh = bc_rh;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }

        let result = mgr
            .solve_timestep()
            .expect("solve must not error on benign step");
        assert!(result, "solve must return Ok(true) on benign step");

        // Verify the converged state is at the BC.
        for cell in mgr.surface_curr.cells.iter() {
            if !matches!(cell.cell_type, CellType::Material(_)) {
                continue;
            }
            let dt_k = (cell.cell_temp - bc_temp).value_in(kelvin).abs();
            let drh = (cell.cell_rh - bc_rh).value().abs();
            assert!(
                dt_k < 1e-3 && drh < 1e-3,
                "post-step ΔT={dt_k}, Δφ={drh} — block-tridiag did not stay at BC"
            );
        }
    }

    /// Property: with `coupled_heat_moisture_enabled = false` AND
    /// `solver_method = BlockTridiagonal`, the dispatch falls back to
    /// BlockJacobi (via the warn! path). The solve completes without error.
    #[test]
    fn block_tridiag_falls_back_when_coupling_disabled() {
        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        mgr.settings.solver_method = SolverMethod::BlockTridiagonal;
        mgr.settings.coupled_heat_moisture_enabled = false;
        // Drive both states to BCs so the step is benign.
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp = bc_temp;
                    cell.cell_rh = bc_rh;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }
        // Should not panic; the dispatcher must fall back to BlockJacobi
        // because BlockTridiagonal requires coupled_heat_moisture_enabled.
        let _ = mgr
            .solve_timestep()
            .expect("fallback to BlockJacobi must not error");
    }
}
