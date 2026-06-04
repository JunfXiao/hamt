//! Outer-loop driver and per-solver dispatch functions.

use crate::accel::{Accelerator, Aitken, RelaxKnobs};
use crate::accel::anderson::AndersonAccelerator;
use crate::cell::CellType;
use crate::surface::Surface;
use crate::uom::constants::VAPOR_GAS_CONSTANT;
use crate::uom::units::{hours, kelvin, m_per_s};
use crate::uom;
use anyhow::Result;
use std::mem;
use tracing::debug;
use tracing::warn;

use super::settings::{HamtSettings, MAX_ALPHA, SolverMethod};
use super::HamtManager;

/// Option-(2) adaptive min-iteration guard for the BlockJacobi outer loop.
/// BlockJacobi with neighbour lagging propagates a BC perturbation exactly one
/// material cell per iteration, so the per-iter δ-tolerance test can fire long
/// before the disturbance has reached the back of the wall. The guard combines
/// a soft floor (one full forward sweep = `n_mat`), a stability check
/// (`consecutive_met >= n_mat`), and a hard cap (`hard_cap`).
pub(super) struct MinIterGuard {
    pub n_mat: i32,
    pub hard_cap: i32,
}

/// Exit policy for [`HamtManager::run_outer_loop`]. Selects which exit gates the
/// shared driver applies. BlockTridiag leaves both optional fields `None` and
/// uses the simple `converged && rh_converged` check; BlockJacobi supplies both.
pub(super) struct ExitPolicy {
    pub max_iter: i32,
    pub min_iter_guard: Option<MinIterGuard>,
    pub stall_patience: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ConvergenceStats {
    pub converged: bool,
    pub rh_converged: bool,
    pub max_delta_t_k: f64,
    pub max_delta_rh: f64,
    pub max_temp_cell: Option<[usize; 3]>,
    pub max_rh_cell: Option<[usize; 3]>,
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

pub(super) fn convergence_stats(
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

impl HamtManager {
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

    /// Shared outer-iteration driver for both inner solvers.
    ///
    /// Hosts the skeleton common to BlockJacobi and BlockTridiag: per iteration
    /// `step -> convergence_stats -> aitken.relax -> swap -> (Anderson) -> exit
    /// test`. The differences are injected via `step` (which inner step to run),
    /// `anderson` (Some only for BlockJacobi this task), and `exit` (which exit
    /// gates apply). Behaviour is bit-identical to the two hand-written loops it
    /// replaces.
    pub(super) fn run_outer_loop<S: crate::surface::IterativeStep>(
        &mut self,
        step: &S,
        aitken: &mut Aitken,
        mut anderson: Option<&mut AndersonAccelerator>,
        exit: ExitPolicy,
    ) -> Result<bool> {
        let mut stats: ConvergenceStats = ConvergenceStats::default();

        let min_alpha = self.settings.min_relaxation_factor;

        // --- BlockJacobi-only exit state (unused when the policy leaves the
        //     corresponding fields None). ---

        // Stall detection.
        let mut best_max_delta = f64::MAX;
        let mut stall_count = 0_i32;

        // Min-iter guard derived quantities.
        let n_mat: i32 = exit.min_iter_guard.as_ref().map_or(0, |g| g.n_mat);
        let hard_cap: i32 = exit.min_iter_guard.as_ref().map_or(0, |g| g.hard_cap);

        // Counter of consecutive iterations where the per-iter delta criterion held.
        let mut consecutive_met: i32 = 0;

        // Track when the per-iter delta criterion FIRST fires.
        let mut premature_iter: Option<i32> = None;
        let mut premature_max_dt_k: f64 = 0.0;
        let mut premature_max_drh: f64 = 0.0;

        // Anderson per-iteration caches.
        let mut x_prev_iter: Option<Vec<(f64, f64)>> = None;
        let mut material_indices: Option<Vec<usize>> = None;

        for iter in 0..exit.max_iter {
            // 1. surface_prev (physical time t-1) supplies the inertial term (tcap/dt).
            // 2. surface_curr (iteration step k) supplies the nonlinear coefficients / Jacobian.
            // 3. results are written to surface_next (iteration step k+1).
            step.step(
                &mut self.surface_next,
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            )?;

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            // Aitken delta-squared dynamic relaxation (Component-wise)
            aitken.relax(
                &mut self.surface_next,
                &self.surface_curr,
                iter,
                RelaxKnobs {
                    min_alpha,
                    max_alpha: MAX_ALPHA,
                },
            );

            // Iteration advancement: overwrite the k state with the k+1 state for the next loop
            mem::swap(&mut self.surface_curr, &mut self.surface_next);

            // Anderson acceleration. After the swap, surface_curr holds
            // G(x_prev) -- the relaxed iterate produced by the inner step
            // plus the per-cell alpha write-back. Only wired for BlockJacobi.
            if let Some(accelerator) = anderson.as_deref_mut() {
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

            // --- Exit logic ---
            if exit.min_iter_guard.is_none() && exit.stall_patience.is_none() {
                // BlockTridiag: simple converged check.
                if criterion_met {
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
            } else {
                // BlockJacobi: option-2 min-iter guard + stall detection.
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
                let hard_cap_hit = iter + 1 >= hard_cap;

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

                if !criterion_met
                    && let Some(stall_patience) = exit.stall_patience
                {
                    let current_max = stats.max_delta_t_k.max(stats.max_delta_rh * 1000.0);
                    if current_max < best_max_delta * 0.99 {
                        best_max_delta = current_max;
                        stall_count = 0;
                    } else {
                        stall_count += 1;
                        if stall_count >= stall_patience {
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
        }

        // --- Fall-through (max iterations reached) ---
        if exit.min_iter_guard.is_none() && exit.stall_patience.is_none() {
            debug!(
                iter = exit.max_iter,
                max_dt_k = stats.max_delta_t_k,
                max_drh = stats.max_delta_rh,
                pinned_t_cell = ?stats.max_temp_cell,
                pinned_rh_cell = ?stats.max_rh_cell,
                n_cells = self.surface_curr.cells.len(),
                dt_hours = self.settings.timestep.value_in(hours),
                exit_reason = "max_outer",
                "solve_timestep_block_tridiag did NOT converge"
            );
        } else {
            debug!(
                iter = exit.max_iter,
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
        }
        Ok(false)
    }

    /// Direct block-tridiagonal solve. See `solver::block_tridiag` for the
    /// kernel and `surface::update::solve_step_block_tridiag` for the 1-D
    /// wall assembly.
    pub(super) fn solve_timestep_block_tridiag(&mut self) -> Result<bool> {
        // The outer loop is much shorter than BlockJacobi: it only iterates to
        // convergence on the nonlinearity (Psat(T), dwdphi(phi), L_eff(phi)).
        // Cap at a small number; if it doesn't converge in ~20 iters the
        // inputs are pathological.
        const MAX_OUTER: i32 = 200;

        // Fully utilize the dimension definition of surface_curr.
        let n_cells = self.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);

        // Anderson acceleration on the outer nonlinear loop. Reset per timestep.
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);
        anderson.reset();

        self.run_outer_loop(
            &crate::surface::BlockTridiagStep,
            &mut aitken,
            if self.settings.anderson_enabled {
                Some(&mut anderson)
            } else {
                None
            },
            ExitPolicy {
                max_iter: MAX_OUTER,
                min_iter_guard: None,
                stall_patience: None,
            },
        )
    }

    pub(super) fn solve_timestep_block_jacobi(&mut self) -> Result<bool> {
        // Reapply surface vapor coefficients if not using constant value.
        // When mass_convection_coeff is non-zero it represents h_mass [m/s] and beta is derived
        // via beta = h_mass / (R_v*T).  When it is zero the caller has set vapor_transport_coeff
        // directly as beta [kg/(m^2*s*Pa)] and no conversion is needed.
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

        // Option-(2) adaptive min-iteration guard. BlockJacobi with neighbour lagging
        // propagates a BC perturbation exactly one material cell per iteration,
        // so the per-iter delta-tolerance test can fire long before the disturbance
        // has reached the back of the wall -- leaving the implicit-Euler equations
        // under-converged at deep cells.
        //
        // Three exit gates work together:
        //   * Soft floor `n_mat`     -- at least a full forward sweep so the wave
        //                              reaches the back of the wall once.
        //   * Stability check (`consecutive_met >= n_mat`) -- criterion has held
        //     uninterrupted for one full additional sweep, indicating the back
        //     wave has propagated forward without breaking the residual.
        //   * Hard cap `2*n_mat`     -- guaranteed exit even if stability never
        //     accumulates (e.g. slow oscillation), preserving the previous
        //     forward-plus-backward worst-case behaviour.
        let n_mat: i32 = self
            .surface_curr
            .cells
            .iter()
            .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            .count() as i32;

        // Aitken delta-squared Dynamic Relaxation Setup
        let n_cells = self.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);

        // Anderson acceleration on the outer fixed-point loop. Reset per timestep.
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);
        anderson.reset();

        self.run_outer_loop(
            &crate::surface::BlockJacobiStep,
            &mut aitken,
            if self.settings.anderson_enabled {
                Some(&mut anderson)
            } else {
                None
            },
            ExitPolicy {
                max_iter: self.settings.max_iterations,
                min_iter_guard: Some(MinIterGuard {
                    n_mat,
                    hard_cap: 2 * n_mat,
                }),
                stall_patience: Some(50),
            },
        )
    }
}
