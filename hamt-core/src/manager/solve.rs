//! Outer-loop driver and per-solver dispatch functions.

use crate::accel::anderson::AndersonAccelerator;
use crate::accel::{Accelerator, Aitken, RelaxKnobs};
use crate::cell::CellType;
use crate::surface::Surface;
use crate::uom;
use crate::uom::constants::VAPOR_GAS_CONSTANT;
use crate::uom::units::{hours, kelvin, m_per_s};
use anyhow::{Context, Result};
use std::mem;
use tracing::debug;
use tracing::warn;

use super::HamtManager;
use super::settings::{HamtSettings, MAX_ALPHA, SolverMethod};

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
        let dim = self.surface_curr.cells.raw_dim();
        if dim[1] > 1 || dim[2] > 1 {
            // BlockJacobi / BlockTridiagonal are 1-D (x-only); multi-dimensional
            // grids are solved by ADI regardless of the configured solver_method.
            return self.solve_timestep_adi();
        }
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
            SolverMethod::Adi => self.solve_timestep_adi(),
        }
    }

    /// Shared outer-iteration driver for both inner solvers.
    ///
    /// Hosts the skeleton common to BlockJacobi and BlockTridiag: per iteration
    /// `step -> convergence_stats -> relaxer.relax -> swap -> (extrapolator) ->
    /// exit test`. The differences are injected via `step` (which inner step to
    /// run), `extrapolator` (Some when Anderson is enabled), and `exit` (which
    /// exit gates apply). Both accelerators are dispatched through the
    /// [`Accelerator`] trait — `relaxer` (Aitken) does per-cell relaxation,
    /// `extrapolator` (Anderson) does the global post-swap extrapolation.
    /// Behaviour is bit-identical to the two hand-written loops it replaces.
    pub(super) fn run_outer_loop<S: crate::surface::IterativeStep>(
        &mut self,
        step: &S,
        relaxer: &mut dyn Accelerator,
        mut extrapolator: Option<&mut dyn Accelerator>,
        exit: ExitPolicy,
    ) -> Result<bool> {
        // Per-timestep reset of accelerator history. Matches the per-timestep
        // reset the solvers previously did on the AndersonAccelerator and the
        // fresh Aitken buffers (Aitken::new starts zeroed; reset keeps it so).
        relaxer.reset();
        if let Some(ex) = extrapolator.as_deref_mut() {
            ex.reset();
        }

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
            // A singular diagonal block during elimination is NOT fatal: it
            // means the current `dt` is too aggressive for the linearisation
            // to retain diagonal dominance. Shrink-and-retry is owned by the
            // adaptive substepper (`solve_timestep_adaptive`), so we surface
            // this as "did not converge" (Ok(false)) instead of aborting.
            // The typed `SolverError` survives the anyhow round-trip via
            // `downcast_ref` because `step_tridiag` / `step_adi` propagate it
            // with `anyhow::Error::from(e).context(...)` rather than
            // stringifying with `anyhow!(format!(...))`.
            let step_result: Result<()> = step.step(
                &mut self.surface_next,
                &self.settings,
                &self.boundary_condition,
                &self.surface_prev,
                &self.surface_curr,
            );
            if let Err(err) = step_result {
                if let Some(crate::solver::block_tridiag::SolverError::Singular { row }) =
                    err.downcast_ref::<crate::solver::block_tridiag::SolverError>()
                {
                    tracing::warn!(
                        row = *row,
                        dt_hours = self.settings.timestep.value_in(hours),
                        outer_iter = iter,
                        "block-tridiag singular: signalling non-convergence so the adaptive substepper can shrink dt and retry"
                    );
                    return Ok(false);
                }
                return Err(err).context("step failed");
            }

            stats = convergence_stats(&self.surface_next, &self.surface_curr, &self.settings);

            // Aitken delta-squared dynamic relaxation (Component-wise),
            // dispatched via the Accelerator trait.
            relaxer.relax(
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
            // plus the per-cell alpha write-back. Active only when an
            // extrapolator (Anderson) is supplied.
            if extrapolator.is_some() {
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

                    if let Some(x_aa) = extrapolator
                        .as_deref_mut()
                        .and_then(|ex| ex.extrapolate(&x_curr_flat, &g_x_flat))
                    {
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
                    // The iteration prologue refreshed the Virtual (surface) cells
                    // from the PREVIOUS iterate (plus Aitken/Anderson have moved the
                    // material state since). Re-derive them from the final state so
                    // the flux queries the substepper integrates (`moisture_flux`,
                    // `heat_flux`) and `report` see a surface node that is consistent
                    // with the material cells the implicit update produced.
                    self.surface_curr
                        .refresh_virtual_cells(&self.boundary_condition);
                    self.surface_curr
                        .update_cell_properties(&self.boundary_condition, &self.settings)?;
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

                if !criterion_met && let Some(stall_patience) = exit.stall_patience {
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

    /// Test-only driver: run the outer loop for an arbitrary [`IterativeStep`]
    /// with the standard ADI accelerator stack (Aitken relaxer + optional
    /// Anderson extrapolator) and report `(converged, outer_iterations)`.
    ///
    /// The iteration count is captured by wrapping the Aitken relaxer in a
    /// counting adapter: `run_outer_loop` calls `relax` exactly once at the top
    /// of every outer-iteration body (before the exit test), so the relax-call
    /// count equals the number of outer iterations executed before exit. Used by
    /// the ADI-vs-direct strong-coupling gate (Phase 2a Task 10).
    #[cfg(test)]
    pub(crate) fn run_outer_loop_counted<S: crate::surface::IterativeStep>(
        &mut self,
        step: &S,
        max_iter: i32,
    ) -> Result<(bool, i32)> {
        use crate::accel::{Accelerator, Aitken, RelaxKnobs};

        /// Counts `relax` invocations while forwarding to an inner relaxer.
        struct CountingRelaxer<'a> {
            inner: &'a mut dyn Accelerator,
            count: i32,
        }
        impl Accelerator for CountingRelaxer<'_> {
            fn relax(&mut self, next: &mut Surface, curr: &Surface, iter: i32, knobs: RelaxKnobs) {
                self.count += 1;
                self.inner.relax(next, curr, iter, knobs);
            }
            fn extrapolate(&mut self, x_curr: &[f64], g_x: &[f64]) -> Option<Vec<f64>> {
                self.inner.extrapolate(x_curr, g_x)
            }
            fn reset(&mut self) {
                self.count = 0;
                self.inner.reset();
            }
        }

        let n_cells = self.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);
        let mut counting = CountingRelaxer { inner: &mut aitken, count: 0 };
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);

        // `run_outer_loop` resets the relaxer at entry (CountingRelaxer::reset
        // zeroes the count), so `counting.count` after the loop is the number of
        // outer iterations executed this call. The inner block scopes the
        // `&mut counting` borrow so it ends before we read `counting.count`.
        let converged = {
            let extrapolator: Option<&mut dyn Accelerator> = if self.settings.anderson_enabled {
                Some(&mut anderson as &mut dyn Accelerator)
            } else {
                None
            };
            self.run_outer_loop(
                step,
                &mut counting,
                extrapolator,
                ExitPolicy { max_iter, min_iter_guard: None, stall_patience: None },
            )?
        };

        Ok((converged, counting.count))
    }

    /// ADI (Alternating-Direction Implicit) solve.
    ///
    /// For a 1-D grid (`ny == 1 && nz == 1`) this delegates to
    /// `solve_timestep_block_tridiag`: a single x-sweep with no cross coupling is
    /// byte-identical to the block-tridiag path.
    ///
    /// For a genuine 2-D/3-D grid (`ny > 1` or `nz > 1`) the alternating
    /// x→y line sweeps are driven by the shared [`HamtManager::run_outer_loop`]
    /// with `AdiStep` as the inner step. The outer loop iterates on the
    /// nonlinearity (`P_sat(T)`, `∂w/∂φ`, `L_eff(φ)`) and the cross-axis lag
    /// until convergence. `solve_timestep` always routes multi-dimensional grids
    /// here regardless of the configured `solver_method`.
    pub(super) fn solve_timestep_adi(&mut self) -> Result<bool> {
        // ADI assembles the same 2×2 (T,φ) coupled block as BlockTridiagonal;
        // the coupling term (∂P_sat/∂T) is essential for the off-diagonal entries.
        debug_assert!(
            self.settings.coupled_heat_moisture_enabled,
            "Adi requires coupled_heat_moisture_enabled=true (the 2×2 block assumes T↔φ coupling)"
        );

        let dim = self.surface_curr.cells.raw_dim();
        if dim[1] == 1 && dim[2] == 1 {
            // 1-D grid: a single x-sweep with no cross coupling is byte-identical
            // to the block-tridiag path. Delegate to guarantee 1-D equivalence.
            return self.solve_timestep_block_tridiag();
        }

        // Genuine 2-D/3-D: alternating x→y line sweeps, iterated on the
        // cross-axis lag (and the nonlinearity) by the shared outer loop.
        let n_cells = self.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);

        self.run_outer_loop(
            &crate::surface::AdiStep,
            &mut aitken,
            if self.settings.anderson_enabled {
                Some(&mut anderson as &mut dyn Accelerator)
            } else {
                None
            },
            ExitPolicy {
                max_iter: 200,
                min_iter_guard: None,
                stall_patience: None,
            },
        )
    }

    /// Direct block-tridiagonal solve. See `solver::block_tridiag` for the
    /// kernel and `surface::step_tridiag` for the 1-D wall assembly.
    pub(super) fn solve_timestep_block_tridiag(&mut self) -> Result<bool> {
        // The outer loop is much shorter than BlockJacobi: it only iterates to
        // convergence on the nonlinearity (Psat(T), dwdphi(phi), L_eff(phi)).
        // Cap at a small number; if it doesn't converge in ~20 iters the
        // inputs are pathological.
        const MAX_OUTER: i32 = 200;

        // Fully utilize the dimension definition of surface_curr.
        let n_cells = self.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);

        // Anderson acceleration on the outer nonlinear loop. History is reset
        // per timestep inside run_outer_loop via the Accelerator trait.
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);

        self.run_outer_loop(
            &crate::surface::BlockTridiagStep,
            &mut aitken,
            if self.settings.anderson_enabled {
                Some(&mut anderson as &mut dyn Accelerator)
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

        // Anderson acceleration on the outer fixed-point loop. History is reset
        // per timestep inside run_outer_loop via the Accelerator trait.
        let mut anderson = AndersonAccelerator::new(self.settings.anderson_history_depth);

        self.run_outer_loop(
            &crate::surface::BlockJacobiStep,
            &mut aitken,
            if self.settings.anderson_enabled {
                Some(&mut anderson as &mut dyn Accelerator)
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

#[cfg(test)]
mod singular_retry_tests {
    //! Regression test for the production crash
    //! `block-tridiag solve failed: Singular { row: 35 }` at time step 2.
    //!
    //! Root cause: `solve_timestep_adaptive` used `?` on the inner
    //! `solve_timestep`, so a `SolverError::Singular` from the block-Thomas
    //! kernel propagated as a fatal error instead of being treated as a
    //! "shrink dt and retry" signal. The fix routes `Singular` through
    //! `run_outer_loop` → `Ok(false)` (non-convergence), which the adaptive
    //! substepper handles by quartering `dt` and retrying.

    use super::*;
    use crate::solver::block_tridiag::SolverError;
    use crate::surface::IterativeStep;
    use std::cell::Cell as StdCell;

    /// Stub step that returns `SolverError::Singular` a configurable number
    /// of times before succeeding. Succeeding means "leave next untouched
    /// (converged state == curr state)".
    struct SingularThenOk {
        remaining_singular: StdCell<usize>,
    }

    impl IterativeStep for SingularThenOk {
        fn step(
            &self,
            _next: &mut Surface,
            _settings: &crate::config::HamtSettings,
            _bc: &crate::boundary::BoundaryCondition,
            _surface_prev: &Surface,
            _surface_curr: &Surface,
        ) -> anyhow::Result<()> {
            let rem = self.remaining_singular.get();
            if rem > 0 {
                self.remaining_singular.set(rem - 1);
                // Propagate the typed SolverError the same way the real
                // BlockTridiagStep does: `anyhow::Error::from(e).context(...)`.
                // The `.context()` wrapper must NOT break downcasting.
                return Err(anyhow::Error::from(SolverError::Singular { row: 35 })
                    .context("block-tridiag solve failed"));
            }
            // Success: leave next untouched; run_outer_loop will see
            // zero delta and report convergence.
            Ok(())
        }
    }

    /// A singular block-Thomas solve must NOT abort `run_outer_loop` with an
    /// error; it must return `Ok(false)` (= "did not converge") so the
    /// adaptive substepper can shrink `dt` and retry. This is the fix for the
    /// production crash where `Singular { row: 35 }` at time step 2 killed
    /// the whole TRNSYS simulation.
    #[test]
    fn singular_returns_non_convergence_not_fatal_error() {
        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);

        // Stub: fail once with Singular, then succeed.
        let step = SingularThenOk {
            remaining_singular: StdCell::new(1),
        };

        let n_cells = mgr.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);

        let result = mgr
            .run_outer_loop(
                &step,
                &mut aitken,
                None,
                ExitPolicy {
                    max_iter: 10,
                    min_iter_guard: None,
                    stall_patience: None,
                },
            )
            .expect("run_outer_loop must NOT propagate Singular as a fatal error");

        // Must report non-convergence so the caller (adaptive substepper)
        // shrinks dt and retries. The stub has one Singular invocation left
        // (run_outer_loop returned early), so no success path was reached.
        assert!(
            !result,
            "run_outer_loop must return Ok(false) on Singular so the adaptive substepper retries with smaller dt"
        );
        assert_eq!(
            step.remaining_singular.get(),
            0,
            "the stub must have been stepped exactly once (the singular call)"
        );
    }

    /// A non-Singular step error (e.g. LengthMismatch) must still propagate
    /// as a fatal error — only Singular gets the retry treatment.
    #[test]
    fn non_singular_step_error_still_fatal() {
        struct AlwaysLengthMismatch;
        impl IterativeStep for AlwaysLengthMismatch {
            fn step(
                &self,
                _next: &mut Surface,
                _settings: &crate::config::HamtSettings,
                _bc: &crate::boundary::BoundaryCondition,
                _surface_prev: &Surface,
                _surface_curr: &Surface,
            ) -> anyhow::Result<()> {
                Err(anyhow::Error::from(SolverError::LengthMismatch)
                    .context("block-tridiag solve failed"))
            }
        }

        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let step = AlwaysLengthMismatch;
        let n_cells = mgr.surface_curr.cells.len();
        let mut aitken = Aitken::new(n_cells);

        let result = mgr.run_outer_loop(
            &step,
            &mut aitken,
            None,
            ExitPolicy {
                max_iter: 10,
                min_iter_guard: None,
                stall_patience: None,
            },
        );

        assert!(
            result.is_err(),
            "non-Singular step errors must still propagate as fatal"
        );
        let err = result.unwrap_err();
        // Verify downcast still finds SolverError::LengthMismatch through
        // the context chain (proves the anyhow wrapping didn't strip type).
        assert!(
            err.downcast_ref::<SolverError>()
                .map(|e| matches!(e, SolverError::LengthMismatch))
                .unwrap_or(false),
            "fatal error must preserve SolverError::LengthMismatch for downcast"
        );
    }
}
