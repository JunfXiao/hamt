//! ADI (Alternating-Direction Implicit) inner step.
//!
//! Sequences per-axis 1-D line solves (x-sweep then y-sweep) reusing the same
//! block-Thomas kernel as the 1-D path. Along-axis material coupling is
//! implicit within each line; cross-axis coupling is lagged once per sweep and
//! converged by the nonlinear outer loop ([`crate::manager::HamtManager::run_outer_loop`]).
//!
//! The prologue (sync → apply BCs → virtual-cell refresh → update properties)
//! is identical to [`Surface::calculate_iteration_block_tridiag`]; only the
//! solve/scatter differs (multiple line sweeps instead of one assemble_1d).
//!
//! On a 1-D wall (ny=1, nz=1) `material_lines_along(0)` yields the single
//! x-line and `material_lines_along(1)` is empty, so the x-sweep reproduces the
//! block-tridiag step exactly. `solve_timestep_adi` short-circuits to the
//! BlockTridiag path for 1-D grids anyway; AdiStep handles the genuine 2-D case.

use crate::uom::units::{dimensionless, k_per_w, kelvin, pa_s_per_kg, pascals};

use super::step_jacobi::relax_write_cell;
use super::*;

use anyhow::Result;

/// ADI inner step: per-axis block-tridiagonal line sweeps (x then y).
/// See [`Surface::calculate_iteration_adi`].
pub struct AdiStep;

impl IterativeStep for AdiStep {
    fn step(
        &self,
        next: &mut Surface,
        settings: &crate::config::HamtSettings,
        bc: &crate::boundary::BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> anyhow::Result<()> {
        next.calculate_iteration_adi(settings, bc, surface_prev, surface_curr)
    }
}

impl Surface {
    /// Single outer ADI step: x-sweep then y-sweep, each a sequence of
    /// block-tridiagonal line solves sharing the block-Thomas kernel.
    ///
    /// `self` is `surface_next` (writable). The prologue mirrors
    /// [`Surface::calculate_iteration_block_tridiag`] exactly:
    ///   1. Sync `self` from `surface_curr`.
    ///   2. Apply BCs.
    ///   3. Refresh Virtual cells against the current iterate.
    ///   4. `update_cell_properties`.
    ///
    /// Then, instead of one `assemble_1d`, it runs the sweeps. The y-sweep reads
    /// the x-updated `self` for its cross-axis lag, so cross-direction coupling
    /// advances within a single outer iteration. Relaxation reference is
    /// `surface_curr` (the outer iterate), matching the BlockTridiag scatter.
    pub fn calculate_iteration_adi(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        self.prepare_iteration_prologue(settings, bc, surface_curr)?;

        // ── ADI sweeps ──
        // x-sweep: solve each x-line implicitly; y-neighbours lagged into RHS.
        self.run_axis_sweep(0, settings, bc, surface_prev, surface_curr)?;
        // y-sweep: reads the x-updated `self` for cross-axis lag.
        self.run_axis_sweep(1, settings, bc, surface_prev, surface_curr)?;

        Ok(())
    }

    /// Shared iteration prologue for the multi-dimensional (T,φ) solvers: sync
    /// the writable `self` from `surface_curr`, apply boundary coefficients,
    /// refresh Virtual cells against the current iterate, and update dependent
    /// cell properties. Both [`Surface::calculate_iteration_adi`] and the
    /// test-only direct solver ([`crate::solver::adi::DirectStep`]) share this so
    /// the ADI-vs-direct comparison differs ONLY in the solve/scatter stage.
    pub(crate) fn prepare_iteration_prologue(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_curr: &Surface,
    ) -> Result<()> {
        self.sync_state_from(surface_curr)?;
        self.apply_boundary_coefficients(bc, settings)?;

        // ── Virtual-cell refresh (3-D-correct port of the block-tridiag step) ──
        // Virtual cells aren't updated by `apply_boundary_coefficients`; the
        // sweep scatter writes only material cells. Compute each Virtual cell as
        // the conductance-weighted average of its non-Virtual neighbours' state —
        // the implicit-Euler answer for a near-zero-capacity cell — so the
        // Virtual ↔ Conv gradient the moisture-flux query relies on is non-zero.
        //
        // Unlike the 1-D block-tridiag path (which could index the `(N,1,1)`
        // array with a flat `[li,0,0]`), the ADI grid is genuinely `(nx,ny,nz)`,
        // so we walk full `[i,j,k]` triplets. Virtual cells are x-axis hubs
        // (`Virtual(Interior)` fans to all higher-x cells in *its own column*,
        // `Virtual(Exterior)` to all lower-x), so `axis_neighbors(idx, 0)`
        // collects the right per-column neighbour set.
        let virtual_nbr_pairs: Vec<([usize; 3], Vec<[usize; 3]>)> = self
            .cells
            .indexed_iter()
            .filter(|(_, cell)| matches!(cell.cell_type, crate::cell::CellType::Virtual(_)))
            .map(|((x, y, z), _)| {
                let idx = [x, y, z];
                (idx, self.axis_neighbors(idx, 0))
            })
            .collect();

        let mut virtual_updates: Vec<([usize; 3], f64, f64, f64)> = Vec::new();
        for (idx, nbr_indices) in virtual_nbr_pairs {
            let cell = &self.cells[idx];

            // strip: linear-algebra kernel boundary
            let mut sum_inv_v: f64 = 0.0;
            let mut sum_inv_v_vp: f64 = 0.0;
            let mut sum_inv_t: f64 = 0.0;
            let mut sum_inv_t_t: f64 = 0.0;

            for nbr_idx in nbr_indices {
                let nbr = &self.cells[nbr_idx];
                let (therm_r, vapor_r) =
                    Surface::calculate_resistance_pair(cell, nbr, bc.pressure_air, 0);
                if vapor_r > 1e-12 * pa_s_per_kg {
                    let inv_v = 1.0 / vapor_r.value_in(pa_s_per_kg);
                    sum_inv_v += inv_v;
                    sum_inv_v_vp += inv_v * nbr.cell_vp.value_in(pascals);
                }
                if therm_r > 1e-12 * k_per_w {
                    let inv_t = 1.0 / therm_r.value_in(k_per_w);
                    sum_inv_t += inv_t;
                    sum_inv_t_t += inv_t * nbr.cell_temp.value_in(kelvin);
                }
            }

            let new_t = if sum_inv_t > 0.0 {
                sum_inv_t_t / sum_inv_t
            } else {
                cell.cell_temp.value_in(kelvin)
            };
            let new_vp = if sum_inv_v > 0.0 {
                sum_inv_v_vp / sum_inv_v
            } else {
                cell.cell_vp.value_in(pascals)
            };
            let psat = crate::physics::psy_psat_fn_temp(new_t * kelvin).value_in(pascals);
            let new_rh = if psat > 0.0 {
                (new_vp / psat).clamp(0.0, 1.01)
            } else {
                0.0
            };
            virtual_updates.push((idx, new_vp, new_t, new_rh));
        }

        for (idx, vp, t, rh) in virtual_updates {
            let cell = &mut self.cells[idx];
            // re-tag at kernel boundary
            cell.cell_vp = vp * pascals;
            cell.cell_temp = t * kelvin;
            cell.cell_rh = rh * dimensionless;
        }

        self.update_cell_properties(bc, settings)?;

        Ok(())
    }

    /// Run a single ADI sweep along `axis`: for each material line parallel to
    /// `axis`, assemble + block-Thomas solve + scatter with α relaxation. Reads
    /// cross-axis lag from `self` (already updated by any prior sweep this
    /// iteration); relaxation reference is `surface_curr`.
    fn run_axis_sweep(
        &mut self,
        axis: usize,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        for line in self.material_lines_along(axis) {
            let sys = crate::solver::adi::assemble_line_sweep(
                self,
                surface_prev,
                bc,
                settings,
                axis,
                &line,
            );
            let sol = crate::solver::block_tridiag::solve(
                &sys.lower,
                &sys.diag,
                &sys.upper,
                &sys.rhs,
            )
            .map_err(|e| anyhow::anyhow!("ADI axis-{axis} sweep solve failed: {:?}", e))?;

            for (row, &idx) in sys.material_indices.iter().enumerate() {
                let (t_new_k, phi_new_raw) = sol[row];
                // Relaxation reference = surface_curr (the outer iterate),
                // matching the BlockTridiag scatter.
                let curr_temp = surface_curr.cells[idx].cell_temp;
                let curr_rh = surface_curr.cells[idx].cell_rh;
                let cell = self
                    .cells
                    .get_mut(idx)
                    .ok_or_else(|| anyhow::anyhow!("invalid cell index {:?}", idx))?;
                relax_write_cell(
                    cell,
                    t_new_k * kelvin,
                    phi_new_raw * dimensionless,
                    curr_temp,
                    curr_rh,
                    bc,
                    settings,
                );
            }
        }
        Ok(())
    }
}
