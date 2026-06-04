//! Per-axis ADI (Alternating-Direction Implicit) line-sweep assembly.
//!
//! Builds the block-tridiagonal system for ONE material line parallel to a
//! chosen `axis` and hands it to the same `block_tridiag::solve` kernel the
//! 1-D path uses. This is the multi-dimensional generalisation of
//! [`crate::solver::assembly::assemble_1d`]:
//!
//! * **Along-axis** material neighbours (the previous/next cell on the same
//!   line) are coupled *implicitly* via the off-diagonal blocks `upper`/`lower`.
//! * **Cross-axis** material neighbours AND **boundary** neighbours are *lagged*:
//!   their current-iterate value is treated as known and folded into the RHS,
//!   exactly like `assemble_1d` folds boundary neighbours into the RHS. The
//!   outer loop (`AdiStep` over x→y sweeps) iterates the cross-axis coupling to
//!   convergence.
//!
//! ## 1-D equivalence (the keystone invariant)
//!
//! On a 1-D wall (ny=1, nz=1) the single x-line has only axis-0 neighbours, so
//! every material neighbour is an along-axis line neighbour (→ off-diagonal) and
//! every boundary neighbour is lagged (→ RHS). The partition therefore collapses
//! to `assemble_1d`'s material-vs-boundary partition, and
//! `assemble_line_sweep(.., axis=0, &x_line)` reproduces `assemble_1d`
//! field-by-field. This is pinned by the equivalence test below.
//!
//! Strip discipline: typed quantities cross into raw f64 only inside the 2×2
//! block / RHS construction (the linear-algebra kernel boundary), mirroring
//! `assemble_1d`. Each `.value_in(...)` site carries a strip comment.

use std::collections::HashMap;

use crate::boundary::BoundaryCondition;
use crate::config::HamtSettings;
use crate::surface::{NeighborSums, Surface};
use crate::uom::constants::WATER_LATENT_HEAT;
use crate::uom::dimensions::{MassFlux, ThermalConductance, VaporConductance};
use crate::uom::units::{
    dimensionless, j_per_kg, k_per_w, kelvin, kg_per_s, pa_s_per_kg, pascals, w_per_k, watts,
};

use super::assembly::AssembledSystem;
use super::block_2x2::Block2x2;

/// Assemble the block-tridiagonal system for ONE material line parallel to
/// `axis`. Reuses [`Surface::build_cell_block`] for each cell's diagonal +
/// cell-local RHS. ALONG-axis material neighbours go to the off-diagonals
/// (upper/lower by position in `line`); CROSS-axis material neighbours AND
/// boundary neighbours are LAGGED into the RHS (their current-iterate value
/// treated as known). `line` is the ordered list of `[i,j,k]` material cells
/// along `axis` (one entry of [`Surface::material_lines_along`]).
///
/// `surface` carries the freshly-prepared iteration state (material cells hold
/// the current iterate's (T, φ); boundary/virtual cells were refreshed against
/// that same iterate); `surface_prev` is the t-1 anchor for the inertial term.
/// `update_cell_properties` must already have refreshed `vp_sat`, etc.
pub(crate) fn assemble_line_sweep(
    surface: &Surface,
    surface_prev: &Surface,
    bc: &BoundaryCondition,
    settings: &HamtSettings,
    axis: usize,
    line: &[[usize; 3]],
) -> AssembledSystem {
    let n = line.len();

    // Position of each cell within this line, so an along-axis material
    // neighbour can be classified as row-1 (lower) / row+1 (upper).
    let pos_in_line: HashMap<[usize; 3], usize> =
        line.iter().copied().enumerate().map(|(r, t)| (t, r)).collect();

    let mut lower = vec![Block2x2::zero(); n];
    let mut diag = vec![Block2x2::zero(); n];
    let mut upper = vec![Block2x2::zero(); n];
    let mut rhs = vec![(0.0, 0.0); n];

    let dt_s = settings.timestep;

    for (row, &idx) in line.iter().enumerate() {
        let cell_curr = &surface.cells[idx];
        let cell_prev = &surface_prev.cells[idx];

        // L_eff at lagged state (identical to assemble_1d). When
        // latent_heat_enabled is false the cross-coupling vanishes entirely.
        let l_eff: crate::uom::dimensions::SpecificEnergy<f64> = if settings.latent_heat_enabled {
            1.0 * WATER_LATENT_HEAT
                + if settings.sorption_enthalpy_enabled {
                    cell_curr.h_sorp_diff
                } else {
                    0.0 * j_per_kg
                }
        } else {
            0.0 * j_per_kg
        };

        // ── All-neighbour conductance sums (typed) feeding the diagonal ──
        // build_cell_block needs oorsum/vpoosum/phioosum summed over ALL
        // neighbours (along-axis material, cross-axis material, AND boundary).
        let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
        let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
        let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s;

        // Lagged-neighbour RHS contributions (cross-axis material + boundary).
        // Accumulated in raw f64 (kernel boundary).
        let mut b_t_lag: f64 = 0.0; // [W]
        let mut b_p_lag: f64 = 0.0; // [kg/s]

        // Walk every face neighbour across every active axis once.
        for (nbr_idx, nbr_axis) in surface.all_face_neighbors(idx) {
            let nbr_curr = &surface.cells[nbr_idx];

            // Resistance triple using the lagged state and the neighbour's axis
            // (same guards as assemble_1d / accumulate_neighbor_sums).
            let r = Surface::calculate_resistance_triple(
                cell_curr,
                nbr_curr,
                bc.pressure_air,
                nbr_axis,
            );

            let cond_th: ThermalConductance<f64> = if r.therm > 1e-12 * k_per_w {
                1.0 / r.therm
            } else {
                0.0 * w_per_k
            };
            let cond_vapor: VaporConductance<f64> = if r.vapor > 1e-12 * pa_s_per_kg {
                1.0 / r.vapor
            } else {
                0.0 * kg_per_s / pascals
            };
            let r_total_phi = r.liquid * dimensionless;
            let cond_phi: MassFlux<f64> = if r_total_phi > 1e-13 * dimensionless / kg_per_s
                && r_total_phi < f64::MAX * dimensionless / kg_per_s
            {
                1.0 / r_total_phi
            } else {
                0.0 * kg_per_s
            };

            // All-neighbour conductance sums for the diagonal block.
            oorsum += cond_th;
            vpoosum += cond_vapor;
            phioosum += cond_phi;

            // strip: linear-algebra kernel boundary
            let inv_th = cond_th.value_in(w_per_k);
            let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
            let inv_phi = cond_phi.value_in(kg_per_s);
            let l_eff_v = l_eff.value_in(j_per_kg);
            let psat_n = nbr_curr.vp_sat.value_in(pascals);

            // An along-axis material neighbour that lies on THIS line at row±1
            // is coupled implicitly (off-diagonal). Everything else
            // (cross-axis material neighbour OR boundary neighbour) is lagged
            // into the RHS as a known value for this sweep.
            let along_axis_line_nbr = nbr_axis == axis
                && pos_in_line
                    .get(&nbr_idx)
                    .is_some_and(|&nrow| nrow + 1 == row || nrow == row + 1);

            if along_axis_line_nbr {
                // Off-diagonal block for this side (same form as assemble_1d):
                //   block_tt = -inv_th
                //   block_tp = -L_eff · inv_vapor · Psat_n
                //   block_pt = 0
                //   block_pp = -(inv_phi + inv_vapor · Psat_n)
                let off = Block2x2 {
                    a: -inv_th,
                    b: -l_eff_v * inv_vapor * psat_n,
                    c: 0.0,
                    d: -(inv_phi + inv_vapor * psat_n),
                };
                let nbr_row = pos_in_line[&nbr_idx];
                // Safety-net: along-axis material neighbour must be the immediately-adjacent
                // member in the line ordering (|row − nbr_row| == 1) so it maps to a
                // contiguous off-diagonal; any gap would silently skip rows in the tridiagonal.
                debug_assert!(
                    nbr_row + 1 == row || row + 1 == nbr_row,
                    "along-axis neighbour {nbr_idx:?} at line position {nbr_row} is not \
                     adjacent to cell {idx:?} at position {row} — ordering or topology bug"
                );
                if nbr_row == row + 1 {
                    upper[row] = off;
                } else {
                    // nbr_row + 1 == row (guaranteed by along_axis_line_nbr)
                    lower[row] = off;
                }
            } else {
                // Lagged neighbour (cross-axis material OR boundary): fold the
                // neighbour-VALUE coupling into the RHS using its current
                // iterate. Matches assemble_1d's boundary-RHS form, extended to
                // cross-axis material neighbours (known in this sweep).
                // strip: linear-algebra kernel boundary — RHS scalars fed into raw f64 accumulator
                let temp_n = nbr_curr.cell_temp.value_in(kelvin);
                let vp_n = nbr_curr.cell_vp.value_in(pascals);
                let phi_or_rh = nbr_curr.cell_rh.value();
                b_t_lag += inv_th * temp_n + l_eff_v * inv_vapor * vp_n;
                b_p_lag += inv_phi * phi_or_rh + inv_vapor * vp_n;
            }
        }

        // ── Cell-local diagonal block + cell-local RHS via the SHARED builder ──
        // The or-sums (neighbour-VALUE coupling) are zero here: the along-axis
        // material part is in the off-diagonals and the lagged part is in the
        // RHS via b_t_lag/b_p_lag.
        let sums = NeighborSums {
            oorsum,
            torsum: 0.0 * watts,
            vapor_diffusion_net_flux: 0.0 * kg_per_s,
            vpoosum,
            vporsum: 0.0 * kg_per_s,
            phioosum,
            phiorsum: 0.0 * kg_per_s,
        };
        let inertia = Surface::cell_inertia(idx[0], cell_curr);
        let sources = surface.cell_sources(idx, surface, bc);

        let (block, (b_t0, b_p0)) =
            Surface::build_cell_block(&sums, &inertia, &sources, cell_curr, cell_prev, l_eff, dt_s);

        diag[row] = block;
        rhs[row] = (b_t0 + b_t_lag, b_p0 + b_p_lag);
    }

    AssembledSystem {
        lower,
        diag,
        upper,
        rhs,
        material_indices: line.to_vec(),
    }
}

// ===========================================================================
// Test-only DIRECT (fully-coupled, no cross-axis lag) solver — the ADI oracle.
// ===========================================================================

/// Test-support fully-coupled direct solver used as an oracle for the ADI
/// fixed point. It performs ONE Picard iterate of the SAME frozen-nonlinearity
/// system that `AdiStep` converges, but assembles the WHOLE `2*Nmat` block-sparse
/// system densely (every material face neighbour — x AND y — enters as an
/// off-diagonal 2×2 block, no direction is lagged) and solves it exactly via
/// Gaussian elimination with partial pivoting.
///
/// At the shared fixed point the per-iterate map of `DirectStep` and `AdiStep`
/// agree, because the ADI lag terms vanish when the cross-direction values stop
/// changing. So driving both through `run_outer_loop` from the same state must
/// land on the same solution; the only difference is HOW MANY outer iterations
/// each needs. This is the strong-coupling stability gate (Phase 2a Task 10).
#[cfg(test)]
pub(crate) struct DirectStep;

#[cfg(test)]
impl crate::surface::IterativeStep for DirectStep {
    fn step(
        &self,
        next: &mut Surface,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> anyhow::Result<()> {
        // Identical prologue to AdiStep (sync → BCs → virtual refresh → props),
        // so the comparison isolates the solve/scatter stage.
        next.prepare_iteration_prologue(settings, bc, surface_curr)?;
        next.calculate_iteration_direct(settings, bc, surface_prev, surface_curr)
    }
}

#[cfg(test)]
impl Surface {
    /// Solve ONE fully-coupled Picard iterate densely (oracle for ADI).
    ///
    /// `self` is `surface_next`, already prepared via
    /// [`Surface::prepare_iteration_prologue`]. Assembles the global
    /// `(2*Nmat) x (2*Nmat)` system (diagonal from `build_cell_block`,
    /// off-diagonal 2×2 blocks for EVERY material face neighbour across all
    /// active axes, boundary neighbours into the RHS) and solves it by Gaussian
    /// elimination with partial pivoting. Scatters with `relax_write_cell`
    /// (relaxation reference = `surface_curr`), matching AdiStep.
    pub(crate) fn calculate_iteration_direct(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> anyhow::Result<()> {
        use crate::surface::step_jacobi::relax_write_cell;

        let mat_idx = self.material_cell_indices();
        let n = mat_idx.len();
        let dt_s = settings.timestep;

        // cell [i,j,k] → block-row index (0..n). HashMap per the task spec.
        let block_row: HashMap<[usize; 3], usize> =
            mat_idx.iter().copied().enumerate().map(|(r, t)| (t, r)).collect();

        // Dense system: 2n scalar unknowns ordered (T_0, φ_0, T_1, φ_1, ...).
        let dim = 2 * n;
        let mut a = vec![vec![0.0_f64; dim]; dim];
        let mut b = vec![0.0_f64; dim];

        for (row, &idx) in mat_idx.iter().enumerate() {
            let cell_curr = &self.cells[idx];
            let cell_prev = &surface_prev.cells[idx];

            // L_eff at lagged state (identical to assemble_line_sweep).
            let l_eff: crate::uom::dimensions::SpecificEnergy<f64> =
                if settings.latent_heat_enabled {
                    1.0 * WATER_LATENT_HEAT
                        + if settings.sorption_enthalpy_enabled {
                            cell_curr.h_sorp_diff
                        } else {
                            0.0 * j_per_kg
                        }
                } else {
                    0.0 * j_per_kg
                };

            // All-neighbour conductance sums for the diagonal block.
            let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
            let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
            let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s;

            // Boundary-neighbour RHS contributions (raw f64, kernel boundary).
            let mut b_t_boundary: f64 = 0.0; // [W]
            let mut b_p_boundary: f64 = 0.0; // [kg/s]

            // Off-diagonal blocks staged by neighbour block-row.
            let mut off_blocks: Vec<(usize, Block2x2)> = Vec::new();

            for (nbr_idx, nbr_axis) in self.all_face_neighbors(idx) {
                let nbr_curr = &self.cells[nbr_idx];
                let r = Surface::calculate_resistance_triple(
                    cell_curr,
                    nbr_curr,
                    bc.pressure_air,
                    nbr_axis,
                );

                let cond_th: ThermalConductance<f64> = if r.therm > 1e-12 * k_per_w {
                    1.0 / r.therm
                } else {
                    0.0 * w_per_k
                };
                let cond_vapor: VaporConductance<f64> = if r.vapor > 1e-12 * pa_s_per_kg {
                    1.0 / r.vapor
                } else {
                    0.0 * kg_per_s / pascals
                };
                let r_total_phi = r.liquid * dimensionless;
                let cond_phi: MassFlux<f64> = if r_total_phi > 1e-13 * dimensionless / kg_per_s
                    && r_total_phi < f64::MAX * dimensionless / kg_per_s
                {
                    1.0 / r_total_phi
                } else {
                    0.0 * kg_per_s
                };

                oorsum += cond_th;
                vpoosum += cond_vapor;
                phioosum += cond_phi;

                // strip: linear-algebra kernel boundary
                let inv_th = cond_th.value_in(w_per_k);
                let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
                let inv_phi = cond_phi.value_in(kg_per_s);
                let l_eff_v = l_eff.value_in(j_per_kg);
                let psat_n = nbr_curr.vp_sat.value_in(pascals);

                if let Some(&nbr_row) = block_row.get(&nbr_idx) {
                    // Material neighbour (ANY axis): fully-coupled off-diagonal
                    // block — the SAME `off` pattern AdiStep uses along-axis,
                    // here applied to EVERY material face neighbour with no lag.
                    let off = Block2x2 {
                        a: -inv_th,
                        b: -l_eff_v * inv_vapor * psat_n,
                        c: 0.0,
                        d: -(inv_phi + inv_vapor * psat_n),
                    };
                    off_blocks.push((nbr_row, off));
                } else {
                    // Boundary neighbour → RHS, exactly as the sweeps fold it.
                    // strip: dense Gaussian-elimination oracle (small linear-algebra kernel) —
                    // RHS scalars fed into raw f64 accumulator at the kernel boundary
                    let temp_n = nbr_curr.cell_temp.value_in(kelvin);
                    let vp_n = nbr_curr.cell_vp.value_in(pascals);
                    let phi_b = nbr_curr.cell_rh.value();
                    b_t_boundary += inv_th * temp_n + l_eff_v * inv_vapor * vp_n;
                    b_p_boundary += inv_phi * phi_b + inv_vapor * vp_n;
                }
            }

            let sums = NeighborSums {
                oorsum,
                torsum: 0.0 * watts,
                vapor_diffusion_net_flux: 0.0 * kg_per_s,
                vpoosum,
                vporsum: 0.0 * kg_per_s,
                phioosum,
                phiorsum: 0.0 * kg_per_s,
            };
            let inertia = Surface::cell_inertia(idx[0], cell_curr);
            let sources = self.cell_sources(idx, self, bc);

            let (block, (b_t0, b_p0)) = Surface::build_cell_block(
                &sums, &inertia, &sources, cell_curr, cell_prev, l_eff, dt_s,
            );

            // Scatter the diagonal 2×2 into the dense matrix.
            let r0 = 2 * row; // T-equation row
            let r1 = 2 * row + 1; // φ-equation row
            let c0 = 2 * row; // T column
            let c1 = 2 * row + 1; // φ column
            a[r0][c0] += block.a;
            a[r0][c1] += block.b;
            a[r1][c0] += block.c;
            a[r1][c1] += block.d;

            // Scatter the off-diagonal 2×2 blocks.
            for (nbr_row, off) in off_blocks {
                let nc0 = 2 * nbr_row;
                let nc1 = 2 * nbr_row + 1;
                a[r0][nc0] += off.a;
                a[r0][nc1] += off.b;
                a[r1][nc0] += off.c;
                a[r1][nc1] += off.d;
            }

            b[r0] = b_t0 + b_t_boundary;
            b[r1] = b_p0 + b_p_boundary;
        }

        // ── Dense Gaussian elimination with partial pivoting. ──
        let x = gaussian_solve(a, b)
            .ok_or_else(|| anyhow::anyhow!("direct dense solve: singular system"))?;

        // ── Scatter with α relaxation (reference = surface_curr), like AdiStep. ──
        for (row, &idx) in mat_idx.iter().enumerate() {
            let t_new_k = x[2 * row];
            let phi_new_raw = x[2 * row + 1];
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

        Ok(())
    }
}

/// Solve `A x = b` for a dense square system via Gaussian elimination with
/// partial pivoting. Returns `None` on a (near-)singular pivot. Test-support
/// only — small systems, clarity over speed.
#[cfg(test)]
fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);

    for col in 0..n {
        // Partial pivot: largest |a[r][col]| in rows col..n.
        let mut pivot = col;
        let mut pivot_mag = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(n).skip(col + 1) {
            let m = row[col].abs();
            if m > pivot_mag {
                pivot_mag = m;
                pivot = r;
            }
        }
        if pivot_mag < 1e-300 || !pivot_mag.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        // Eliminate below the pivot. `split_at_mut` separates the pivot row
        // (immutable source) from the rows being updated (mutable target),
        // avoiding a simultaneous index borrow of `a`.
        let (head, tail) = a.split_at_mut(col + 1);
        let pivot_row = &head[col];
        let pivot_val = pivot_row[col];
        let b_col = b[col];
        for (offset, target_row) in tail.iter_mut().enumerate() {
            let r = col + 1 + offset;
            let factor = target_row[col] / pivot_val;
            if factor == 0.0 {
                continue;
            }
            for (tc, pc) in target_row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *tc -= factor * *pc;
            }
            b[r] -= factor * b_col;
        }
    }

    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let row_vals = &a[row];
        let acc0 = b[row];
        let acc = x
            .iter()
            .enumerate()
            .skip(row + 1)
            .fold(acc0, |acc, (c, &xc)| acc - row_vals[c] * xc);
        let diag = a[row][row];
        if diag.abs() < 1e-300 || !diag.is_finite() {
            return None;
        }
        x[row] = acc / diag;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n_div: usize) -> crate::manager::HamtManager {
        crate::solver::test_fixtures::build_fixture_manager(n_div)
    }

    /// KEYSTONE: on a (N,1,1) wall, `assemble_line_sweep(.., axis=0, &x_line)`
    /// must equal `assemble_1d` field-by-field. This pins the ADI assembler
    /// against the verified 1-D path.
    #[test]
    fn line_sweep_axis0_matches_assemble_1d_on_1d_wall() {
        let mut mgr = fixture(6);

        // Mirror block_tridiag_tests' setup: refresh BC + dependent state on
        // both surfaces so all coefficients (incl. boundary/virtual cells) are
        // populated and consistent with the material-cell iterate.
        for surf in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            surf.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            surf.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }

        let lines = mgr.surface_curr.material_lines_along(0);
        assert_eq!(lines.len(), 1, "1-D wall must have exactly one x-line");
        let line = &lines[0];

        let sweep = assemble_line_sweep(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
            0,
            line,
        );
        let direct = crate::solver::assembly::assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );

        assert_eq!(
            sweep.material_indices, direct.material_indices,
            "material_indices differ"
        );
        let n = direct.diag.len();
        assert_eq!(sweep.diag.len(), n, "diag length differs");

        let tol = 1e-12;
        let cmp_block = |a: &Block2x2, b: &Block2x2, what: &str, row: usize| {
            assert!(
                (a.a - b.a).abs() < tol
                    && (a.b - b.b).abs() < tol
                    && (a.c - b.c).abs() < tol
                    && (a.d - b.d).abs() < tol,
                "{what}[{row}] differs: sweep={a:?} direct={b:?}"
            );
        };
        for row in 0..n {
            cmp_block(&sweep.diag[row], &direct.diag[row], "diag", row);
            cmp_block(&sweep.lower[row], &direct.lower[row], "lower", row);
            cmp_block(&sweep.upper[row], &direct.upper[row], "upper", row);
            assert!(
                (sweep.rhs[row].0 - direct.rhs[row].0).abs() < tol
                    && (sweep.rhs[row].1 - direct.rhs[row].1).abs() < tol,
                "rhs[{row}] differs: sweep={:?} direct={:?}",
                sweep.rhs[row],
                direct.rhs[row]
            );
        }
    }

    /// Sanity: `gaussian_solve` recovers the solution of a small known system.
    #[test]
    fn gaussian_solve_recovers_known_solution() {
        // A x = b, with x = (1, -2, 3).
        //   2x + y -  z =  8 - ... build from x to be exact.
        let mat = vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ];
        let x_true = [1.0_f64, -2.0, 3.0];
        let b: Vec<f64> = mat
            .iter()
            .map(|row| row[0] * x_true[0] + row[1] * x_true[1] + row[2] * x_true[2])
            .collect();
        let x = gaussian_solve(mat, b).expect("non-singular");
        for (g, t) in x.iter().zip(x_true.iter()) {
            assert!((g - t).abs() < 1e-12, "got {g}, expected {t}");
        }
    }

    /// STRONG-COUPLING STABILITY GATE (Phase 2a Task 10).
    ///
    /// On a small 2-D thermal-bridge config (ACTIVE y-faces + a strong initial
    /// y-gradient in T and φ, so the cross-axis off-diagonal blocks carry real
    /// flux — the regime where ADI's lagged-direction splitting is weakest),
    /// `AdiStep`+Anderson driven by `run_outer_loop` must converge to the SAME
    /// fixed point as the fully-coupled DIRECT dense oracle. We assert agreement
    /// to 1e-7 and RECORD the outer-iteration count of each — the count is the
    /// decision datum for whether ADI is sufficient or escalation is needed.
    #[test]
    fn adi_matches_direct_on_thermal_bridge() {
        use crate::cell::CellType;
        use crate::uom::units::{dimensionless, kelvin};

        // Small grid: thin 1-layer wall, ~6 x-cells, ny = 6, y-faces ACTIVE.
        // Nmat = 6*6 = 36 → dense system is 72×72 (cheap).
        let n_div_x = 6usize;
        let ny = 6usize;
        let mut base = crate::solver::test_fixtures::build_fixture_manager_2d(n_div_x, ny, true);

        let dim = base.surface_curr.cells.raw_dim();
        eprintln!(
            "thermal-bridge grid: ({}, {}, {}), cells={}",
            dim[0], dim[1], dim[2], base.surface_curr.cells.len()
        );

        // ── Impose a STRONG off-axis (y) coupling feature. ──
        // The global BoundaryCondition cannot express per-row interior
        // temperatures, so (per the task) we initialise the wall with a strong
        // y-gradient: row 0 hot & wet, row ny-1 cold & dry, with a mild x-tilt
        // so the bridge is genuinely 2-D (not separable). This makes every
        // material cell differ from its y-neighbour, so the cross-axis 2×2
        // off-diagonals (lagged by ADI, fully implicit by Direct) carry large
        // flux on the first iterates — the strong-coupling stress case.
        let nx = dim[0];
        let bc = base.boundary_condition.clone();
        let settings = base.settings.clone();
        for surf in [
            &mut base.surface_prev,
            &mut base.surface_curr,
            &mut base.surface_next,
        ] {
            for j in 0..ny {
                let fy = j as f64 / (ny as f64 - 1.0);
                for i in 0..nx {
                    let is_mat =
                        matches!(surf.cells[[i, j, 0]].cell_type, CellType::Material(_));
                    if !is_mat {
                        continue;
                    }
                    let fx = i as f64 / (nx as f64 - 1.0);
                    let t_c = 35.0 - 25.0 * fy + 4.0 * (fx - 0.5);
                    let phi = (0.80 - 0.55 * fy + 0.05 * (fx - 0.5)).clamp(0.05, 0.99);
                    let cell = &mut surf.cells[[i, j, 0]];
                    cell.cell_temp = crate::boundary::celsius_to_kelvin(t_c);
                    cell.cell_rh = phi * dimensionless;
                }
            }
            surf.update_cell_properties(&bc, &settings).unwrap();
        }

        // Solve ONE implicit-Euler step (dt = 300 s) to a tight 1e-9 fixed point
        // so the two methods land on the SAME solution to well within the 1e-7
        // agreement gate. `min_relaxation_factor = 1.0` removes Aitken's
        // down-damping (which otherwise throttles BOTH solvers equally and just
        // inflates the shared iteration count): we want the iteration count to
        // reflect the cross-axis-lag cost of ADI relative to the fully-coupled
        // DIRECT oracle, not relaxation tuning.
        base.settings.timestep = 300.0 * crate::uom::units::seconds;
        base.settings.temp_convergence_tolerance = 1e-9 * kelvin;
        base.settings.rh_convergence_tolerance = 1e-9 * dimensionless;
        base.settings.min_relaxation_factor = 1.0;
        base.settings.solver_method = crate::manager::SolverMethod::Adi;
        base.settings.anderson_enabled = true;

        // Two independent runs from the SAME initial state + settings.
        let mut mgr_adi = base.clone();
        let mut mgr_dir = base.clone();

        let max_iter = 500;
        let (conv_adi, iters_adi) = mgr_adi
            .run_outer_loop_counted(&crate::surface::AdiStep, max_iter)
            .expect("ADI outer loop");
        let (conv_dir, iters_dir) = mgr_dir
            .run_outer_loop_counted(&DirectStep, max_iter)
            .expect("Direct outer loop");

        // ── Agreement: compare final material T and φ (in surface_curr). ──
        let mut max_dt = 0.0_f64;
        let mut max_dphi = 0.0_f64;
        for (ca, cb) in mgr_adi
            .surface_curr
            .cells
            .iter()
            .zip(mgr_dir.surface_curr.cells.iter())
        {
            if !matches!(ca.cell_type, CellType::Material(_)) {
                continue;
            }
            let dt = (ca.cell_temp - cb.cell_temp).value_in(kelvin).abs();
            let dphi = (ca.cell_rh - cb.cell_rh).value().abs();
            max_dt = max_dt.max(dt);
            max_dphi = max_dphi.max(dphi);
        }

        // KEY RESULT: ADI's outer-iteration count vs the fully-coupled DIRECT
        // oracle's. Their RATIO isolates the cost of ADI's cross-axis lag — the
        // shared (high) absolute count is the fixture's intrinsic nonlinear
        // Picard cost, which DIRECT (an EXACT linear solve every iterate) incurs
        // just the same. ADI ≈ DIRECT ⇒ the lag adds no meaningful overhead.
        let ratio = iters_adi as f64 / iters_dir.max(1) as f64;
        eprintln!(
            "=== ADI-vs-DIRECT thermal-bridge result (Phase 2a Task 10) ===\n\
             ADI:    converged={conv_adi}, outer_iters={iters_adi}\n\
             DIRECT: converged={conv_dir}, outer_iters={iters_dir}\n\
             ADI/DIRECT iteration ratio = {ratio:.3}\n\
             agreement: max|ΔT|={max_dt:.3e} K, max|Δφ|={max_dphi:.3e} (gate 1e-7)"
        );

        assert!(conv_adi, "ADI did not converge within {max_iter} outer iters");
        assert!(conv_dir, "DIRECT did not converge within {max_iter} outer iters");
        // Same fixed point (the load-bearing equivalence assertion).
        assert!(
            max_dt < 1e-7 && max_dphi < 1e-7,
            "ADI and DIRECT disagree: max|ΔT|={max_dt:.3e} K, max|Δφ|={max_dphi:.3e} \
             (gate 1e-7). ADI iters={iters_adi}, DIRECT iters={iters_dir}"
        );
        // Cross-axis lag must not blow up: ADI stays within a small multiple of
        // the fully-coupled oracle. (Empirically ratio ≈ 1.0–1.3 here.)
        assert!(
            ratio <= 1.75,
            "ADI cross-axis lag inflated iterations {ratio:.2}× over DIRECT \
             (ADI={iters_adi}, DIRECT={iters_dir}) — escalation (nested-loop / Krylov) indicated"
        );
    }
}
