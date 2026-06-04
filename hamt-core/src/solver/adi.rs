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
}
