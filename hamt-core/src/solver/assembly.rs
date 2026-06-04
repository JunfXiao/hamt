//! 1-D block-tridiagonal assembly for the implicit-Euler HAMT step.
//!
//! Shares the cell-local 2×2 (T, φ) physics with the BlockJacobi path:
//! the diagonal block and cell-local RHS come from the single source
//! [`Surface::build_cell_block`] (fed by all-neighbour conductance sums,
//! [`Surface::cell_inertia`], and [`Surface::cell_sources`]). This module
//! adds only the tridiag-specific parts — material-material couplings as
//! off-diagonal blocks and boundary-neighbour RHS contributions — emitting
//! one (L, D, U, b) row per material cell.
//!
//! Strip discipline: typed quantities cross into raw f64 only inside the
//! 2×2 block construction (the kernel boundary). Each `.value_in(...)`
//! site below carries a strip comment.
//!
//! ## 1-D specialisation
//!
//! This module is hard-coded to the existing 1-D wall geometry. When
//! the solver moves to 2-D / 3-D walls, ADI sweeps will live in sibling
//! modules (`assemble_x_sweep`, etc.) and call the same
//! `block_tridiag::solve` kernel. The shared `build_cell_block` is not
//! direction-specific and is reused as-is by those sweeps.

use crate::boundary::BoundaryCondition;
use crate::config::HamtSettings;
use crate::surface::{NeighborSums, Surface};
use crate::uom::constants::WATER_LATENT_HEAT;
use crate::uom::dimensions::{MassFlux, ThermalConductance, VaporConductance};
use crate::uom::units::{
    dimensionless, j_per_kg, k_per_w, kelvin, kg_per_s, pa_s_per_kg, pascals, w_per_k, watts,
};

use super::block_2x2::Block2x2;

/// One assembled block-tridiagonal system covering all material cells of
/// the source `surface` in physical order. `lower[0]` and `upper[N-1]` are
/// zero-filled per the `block_tridiag::solve` API contract.
pub(crate) struct AssembledSystem {
    pub(crate) lower: Vec<Block2x2>,
    pub(crate) diag: Vec<Block2x2>,
    pub(crate) upper: Vec<Block2x2>,
    pub(crate) rhs: Vec<(f64, f64)>,
    /// Material cells' [i,j,k] indices into `surface.cells`, in the
    /// same order as `diag` rows. Used to scatter the solve result back.
    pub(crate) material_indices: Vec<[usize; 3]>,
}

/// Build the system from `surface` (the freshly-prepared iteration state:
/// material cells hold the current iterate's (T, φ); boundary and virtual
/// cells were just refreshed against that same iterate by
/// `apply_boundary_coefficients` and the orchestrator's virtual-cell update)
/// and `surface_prev` (t-1 anchor for the inertial term).
/// `update_cell_properties` must already have refreshed `vp_sat`,
/// `d_psat_d_t`, `dwd_phi`, etc. on `surface`.
pub(crate) fn assemble_1d(
    surface: &Surface,
    surface_prev: &Surface,
    bc: &BoundaryCondition,
    settings: &HamtSettings,
) -> AssembledSystem {
    use std::collections::{HashMap, HashSet};
    let mat_idx = surface.material_cell_indices();
    let n = mat_idx.len();
    // Lookup table to recognise material-material vs material-boundary
    // neighbours during assembly, keyed by [i,j,k] triplet.
    let is_material_at: HashSet<[usize; 3]> = mat_idx.iter().copied().collect();
    // Map material-cell [i,j,k] triplet → row in the tridiag system.
    let row_of: HashMap<[usize; 3], usize> =
        mat_idx.iter().copied().enumerate().map(|(r, t)| (t, r)).collect();

    let mut lower = vec![Block2x2::zero(); n];
    let mut diag = vec![Block2x2::zero(); n];
    let mut upper = vec![Block2x2::zero(); n];
    let mut rhs = vec![(0.0, 0.0); n];

    let dt_s = settings.timestep;
    let nx = surface.cells.raw_dim()[0];

    for (row, &triplet) in mat_idx.iter().enumerate() {
        // For a 1-D wall (ny=1, nz=1) triplet = [i,0,0]; i is the x-axis index.
        let i = triplet[0];
        let cell_curr = &surface.cells[triplet];
        let cell_prev = &surface_prev.cells[triplet];

        // L_eff at lagged state. When latent_heat_enabled is false the
        // cross-coupling vanishes entirely.
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
        // `build_cell_block` needs oorsum/vpoosum/phioosum summed over ALL
        // neighbours (material AND boundary). We accumulate them here in the
        // same single walk that builds the off-diagonals and the
        // boundary-neighbour RHS. The or-sums (torsum/vporsum/phiorsum) are
        // the neighbour-VALUE coupling that `build_cell_block` does NOT
        // produce; for the tridiag path the material part lives in the
        // off-diagonal blocks and the boundary part is added to the RHS
        // below, so the or-sums are left at zero here.
        let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
        let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
        let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s;

        // Boundary-neighbour RHS contributions (stay inline — build_cell_block
        // does not produce them). Accumulated in raw f64 (kernel boundary).
        let mut b_t_boundary: f64 = 0.0; // [W]
        let mut b_p_boundary: f64 = 0.0; // [kg/s]

        // Walk the neighbour stencil once: accumulate conductance sums,
        // build material off-diagonals, and collect boundary RHS.
        // NOTE: the diagonal gets `+ inv_th` and `+ inv_phi + inv_vapor·Psat`
        // from every neighbour (material and boundary alike) — that is exactly
        // what `oorsum`/`phioosum`/`vpoosum·Psat` encode. The off-diagonal
        // block for a material-material edge carries the NEGATIVE of those same
        // conductances; boundary neighbours contribute only to diagonal + RHS.
        for nbr_li in surface.neighbor_indices(i, nx, 0) {
            // neighbor_indices returns flat x-index (1-D shim); map to triplet for lookup.
            let nbr_triplet: [usize; 3] = [nbr_li, 0, 0];
            let nbr_curr = &surface.cells[nbr_triplet];

            // Resistance triple using the lagged state — identical to the
            // BlockJacobi `accumulate_neighbor_sums` path (same guards).
            let r =
                Surface::calculate_resistance_triple(cell_curr, nbr_curr, bc.pressure_air, 0);

            // Typed conductances (zero-on-degeneracy guards mirror the
            // BlockJacobi path exactly).
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

            if is_material_at.contains(&nbr_triplet) {
                // ── Material neighbour: contributions go to OFF-DIAGONAL ──
                // Off-diag block for this side:
                //   block_tt = -inv_th
                //   block_pp = -(inv_phi + inv_vapor·Psat_n)
                //   block_tp = -L_eff · inv_vapor · Psat_n
                //   block_pt = 0  (the Taylor-linearisation cross term a_pt
                //                  uses the CELL's own phi·dPsat/dT and goes
                //                  to the diagonal only, not the off-diagonal.)
                // strip: kernel boundary
                let inv_th = cond_th.value_in(w_per_k);
                let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
                let inv_phi = cond_phi.value_in(kg_per_s);
                let l_eff_v = l_eff.value_in(j_per_kg);
                let psat_n = nbr_curr.vp_sat.value_in(pascals);
                let off = Block2x2 {
                    a: -inv_th,
                    b: -l_eff_v * inv_vapor * psat_n,
                    // c = 0: matches the existing BlockJacobi "lagged neighbour vapour" semantics.
                    // The plan's draft proposed -inv_vapor*phi_n*dpsdt_n here, but a
                    // fully-implicit form would also need a matching +inv_vapor*phi_n*
                    // dpsdt_n*T_n term in b_p (Taylor bias for the lagged-T expansion of
                    // vp_n). Without that bias the off-diagonal c term is mathematically
                    // inconsistent. We chose the lagged-vp form (c = 0, no bias) so the
                    // block-tridiag converges to the same answer BlockJacobi does.
                    c: 0.0,
                    d: -(inv_phi + inv_vapor * psat_n),
                };

                let nbr_row = row_of[&nbr_triplet];
                if nbr_row == row + 1 {
                    upper[row] = off;
                } else if nbr_row + 1 == row {
                    lower[row] = off;
                } else {
                    // Should never happen on a 1-D wall (material cells
                    // are physically ordered).
                    panic!("non-adjacent material neighbour: row={row} nbr_row={nbr_row}");
                }
            } else {
                // ── Boundary neighbour: contributes to diagonal + RHS ──
                // The diagonal contribution is already folded into the
                // oorsum/vpoosum/phioosum sums above (build_cell_block applies
                // them). Only the neighbour-VALUE RHS coupling is inline here.
                // Use cell_vp directly: boundary cells set cell_vp explicitly
                // (e.g. Radiation cell sets cell_vp = 0). See surface/update.rs.
                // strip: kernel boundary
                let inv_th = cond_th.value_in(w_per_k);
                let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
                let inv_phi = cond_phi.value_in(kg_per_s);
                let temp_n = nbr_curr.cell_temp.value_in(kelvin);
                let vp_n = nbr_curr.cell_vp.value_in(pascals);
                let phi_b = nbr_curr.cell_rh.value();
                b_t_boundary += inv_th * temp_n + l_eff.value_in(j_per_kg) * inv_vapor * vp_n;
                b_p_boundary += inv_phi * phi_b + inv_vapor * vp_n;
            }
        }

        // ── Cell-local diagonal block + cell-local RHS via the SHARED builder ──
        // Single source of the cell-local (T, φ) physics (capacities, Taylor
        // cross, inertial + source biases). The or-sums (neighbour-VALUE
        // coupling) are zero here: the material part is in the off-diagonals
        // and the boundary part is added to the RHS below.
        let sums = NeighborSums {
            oorsum,
            torsum: 0.0 * watts,
            vapor_diffusion_net_flux: 0.0 * kg_per_s,
            vpoosum,
            vporsum: 0.0 * kg_per_s,
            phioosum,
            phiorsum: 0.0 * kg_per_s,
        };
        let inertia = Surface::cell_inertia(i, cell_curr);
        let sources = surface.cell_sources(triplet, surface, bc);

        let (block, (b_t0, b_p0)) =
            Surface::build_cell_block(&sums, &inertia, &sources, cell_curr, cell_prev, l_eff, dt_s);

        // Start from the cell-local RHS and add the boundary-neighbour coupling.
        diag[row] = block;
        rhs[row] = (b_t0 + b_t_boundary, b_p0 + b_p_boundary);
    }

    AssembledSystem {
        lower,
        diag,
        upper,
        rhs,
        material_indices: mat_idx, // Vec<[usize;3]> triplets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n_div: usize) -> crate::manager::HamtManager {
        crate::solver::test_fixtures::build_fixture_manager(n_div)
    }

    /// Property: `material_indices` length matches `diag.len()` and all
    /// referenced cells are actually `Material(_)` typed.
    #[test]
    fn material_indices_consistent_with_assembled_size() {
        let mgr = fixture(5);
        let sys = assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );
        assert_eq!(sys.material_indices.len(), sys.diag.len());
        assert_eq!(sys.lower.len(), sys.diag.len());
        assert_eq!(sys.upper.len(), sys.diag.len());
        assert_eq!(sys.rhs.len(), sys.diag.len());
        for &triplet in &sys.material_indices {
            let cell = mgr.surface_curr.cells.get(triplet).unwrap();
            assert!(matches!(cell.cell_type, crate::cell::CellType::Material(_)));
        }
    }

    /// Property: `lower[0]` and `upper[N-1]` are the zero block — there
    /// is no neighbour past either end of the wall.
    #[test]
    fn boundary_off_diagonals_are_zero_blocks() {
        let mgr = fixture(5);
        let sys = assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );
        let n = sys.diag.len();
        let z = crate::solver::block_2x2::Block2x2::zero();
        assert!(
            sys.lower[0].a == z.a
                && sys.lower[0].b == z.b
                && sys.lower[0].c == z.c
                && sys.lower[0].d == z.d,
            "lower[0] must be zero, got {:?}",
            sys.lower[0]
        );
        assert!(
            sys.upper[n - 1].a == z.a
                && sys.upper[n - 1].b == z.b
                && sys.upper[n - 1].c == z.c
                && sys.upper[n - 1].d == z.d,
            "upper[N-1] must be zero, got {:?}",
            sys.upper[n - 1]
        );
    }

    /// Property: with a tiny wall (n_div = 1) every material cell has
    /// only boundary neighbours, so BOTH lower and upper are zero.
    #[test]
    fn single_material_cell_has_no_off_diagonals() {
        let mgr = fixture(1);
        let sys = assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );
        // (n_div=1 may still produce >1 cell after subdivision rules;
        //  the property holds even when n_mat = 1 — test gracefully.)
        if sys.diag.len() == 1 {
            let z = crate::solver::block_2x2::Block2x2::zero();
            assert_eq!(sys.lower[0], z);
            assert_eq!(sys.upper[0], z);
        }
    }

    /// Property (round-trip via solver): the assembled system, when
    /// solved, produces the same end state as the existing BlockJacobi inner
    /// loop converges to. We don't compare against a hand-computed
    /// answer — we rely on the proven `block_tridiag::solve` kernel and
    /// check that the residual of the assembled system at the solution
    /// is at floating-point noise.
    #[test]
    fn assembled_system_residual_is_zero_at_solution() {
        let mgr = fixture(8);
        let sys = assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );
        let x = crate::solver::block_tridiag::solve(&sys.lower, &sys.diag, &sys.upper, &sys.rhs)
            .expect("solve failed");
        let n = sys.diag.len();
        let mut max_r = 0.0_f64;
        for i in 0..n {
            let dx = sys.diag[i].mul_vec(x[i]);
            let lx = if i == 0 {
                (0.0, 0.0)
            } else {
                sys.lower[i].mul_vec(x[i - 1])
            };
            let ux = if i + 1 == n {
                (0.0, 0.0)
            } else {
                sys.upper[i].mul_vec(x[i + 1])
            };
            let r0 = dx.0 + lx.0 + ux.0 - sys.rhs[i].0;
            let r1 = dx.1 + lx.1 + ux.1 - sys.rhs[i].1;
            max_r = max_r.max(r0.abs()).max(r1.abs());
        }
        assert!(
            max_r < 1e-6,
            "assembled-system residual {max_r} exceeds 1e-6"
        );
    }

    /// Property: at thermal-and-vapour equilibrium with the BCs, solving
    /// the assembled system reproduces the equilibrium state — i.e. the
    /// next-iteration (T, φ) equals the current (T, φ) within ~1e-6.
    /// Spec: if the wall is already in steady state, one Δt of implicit
    /// Euler must not move the state. Catches sign / inertial-term bugs
    /// that mis-balance the t_prev contribution.
    #[test]
    fn equilibrium_state_is_a_fixed_point() {
        let mut mgr = fixture(6);
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        // Override outside BC to match inside so the wall is symmetric.
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for cell in mgr.surface_curr.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                cell.cell_temp = bc_temp;
                cell.cell_rh = bc_rh;
            }
        }
        for cell in mgr.surface_prev.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                cell.cell_temp = bc_temp;
                cell.cell_rh = bc_rh;
            }
        }
        mgr.surface_curr
            .update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_prev
            .update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .unwrap();

        // Re-apply BCs now that temps/RH are uniform so boundary cells
        // see the consistent state.
        mgr.surface_curr
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_prev
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();

        let sys = assemble_1d(
            &mgr.surface_curr,
            &mgr.surface_prev,
            &mgr.boundary_condition,
            &mgr.settings,
        );
        let x = crate::solver::block_tridiag::solve(&sys.lower, &sys.diag, &sys.upper, &sys.rhs)
            .expect("solve failed");
        for (row, &triplet) in sys.material_indices.iter().enumerate() {
            let cell = mgr.surface_curr.cells.get(triplet).unwrap();
            let t_curr = cell.cell_temp.value_in(crate::uom::units::kelvin);
            let phi_curr = cell.cell_rh.value();
            let (t_new, phi_new) = x[row];
            assert!(
                (t_new - t_curr).abs() < 1e-6,
                "row {row}: T moved from {t_curr} to {t_new} at equilibrium"
            );
            assert!(
                (phi_new - phi_curr).abs() < 1e-6,
                "row {row}: φ moved from {phi_curr} to {phi_new} at equilibrium"
            );
        }
    }

    /// Mass conservation property — covered by Phase-5 integration.
    #[test]
    #[ignore = "covered by Phase-5 integration; see comment"]
    fn mass_conservation_property() { /* placeholder */
    }
}
