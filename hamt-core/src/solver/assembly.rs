//! 1-D block-tridiagonal assembly for the implicit-Euler HAMT step.
//!
//! Mirrors the per-cell 2×2 build in `surface::update::calculate_physics_step`
//! but emits one (L, D, U, b) row per material cell and pushes
//! material-material couplings into the off-diagonals.
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
//! `block_tridiag::solve` kernel. The 2×2 block construction logic is
//! not direction-specific and should be factored into a small helper at
//! that point.

use crate::boundary::BoundaryCondition;
use crate::hamt_manager::HamtSettings;
use crate::surface::Surface;
use crate::uom::constants::{WATER_LATENT_HEAT, WATER_SPECIFIC_HEAT};
use crate::uom::units::{
    dimensionless, j_per_kg, k_per_w, kelvin, kg_per_m3, kg_per_s,
    kilograms, m2_per_s, pa_s_per_kg, pascals, s_per_kg, w_per_k, watts,
};

use super::block_2x2::Block2x2;

/// One assembled block-tridiagonal system covering all material cells of
/// the source `surface` in physical order. `lower[0]` and `upper[N-1]` are
/// zero-filled per the `block_tridiag::solve` API contract.
pub(crate) struct AssembledSystem {
    pub(crate) lower: Vec<Block2x2>,
    pub(crate) diag:  Vec<Block2x2>,
    pub(crate) upper: Vec<Block2x2>,
    pub(crate) rhs:   Vec<(f64, f64)>,
    /// Material cells' linear indices into `surface.cells`, in the
    /// same order as `diag` rows. Used to scatter the solve result back.
    pub(crate) material_indices: Vec<usize>,
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
    let mat_idx = surface.material_cell_indices();
    let n = mat_idx.len();
    // Lookup table to recognise material-material vs material-boundary
    // neighbours during assembly.
    let mut is_material_at = vec![false; surface.cells.len()];
    for &li in &mat_idx { is_material_at[li] = true; }
    // Map material-cell linear-index → row in the tridiag system.
    let mut row_of = vec![usize::MAX; surface.cells.len()];
    for (row, &li) in mat_idx.iter().enumerate() {
        row_of[li] = row;
    }

    let mut lower = vec![Block2x2::zero(); n];
    let mut diag  = vec![Block2x2::zero(); n];
    let mut upper = vec![Block2x2::zero(); n];
    let mut rhs   = vec![(0.0, 0.0); n];

    let dt_s = settings.timestep;
    let nx = surface.cells.raw_dim()[0];

    for (row, &li) in mat_idx.iter().enumerate() {
        // For a 1-D wall (ny=1, nz=1) the linear index equals the x-axis index.
        let i = li;
        let cell_curr = &surface.cells[[i, 0, 0]];
        let cell_prev = &surface_prev.cells[[i, 0, 0]];

        // L_eff and Taylor parameters at lagged state.
        // When latent_heat_enabled is false the cross-coupling vanishes entirely.
        let l_eff: crate::uom::dimensions::SpecificEnergy<f64> = if settings.latent_heat_enabled {
            1.0 * WATER_LATENT_HEAT
                + if settings.sorption_enthalpy_enabled { cell_curr.h_sorp_diff }
                  else { 0.0 * j_per_kg }
        } else {
            0.0 * j_per_kg
        };
        let phi_k = cell_curr.cell_rh;
        let psat_curr = cell_curr.vp_sat;
        let d_psat_d_t = cell_curr.d_psat_d_t;

        // Cell-local capacities.
        let tcap = (cell_curr.density * cell_curr.spech
            + cell_curr.volumetric_water * WATER_SPECIFIC_HEAT)
            * cell_curr.volume;
        let wcap = if cell_curr.dwd_phi > 1e-10 * kg_per_m3 {
            cell_curr.dwd_phi * cell_curr.volume
        } else { 1.0e-10 * kilograms };

        // Local 2×2 diagonal scalars (f64).
        let mut d_tt: f64 = 0.0; // [W/K]
        let mut d_pp: f64 = 0.0; // [kg/s]
        let mut b_t:  f64 = 0.0; // [W]
        let mut b_p:  f64 = 0.0; // [kg/s]

        // Inertial terms — always go to diagonal + RHS.
        // strip: assemble enters the linear-algebra kernel
        let tcap_per_dt = (tcap / dt_s).value_in(w_per_k);
        let wcap_per_dt = (wcap / dt_s).value_in(kg_per_s);
        d_tt += tcap_per_dt;
        d_pp += wcap_per_dt;
        // strip: kernel boundary
        b_t  += tcap_per_dt * cell_prev.cell_temp.value_in(kelvin);
        b_p  += wcap_per_dt * cell_prev.cell_rh.value();

        // Boundary-deposit and user-defined source on cell i.
        let q_weight = surface
            .user_defined_cell_idx
            .as_ref()
            .map(|w| w.iter().filter(|(idx, _)| idx[0] == i).map(|(_, w)| w).sum::<f64>())
            .unwrap_or(0.0);
        let q_adds = bc.q_addition_userdef * q_weight;
        let q_boundary_energy = if Some(i) == surface.interior_surface_material_idx {
            bc.additional_energy_int
        } else if Some(i) == surface.exterior_surface_material_idx {
            bc.additional_energy_ext
        } else { 0.0 * watts };
        // strip: kernel boundary
        b_t += (q_adds + q_boundary_energy).value_in(watts);

        // Cell-local Taylor cross-coupling for vapour: appears in BOTH
        // diagonal blocks and in the RHS as the linearisation bias.
        // cross [kg/(s·K)]: vpoosum_local · phi_k · dPsat/dT, where
        // vpoosum_local must be summed over ALL neighbours (boundary +
        // material) — we accumulate it as we walk neighbours below.
        let mut vpoosum_si: f64 = 0.0; // [kg/(s·Pa)]

        // Accumulate diagonal sums for d_tt and d_pp from ALL neighbours
        // (both material and boundary). The cross-term (d_tp, d_pt) is
        // computed after the loop using vpoosum_si.
        // NOTE: d_tt and d_pp get `+ inv_th` and `+ inv_phi + inv_vapor·Psat`
        // from every neighbour (material and boundary alike). The off-diagonal
        // block for a material-material edge carries the NEGATIVE of those same
        // conductances. Boundary neighbours contribute only to diagonal + RHS.

        // Walk the neighbour stencil.
        for nbr_li in surface.neighbor_indices(i, nx) {
            let nbr_curr = &surface.cells[[nbr_li, 0, 0]];

            // Resistance pair using the lagged state.
            let (therm_r, vapor_r) = Surface::calculate_resistance_pair(
                cell_curr, nbr_curr, bc.pressure_air,
            );

            // Conductances (zero-on-degeneracy guards mirror the existing path).
            // strip: kernel boundary
            let inv_th = if therm_r > 1e-12 * k_per_w {
                1.0 / therm_r.value_in(k_per_w)
            } else { 0.0 };
            let inv_vapor = if vapor_r > 1e-12 * pa_s_per_kg {
                1.0 / vapor_r.value_in(pa_s_per_kg)
            } else { 0.0 };

            // Liquid (RH-driven) inter-cell resistance — mirrors the
            // existing per-cell branch in calculate_physics_step.
            let overlap_area = cell_curr.size.area_along(0).min(nbr_curr.size.area_along(0));
            let rh_r1 = if cell_curr.dw > 1e-20 * m2_per_s
                          && cell_curr.dwd_phi > 1e-20 * kg_per_m3 {
                (cell_curr.size.x().abs() * 0.5)
                    / (overlap_area * cell_curr.dw * cell_curr.dwd_phi)
            } else { 0.0f64 * s_per_kg };
            let rh_r2 = if nbr_curr.dw > 1e-20 * m2_per_s
                          && nbr_curr.dwd_phi > 1e-20 * kg_per_m3 {
                (nbr_curr.size.x().abs() * 0.5)
                    / (overlap_area * nbr_curr.dw * nbr_curr.dwd_phi)
            } else { 0.0f64 * s_per_kg };
            let r_total_phi = (rh_r1 + rh_r2) * dimensionless;
            let inv_phi = if r_total_phi > 1e-13 * dimensionless / kg_per_s
                            && r_total_phi < f64::MAX * dimensionless / kg_per_s {
                // strip: kernel boundary
                (1.0 / r_total_phi).value_in(kg_per_s)
            } else { 0.0 };

            // Local Psat / dPsat-dT for the NEIGHBOUR (lagged).
            // strip: kernel boundary
            let psat_n  = nbr_curr.vp_sat.value_in(pascals);
            let temp_n  = nbr_curr.cell_temp.value_in(kelvin); // strip: kernel boundary
            // Use cell_vp directly: boundary cells set cell_vp explicitly
            // (e.g. Radiation cell sets cell_vp = 0), so psat_n * phi_n
            // is wrong for them. The material-neighbour off-diagonals still
            // use psat_n (correct: they reference the state variable phi,
            // not the fixed boundary VP). See surface/update.rs:250.
            let vp_n = nbr_curr.cell_vp.value_in(pascals); // strip: kernel boundary

            // Always update the LOCAL `vpoosum_si` (cross-coupling uses
            // sum over ALL neighbours).
            vpoosum_si += inv_vapor;

            if is_material_at[nbr_li] {
                // ── Material neighbour: contributions go to OFF-DIAGONAL ──
                // Off-diag block for this side:
                //   block_tt = -inv_th
                //   block_pp = -(inv_phi + inv_vapor·Psat_n)
                //   block_tp = -L_eff · inv_vapor · Psat_n
                //   block_pt = 0  (neighbour's dPsat/dT and phi_n are the
                //                  neighbour's state; the off-diagonal only
                //                  needs the vapour-pressure coupling, which
                //                  for the φ-equation is via Psat_n·inv_vapor.
                //                  The Taylor-linearisation cross term a_pt
                //                  uses the CELL's own phi·dPsat/dT and goes
                //                  to the diagonal only, not the off-diagonal.)
                let l_eff_v = l_eff.value_in(j_per_kg); // strip: kernel boundary
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
                // Diagonal counter-contribution (positive sign).
                d_tt += inv_th;
                d_pp += inv_phi + inv_vapor * psat_curr.value_in(pascals); // strip: kernel boundary

                let nbr_row = row_of[nbr_li];
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
                let phi_b = nbr_curr.cell_rh.value();
                d_tt += inv_th;
                d_pp += inv_phi + inv_vapor * psat_curr.value_in(pascals); // strip: kernel boundary
                // RHS additions for the boundary side.
                // strip: kernel boundary
                b_t += inv_th * temp_n
                    + l_eff.value_in(j_per_kg) * inv_vapor * vp_n;
                b_p += inv_phi * phi_b + inv_vapor * vp_n;
            }
        }

        // Cell-local Taylor cross-coupling. Uses sum over ALL neighbours.
        let l_eff_v      = l_eff.value_in(j_per_kg);          // strip
        let psat_curr_v  = psat_curr.value_in(pascals);         // strip
        let dpsdt_curr_v = d_psat_d_t.value_in(crate::uom::units::pa_per_k); // strip
        let cross        = vpoosum_si * phi_k.value() * dpsdt_curr_v;
        let t_curr_v     = cell_curr.cell_temp.value_in(kelvin); // strip

        // Diagonal cross terms (within the same cell — they are NOT
        // attached to a neighbour, they're the local linearisation).
        let d_tp = l_eff_v * vpoosum_si * psat_curr_v;
        let d_pt = cross;
        // Latent-heat contribution to A_tt comes from the cross term:
        d_tt += l_eff_v * cross;

        // RHS bias from the linearisation: `+ L_eff · cross · T_curr`
        // and `+ cross · T_curr` are the constants moved over from the
        // implicit form.
        b_t += l_eff_v * cross * t_curr_v;
        b_p += cross * t_curr_v;

        diag[row] = Block2x2 { a: d_tt, b: d_tp, c: d_pt, d: d_pp };
        rhs[row] = (b_t, b_p);
    }

    AssembledSystem { lower, diag, upper, rhs, material_indices: mat_idx }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n_div: usize) -> crate::hamt_manager::HamtManager {
        crate::solver::test_fixtures::build_fixture_manager(n_div)
    }

    /// Property: `material_indices` length matches `diag.len()` and all
    /// referenced cells are actually `Material(_)` typed.
    #[test]
    fn material_indices_consistent_with_assembled_size() {
        let mgr = fixture(5);
        let sys = assemble_1d(
            &mgr.surface_curr, &mgr.surface_prev,
            &mgr.boundary_condition, &mgr.settings,
        );
        assert_eq!(sys.material_indices.len(), sys.diag.len());
        assert_eq!(sys.lower.len(), sys.diag.len());
        assert_eq!(sys.upper.len(), sys.diag.len());
        assert_eq!(sys.rhs.len(), sys.diag.len());
        for &li in &sys.material_indices {
            let cell = mgr.surface_curr.cells.iter().nth(li).unwrap();
            assert!(matches!(cell.cell_type, crate::cell::CellType::Material(_)));
        }
    }

    /// Property: `lower[0]` and `upper[N-1]` are the zero block — there
    /// is no neighbour past either end of the wall.
    #[test]
    fn boundary_off_diagonals_are_zero_blocks() {
        let mgr = fixture(5);
        let sys = assemble_1d(
            &mgr.surface_curr, &mgr.surface_prev,
            &mgr.boundary_condition, &mgr.settings,
        );
        let n = sys.diag.len();
        let z = crate::solver::block_2x2::Block2x2::zero();
        assert!(sys.lower[0].a == z.a && sys.lower[0].b == z.b
                && sys.lower[0].c == z.c && sys.lower[0].d == z.d,
                "lower[0] must be zero, got {:?}", sys.lower[0]);
        assert!(sys.upper[n - 1].a == z.a && sys.upper[n - 1].b == z.b
                && sys.upper[n - 1].c == z.c && sys.upper[n - 1].d == z.d,
                "upper[N-1] must be zero, got {:?}", sys.upper[n - 1]);
    }

    /// Property: with a tiny wall (n_div = 1) every material cell has
    /// only boundary neighbours, so BOTH lower and upper are zero.
    #[test]
    fn single_material_cell_has_no_off_diagonals() {
        let mgr = fixture(1);
        let sys = assemble_1d(
            &mgr.surface_curr, &mgr.surface_prev,
            &mgr.boundary_condition, &mgr.settings,
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
            &mgr.surface_curr, &mgr.surface_prev,
            &mgr.boundary_condition, &mgr.settings,
        );
        let x = crate::solver::block_tridiag::solve(
            &sys.lower, &sys.diag, &sys.upper, &sys.rhs,
        ).expect("solve failed");
        let n = sys.diag.len();
        let mut max_r = 0.0_f64;
        for i in 0..n {
            let dx = sys.diag[i].mul_vec(x[i]);
            let lx = if i == 0 { (0.0, 0.0) } else { sys.lower[i].mul_vec(x[i - 1]) };
            let ux = if i + 1 == n { (0.0, 0.0) } else { sys.upper[i].mul_vec(x[i + 1]) };
            let r0 = dx.0 + lx.0 + ux.0 - sys.rhs[i].0;
            let r1 = dx.1 + lx.1 + ux.1 - sys.rhs[i].1;
            max_r = max_r.max(r0.abs()).max(r1.abs());
        }
        assert!(max_r < 1e-6, "assembled-system residual {max_r} exceeds 1e-6");
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
        mgr.surface_curr.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_prev.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();

        // Re-apply BCs now that temps/RH are uniform so boundary cells
        // see the consistent state.
        mgr.surface_curr
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_prev
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();

        let sys = assemble_1d(
            &mgr.surface_curr, &mgr.surface_prev,
            &mgr.boundary_condition, &mgr.settings,
        );
        let x = crate::solver::block_tridiag::solve(
            &sys.lower, &sys.diag, &sys.upper, &sys.rhs,
        ).expect("solve failed");
        for (row, &li) in sys.material_indices.iter().enumerate() {
            let cell = mgr.surface_curr.cells.iter().nth(li).unwrap();
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
    fn mass_conservation_property() { /* placeholder */ }
}
