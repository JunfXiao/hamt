//! 1-D block-tridiagonal assembly for the implicit-Euler HAMT step.
//!
//! The 1-D wall is the single-line special case of the per-axis ADI sweep:
//! all material cells of a `(N,1,1)` grid form one x-line, every material
//! neighbour is an along-axis (off-diagonal) neighbour, and the boundary /
//! surface nodes are folded into the diagonal and RHS (the massless surface node
//! is eliminated implicitly, not lagged). So [`assemble_1d`] is now a thin
//! delegation to [`crate::solver::adi::assemble_line_sweep`] with `axis = 0`
//! and that single line — there is no longer a separate 1-D assembler body.
//! The shared physics still flows through [`Surface::build_cell_block`]; see
//! `assemble_line_sweep` for the cell-local 2×2 construction and strip
//! discipline.

use crate::boundary::BoundaryCondition;
use crate::config::HamtSettings;
use crate::surface::Surface;

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
///
/// A 1-D `(N,1,1)` wall is exactly one x-line:
/// [`Surface::material_lines_along(0)`] returns a single line containing all
/// material cells in x-order, so this is the single-line special case of the
/// per-axis ADI sweep. We delegate to
/// [`crate::solver::adi::assemble_line_sweep`] with `axis = 0` and that line;
/// on a 1-D wall every material neighbour is along-axis (→ off-diagonal) and
/// every boundary / surface node is folded into the diagonal + RHS, so the result is field-for-field
/// identical to the former inline 1-D assembler (pinned by the equivalence test
/// `crate::solver::adi::tests::line_sweep_axis0_matches_assemble_1d_on_1d_wall`).
pub(crate) fn assemble_1d(
    surface: &Surface,
    surface_prev: &Surface,
    bc: &BoundaryCondition,
    settings: &HamtSettings,
) -> AssembledSystem {
    let lines = surface.material_lines_along(0);
    debug_assert_eq!(lines.len(), 1, "assemble_1d expects a 1-D (single-line) grid");
    crate::solver::adi::assemble_line_sweep(surface, surface_prev, bc, settings, 0, &lines[0])
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

    /// MASS-CONSERVATION PROPERTY of the assembled (linearised) system at ANY
    /// iterate, not only at the fixed point.
    ///
    /// Sum the moisture (φ) rows of `M·x − b` over all cells for an arbitrary
    /// candidate `x` (deliberately NOT the solution). Every internal interface
    /// flux appears once with each sign and cancels exactly iff the two cells
    /// linearise `P_v = φ·P_sat(T)` identically, leaving
    ///
    /// ```text
    ///   Σ_i r_φ,i  =  Σ_i (wcap_i/Δt)·(φ_i − φ_i^n)  −  F_boundary(x)
    /// ```
    ///
    /// with `F_boundary` the inflow through the two massless surface nodes,
    /// eliminated to series conductances (`eliminate_surface_node`) — computed here
    /// independently from `x`. A dropped neighbour `P_sat(T)` term leaves
    /// `−Σ G_v·φ_j^k·P_sat'_j·ΔT_j` (∝ the interface conductance, i.e. ∝ 1/Δx); a
    /// lagged surface node leaves `G_1·(P_v,virt^k − P_v,virt(x))`. Either one
    /// breaks this identity by orders of magnitude on the violently non-converged
    /// state used here.
    #[test]
    fn mass_conservation_property() {
        use crate::boundary::celsius_to_kelvin;
        use crate::cell::BoundaryLocation;
        use crate::solver::adi::eliminate_surface_node;
        use crate::surface::Surface;
        use crate::uom::units::{dimensionless, kelvin, kg_per_s, kilograms, pascals, seconds};

        let mut mgr = fixture(7);
        let bc = mgr.boundary_condition.clone();
        let settings = mgr.settings.clone();
        let mat_idx = mgr.surface_curr.material_cell_indices();

        // A strongly non-converged iterate: T and φ vary wildly from cell to cell.
        for (n, idx) in mat_idx.iter().enumerate() {
            let c = &mut mgr.surface_curr.cells[*idx];
            c.cell_temp = celsius_to_kelvin(20.0 + 6.0 * (0.9 * n as f64).sin());
            c.cell_rh = (0.45 + 0.35 * (1.3 * n as f64 + 1.0).sin()) * dimensionless;
        }
        mgr.surface_curr
            .update_cell_properties(&bc, &settings)
            .unwrap();
        // The state the assembler sees inside the solver: BCs applied and dependent
        // properties refreshed from the iterate.
        let mut it = mgr.surface_curr.clone();
        it.prepare_iteration_prologue(&settings, &bc, &mgr.surface_curr)
            .unwrap();

        let sys = assemble_1d(&it, &mgr.surface_prev, &bc, &settings);
        let n = sys.diag.len();
        assert_eq!(n, mat_idx.len());

        // Arbitrary candidate solution x = (T_i, φ_i), far from the iterate.
        let x: Vec<(f64, f64)> = mat_idx
            .iter()
            .enumerate()
            .map(|(row, idx)| {
                let c = &it.cells[*idx];
                (
                    c.cell_temp.value_in(kelvin) + 0.8 * (2.1 * row as f64).cos(),
                    (c.cell_rh.value() + 0.06 * (1.7 * row as f64).sin()).clamp(0.05, 0.95),
                )
            })
            .collect();

        // Σ of the φ-rows of (M x − b), and Σ |r_φ| as the cancellation scale.
        let mut sum_rp = 0.0_f64;
        let mut scale = 0.0_f64;
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
            let r_p = dx.1 + lx.1 + ux.1 - sys.rhs[i].1;
            sum_rp += r_p;
            scale += r_p.abs();
        }

        // Σ storage increments (wcap/Δt)(φ_i − φ_i^n).
        let dt_s = settings.timestep.value_in(seconds);
        let mut storage = 0.0_f64;
        for (row, idx) in mat_idx.iter().enumerate() {
            let inertia =
                Surface::cell_inertia(idx[0], &it.cells[*idx], &mgr.surface_prev.cells[*idx]);
            let phi_prev = mgr.surface_prev.cells[*idx].cell_rh.value();
            storage += inertia.wcap.value_in(kilograms) / dt_s * (x[row].1 - phi_prev);
        }

        // Boundary inflow through both massless surface nodes at candidate x, from
        // the series-eliminated conductances and the linearised surface-cell P_v.
        let mut f_boundary = 0.0_f64;
        for loc in [BoundaryLocation::Exterior, BoundaryLocation::Interior] {
            let v_idx = *it.virtual_cells(loc.clone()).next().unwrap();
            let m_idx = it.neighbor_toward_center(v_idx, &loc).unwrap();
            let m = &it.cells[m_idx];
            let v = &it.cells[v_idx];
            let r = Surface::calculate_resistance_triple(m, v, bc.pressure_air, 0);
            let g_th = 1.0 / r.therm;
            let g_vap = 1.0 / r.vapor;
            let e = eliminate_surface_node(&it, v_idx, m_idx, g_th, g_vap, &bc);
            let row = mat_idx.iter().position(|i| *i == m_idx).unwrap();
            let (t_x, phi_x) = x[row];
            let p_lin = m.vp_sat.value_in(pascals) * phi_x
                + m.cell_rh.value()
                    * m.d_psat_d_t.value_in(crate::uom::units::pa_per_k)
                    * (t_x - m.cell_temp.value_in(kelvin));
            f_boundary +=
                e.rhs_vap.value_in(kg_per_s) - e.cond_vap.value_in(kg_per_s / pascals) * p_lin;
        }

        let residual = sum_rp - storage + f_boundary;
        let ref_scale = scale + storage.abs() + f_boundary.abs();
        assert!(
            ref_scale > 0.0 && residual.abs() <= 1e-9 * ref_scale,
            "moisture rows are not conservative at a non-converged iterate: \
             Σr_φ={sum_rp:.6e}, storage={storage:.6e}, boundary inflow={f_boundary:.6e}, \
             residual={residual:.3e} (scale {ref_scale:.3e})"
        );
    }
}
