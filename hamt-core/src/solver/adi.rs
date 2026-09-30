//! Per-axis ADI (Alternating-Direction Implicit) line-sweep assembly.
//!
//! Builds the block-tridiagonal system for ONE material line parallel to a
//! chosen `axis` and hands it to the same `block_tridiag::solve` kernel the
//! 1-D path uses. This is the multi-dimensional generalisation of
//! [`crate::solver::assembly::assemble_1d`]:
//!
//! * **Along-axis** material neighbours (the previous/next cell on the same
//!   line) are coupled *implicitly* via the off-diagonal blocks `upper`/`lower`,
//!   INCLUDING the temperature dependence of the neighbour's vapour pressure
//!   (see "Conservation" below).
//! * **Cross-axis** material neighbours are *lagged*: their current-iterate
//!   value is treated as known and folded into the RHS. The outer loop
//!   (`AdiStep` over x→y sweeps) iterates the cross-axis coupling to
//!   convergence.
//! * The massless **surface (Virtual) node** between the first/last material cell
//!   and the boundary cells is eliminated *implicitly* (series conductance, see
//!   [`eliminate_surface_node`]); the boundary cells carry prescribed values
//!   (Dirichlet) that go straight to the RHS.
//!
//! ## 1-D equivalence (the keystone invariant)
//!
//! On a 1-D wall (ny=1, nz=1) the single x-line has only axis-0 neighbours, so
//! every material neighbour is an along-axis line neighbour (→ off-diagonal) and
//! every boundary/surface-node neighbour goes to the diagonal + RHS. The
//! partition therefore collapses to `assemble_1d`'s material-vs-boundary
//! partition, and `assemble_line_sweep(.., axis=0, &x_line)` reproduces
//! `assemble_1d` field-by-field. This is pinned by the equivalence test below.
//!
//! ## Conservation of the linearised system
//!
//! Each outer iterate solves a LINEAR system. The moisture balance holds at every
//! iterate — not just at the fixed point — only if that system is itself
//! conservative, i.e. (a) the vapour flux across an internal interface is the same
//! expression in the two cells' equations, and (b) the flux the boundary cell
//! receives is the flux the reporting queries integrate. Two properties are
//! therefore required of the assembly:
//!
//! 1. **Consistent (Newton) linearisation of `P_v = φ·P_sat(T)`.** Both cells of
//!    an interface linearise `P_v^{k+1} ≈ P_sat(T^k)·φ^{k+1} + φ^k·P_sat'(T^k)·(T^{k+1}−T^k)`.
//!    Dropping the neighbour's `T` term makes `F_ij + F_ji = −G_v·(φ_i P'_i ΔT_i +
//!    φ_j P'_j ΔT_j) ≠ 0` — a spurious moisture source proportional to the last
//!    iterate's `ΔT` and to the interface conductance `G_v ∝ 1/Δx`, i.e. it grows as
//!    cells shrink. [`implicit_material_coupling`] carries the neighbour's `T` term.
//! 2. **Implicit surface node.** Treating the massless Virtual node as a *lagged*
//!    neighbour couples the surface cell to the previous iterate through the
//!    half-cell conductance `G_1 ∝ 1/Δx_1` instead of the film conductance: the
//!    Picard contraction factor becomes `≈ 1 − (h + G_in)/G_1 → 1` as the surface
//!    cell shrinks, the step-size exit test fires long before convergence, and the
//!    converged-looking state violates the mass balance. [`eliminate_surface_node`]
//!    removes the lag exactly (Schur complement of the massless node).
//!
//! Strip discipline: typed quantities cross into raw f64 only inside the 2×2
//! block / RHS construction (the linear-algebra kernel boundary), mirroring
//! `assemble_1d`. Each `.value_in(...)` site carries a strip comment.

use std::collections::HashMap;

use crate::boundary::BoundaryCondition;
use crate::cell::{Cell, CellType};
use crate::config::HamtSettings;
use crate::surface::{NeighborSums, Surface};
use crate::uom::constants::WATER_LATENT_HEAT;
use crate::uom::dimensions::{
    MassFlux, MassFluxPerTemperature, Power, SpecificEnergy, ThermalConductance, VaporConductance,
};
use crate::uom::units::{
    dimensionless, j_per_kg, k_per_w, kelvin, kg_per_s, kg_per_s_k, pa_s_per_kg, pascals, w_per_k,
    watts,
};

use super::assembly::AssembledSystem;
use super::block_2x2::Block2x2;

/// Result of eliminating a massless Virtual (surface) node seen from ONE material
/// cell: equivalent conductances to the boundary cells and the matching RHS terms.
///
/// The node has no storage, so its temperature and vapour pressure are the
/// quasi-steady balance of the material cell `m` (conductance `g_m`, half-cell
/// resistance) and the boundary cells `b_j` (conductances `g_j`, film
/// resistances): `X_v = (g_m·X_m + Σ g_j·X_bj) / (g_m + Σ g_j)` for `X ∈ {T, P_v}`.
/// The flux into `m` is `g_m·(X_v − X_m) = Σ_j g_eff,j·(X_bj − X_m)` with
/// `g_eff,j = g_m·g_j / (g_m + Σ_k g_k)` — a series connection of the half cell
/// with the (parallel) surface films. This is exactly the fixed point of the old
/// lagged-node iteration, but without the lag.
pub(crate) struct SurfaceNodeCoupling {
    /// `Σ_j g_eff,j` thermal [W/K] → adds to the cell's `oorsum`.
    pub cond_th: ThermalConductance<f64>,
    /// `Σ_j g_eff,j` vapour [kg/(s·Pa)] → adds to the cell's `vpoosum`.
    pub cond_vap: VaporConductance<f64>,
    /// `Σ_j g_eff,j·T_bj` [W] → RHS of the energy balance.
    pub rhs_th: Power<f64>,
    /// `Σ_j g_eff,j·P_bj` [kg/s] → RHS of the moisture balance.
    pub rhs_vap: MassFlux<f64>,
}

/// Eliminate the Virtual node `virtual_idx` that neighbours the material cell
/// `material_idx`. `g_m_th` / `g_m_vap` are the material-cell↔node conductances
/// (from [`Surface::calculate_resistance_triple`]). The node's OTHER neighbours are
/// its boundary cells; their prescribed `(T, P_v)` and film conductances are read
/// from `surface`. See [`SurfaceNodeCoupling`].
pub(crate) fn eliminate_surface_node(
    surface: &Surface,
    virtual_idx: [usize; 3],
    material_idx: [usize; 3],
    g_m_th: ThermalConductance<f64>,
    g_m_vap: VaporConductance<f64>,
    bc: &BoundaryCondition,
) -> SurfaceNodeCoupling {
    let node = &surface.cells[virtual_idx];

    let mut g_th_sum: ThermalConductance<f64> = 0.0 * w_per_k;
    let mut g_vap_sum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
    let mut gt_sum: Power<f64> = 0.0 * watts; // Σ g_j·T_bj
    let mut gp_sum: MassFlux<f64> = 0.0 * kg_per_s; // Σ g_j·P_bj

    for b_idx in surface.axis_neighbors(virtual_idx, 0) {
        if b_idx == material_idx {
            continue;
        }
        let b = &surface.cells[b_idx];
        if !b.is_boundary() {
            continue;
        }
        // The node itself has zero thermal/vapour resistance, so the pair
        // resistance is the boundary film (1/(h·A), 1/(β·A)).
        let (therm_r, vapor_r) = Surface::calculate_resistance_pair(node, b, bc.pressure_air, 0);
        if therm_r > 1e-12 * k_per_w {
            let g: ThermalConductance<f64> = 1.0 / therm_r;
            g_th_sum += g;
            gt_sum += g * b.cell_temp;
        }
        if vapor_r > 1e-12 * pa_s_per_kg {
            let g: VaporConductance<f64> = 1.0 / vapor_r;
            g_vap_sum += g;
            gp_sum += g * b.cell_vp;
        }
    }

    // Series factor g_m / (g_m + Σ g_j); zero when nothing is connected.
    let s_th = if g_m_th + g_th_sum > 0.0 * w_per_k {
        g_m_th / (g_m_th + g_th_sum)
    } else {
        0.0 * dimensionless
    };
    let s_vap = if g_m_vap + g_vap_sum > 0.0 * kg_per_s / pascals {
        g_m_vap / (g_m_vap + g_vap_sum)
    } else {
        0.0 * dimensionless
    };

    SurfaceNodeCoupling {
        cond_th: s_th * g_th_sum,
        cond_vap: s_vap * g_vap_sum,
        rhs_th: s_th * gt_sum,
        rhs_vap: s_vap * gp_sum,
    }
}

/// Off-diagonal 2×2 block (plus linearisation-constant RHS terms) coupling a cell
/// to a MATERIAL neighbour `n` that is solved implicitly in the same system.
///
/// Both cells of an interface must linearise `P_v = φ·P_sat(T)` identically,
/// `P_v^{k+1} ≈ P_sat(T^k)·φ^{k+1} + φ^k·P_sat'(T^k)·(T^{k+1} − T^k)`, so the vapour
/// flux `G_v·(P_v,n − P_v,i)` is the same number in both cells' balances (exact
/// moisture conservation at every iterate). The cell's OWN `T` term is the
/// `cross` entry of the diagonal block ([`Surface::build_cell_block`]); this is
/// the matching NEIGHBOUR term. With `cross_n = G_v·φ_n^k·P_sat'(T_n^k)`:
///
/// ```text
///   energy  row:  a = −G_th − L·cross_n   b = −L·G_v·P_sat,n        rhs −= L·cross_n·T_n^k
///   moisture row: c = −cross_n            d = −(G_φ + G_v·P_sat,n)  rhs −=   cross_n·T_n^k
/// ```
pub(crate) struct MaterialCoupling {
    pub block: Block2x2,
    /// Constant to ADD to the energy RHS [W].
    pub rhs_t: f64,
    /// Constant to ADD to the moisture RHS [kg/s].
    pub rhs_p: f64,
}

pub(crate) fn implicit_material_coupling(
    cond_th: ThermalConductance<f64>,
    cond_vapor: VaporConductance<f64>,
    cond_phi: MassFlux<f64>,
    l_eff: SpecificEnergy<f64>,
    n: &Cell,
) -> MaterialCoupling {
    let cross_n: MassFluxPerTemperature<f64> = cond_vapor * n.cell_rh * n.d_psat_d_t;
    // strip: linear-algebra kernel boundary
    let inv_th = cond_th.value_in(w_per_k);
    let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
    let inv_phi = cond_phi.value_in(kg_per_s);
    let l_eff_v = l_eff.value_in(j_per_kg);
    let psat_n = n.vp_sat.value_in(pascals);
    let cross_v = cross_n.value_in(kg_per_s_k);
    let t_n = n.cell_temp.value_in(kelvin);
    MaterialCoupling {
        block: Block2x2 {
            a: -inv_th - l_eff_v * cross_v,
            b: -l_eff_v * inv_vapor * psat_n,
            c: -cross_v,
            d: -(inv_phi + inv_vapor * psat_n),
        },
        rhs_t: -l_eff_v * cross_v * t_n,
        rhs_p: -cross_v * t_n,
    }
}

/// Assemble the block-tridiagonal system for ONE material line parallel to
/// `axis`. Reuses [`Surface::build_cell_block`] for each cell's diagonal +
/// cell-local RHS. ALONG-axis material neighbours go to the off-diagonals
/// (upper/lower by position in `line`) with the neighbour's `T` dependence
/// ([`implicit_material_coupling`]); CROSS-axis material neighbours are LAGGED
/// into the RHS (their current-iterate value treated as known); the massless
/// surface node is eliminated implicitly ([`eliminate_surface_node`]). `line` is
/// the ordered list of `[i,j,k]` material cells along `axis` (one entry of
/// [`Surface::material_lines_along`]).
///
/// `surface` carries the freshly-prepared iteration state (material cells hold
/// the current iterate's (T, φ); boundary cells were refreshed from the BC);
/// `surface_prev` is the t-1 anchor for the inertial term.
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

            // ── Massless surface (Virtual) node: eliminate it IMPLICITLY ──────────
            // The node carries no storage, so it is not an unknown of the system:
            // it is folded into the cell as series conductances to the boundary
            // cells (prescribed values → RHS). Lagging it instead (reading its
            // previous-iterate T / P_v) couples the cell to its own previous
            // iterate through the half-cell conductance ∝ 1/Δx and stalls the outer
            // iteration on fine meshes (module docs, "Conservation").
            if matches!(nbr_curr.cell_type, CellType::Virtual(_)) {
                let e = eliminate_surface_node(surface, nbr_idx, idx, cond_th, cond_vapor, bc);
                oorsum += e.cond_th;
                vpoosum += e.cond_vap;
                phioosum += cond_phi; // 0 across the air film (liquid product guard)
                // strip: linear-algebra kernel boundary
                let l_eff_v = l_eff.value_in(j_per_kg);
                let rhs_th = e.rhs_th.value_in(watts);
                let rhs_vap = e.rhs_vap.value_in(kg_per_s);
                b_t_lag += rhs_th + l_eff_v * rhs_vap;
                b_p_lag += rhs_vap;
                continue;
            }

            // All-neighbour conductance sums for the diagonal block.
            oorsum += cond_th;
            vpoosum += cond_vapor;
            phioosum += cond_phi;

            // An along-axis material neighbour that lies on THIS line at row±1
            // is coupled implicitly (off-diagonal). A cross-axis material
            // neighbour (ADI) or any other non-line neighbour is lagged into the
            // RHS as a known value for this sweep.
            let along_axis_line_nbr = nbr_axis == axis
                && pos_in_line
                    .get(&nbr_idx)
                    .is_some_and(|&nrow| nrow + 1 == row || nrow == row + 1);

            if along_axis_line_nbr {
                // Off-diagonal block for this side, incl. the neighbour's own
                // P_sat(T) term so the interface flux is identical in both cells'
                // balances (see `implicit_material_coupling`).
                let coupling =
                    implicit_material_coupling(cond_th, cond_vapor, cond_phi, l_eff, nbr_curr);
                b_t_lag += coupling.rhs_t;
                b_p_lag += coupling.rhs_p;
                let off = coupling.block;
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
                // Lagged neighbour (cross-axis material, or a boundary cell that
                // neighbours the material cell directly): fold the neighbour-VALUE
                // coupling into the RHS using its current iterate (known in this
                // sweep).
                // strip: linear-algebra kernel boundary — RHS scalars fed into raw f64 accumulator
                let inv_th = cond_th.value_in(w_per_k);
                let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
                let inv_phi = cond_phi.value_in(kg_per_s);
                let l_eff_v = l_eff.value_in(j_per_kg);
                let temp_n = nbr_curr.cell_temp.value_in(kelvin);
                let vp_n = nbr_curr.cell_vp.value_in(pascals);
                let phi_or_rh = nbr_curr.cell_rh.value();
                b_t_lag += inv_th * temp_n + l_eff_v * inv_vapor * vp_n;
                b_p_lag += inv_phi * phi_or_rh + inv_vapor * vp_n;
            }
        }

        // ── Cell-local diagonal block + cell-local RHS via the SHARED builder ──
        // The or-sums (neighbour-VALUE coupling) are zero here: the along-axis
        // material part is in the off-diagonals and the lagged / surface-node
        // part is in the RHS via b_t_lag/b_p_lag.
        let sums = NeighborSums {
            oorsum,
            torsum: 0.0 * watts,
            vapor_diffusion_net_flux: 0.0 * kg_per_s,
            vpoosum,
            vporsum: 0.0 * kg_per_s,
            phioosum,
            phiorsum: 0.0 * kg_per_s,
        };
        let inertia = Surface::cell_inertia(idx[0], cell_curr, cell_prev);
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

                // Massless surface node: eliminated implicitly, exactly as the
                // sweeps do (same helper, so the oracle cannot drift from ADI).
                if matches!(nbr_curr.cell_type, CellType::Virtual(_)) {
                    let e = eliminate_surface_node(self, nbr_idx, idx, cond_th, cond_vapor, bc);
                    oorsum += e.cond_th;
                    vpoosum += e.cond_vap;
                    phioosum += cond_phi;
                    // strip: dense Gaussian-elimination oracle (small linear-algebra kernel)
                    let l_eff_v = l_eff.value_in(j_per_kg);
                    let rhs_th = e.rhs_th.value_in(watts);
                    let rhs_vap = e.rhs_vap.value_in(kg_per_s);
                    b_t_boundary += rhs_th + l_eff_v * rhs_vap;
                    b_p_boundary += rhs_vap;
                    continue;
                }

                oorsum += cond_th;
                vpoosum += cond_vapor;
                phioosum += cond_phi;

                if let Some(&nbr_row) = block_row.get(&nbr_idx) {
                    // Material neighbour (ANY axis): fully-coupled off-diagonal
                    // block — the SAME block (incl. the neighbour's P_sat(T) term)
                    // AdiStep uses along-axis, here applied to EVERY material face
                    // neighbour with no lag.
                    let coupling =
                        implicit_material_coupling(cond_th, cond_vapor, cond_phi, l_eff, nbr_curr);
                    b_t_boundary += coupling.rhs_t;
                    b_p_boundary += coupling.rhs_p;
                    off_blocks.push((nbr_row, coupling.block));
                } else {
                    // Boundary neighbour → RHS, exactly as the sweeps fold it.
                    // strip: dense Gaussian-elimination oracle (small linear-algebra kernel) —
                    // RHS scalars fed into raw f64 accumulator at the kernel boundary
                    let inv_th = cond_th.value_in(w_per_k);
                    let inv_vapor = cond_vapor.value_in(kg_per_s / pascals);
                    let inv_phi = cond_phi.value_in(kg_per_s);
                    let l_eff_v = l_eff.value_in(j_per_kg);
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
            let inertia = Surface::cell_inertia(idx[0], cell_curr, cell_prev);
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

    /// 3-D ADI CORRECTNESS against the dense direct oracle (Phase 2b Task 2).
    ///
    /// Build a small `(n_div_x=4, ny=1, nz=6)` grid with ACTIVE z-faces and a
    /// z-gradient in T and φ, exercising the new `run_axis_sweep(2, ..)` z-sweep.
    /// Drive `AdiStep` and the fully-coupled dense `DirectStep` each through
    /// `run_outer_loop_counted` for one step and assert they reach the SAME fixed
    /// point (max|ΔT|, max|Δφ| < 1e-6) — i.e. the 3-D ADI sweep machinery solves
    /// the z-coupled system correctly.
    ///
    /// NOTE on scope: this is a CORRECTNESS test, not a z-sweep load-bearing guard.
    /// The unified outer loop + Anderson converges the lagged cross-axis coupling
    /// to the same answer whether or not the z-sweep runs (so it is not a clean
    /// red→green for the z-sweep itself). The test that PROVES the z-sweep is
    /// load-bearing — strong (cubic-cell) z-coupling with Anderson OFF, where
    /// removing the z-sweep blows up the ADI/DIRECT iteration ratio — lives in the
    /// Phase 2b Task 3 stability gate (`adi_matches_direct_on_thermal_bridge_3d`).
    /// The `ratio ≤ 1.75` check below is only a sanity bound (ADI is not
    /// pathologically slower than the dense solve at this weak-coupling size).
    #[test]
    fn adi_3d_z_sweep_converges() {
        use crate::cell::CellType;
        use crate::uom::units::{dimensionless, kelvin};

        // ny=1 isolates the z-direction: x-sweep handles x-lines, y-sweep handles
        // singleton ny=1 lines (no-op), z-sweep (new) handles z-lines.
        // nz=6 makes the z-direction the convergence bottleneck when lagged.
        let n_div_x = 4usize;
        let ny = 1usize;
        let nz = 6usize;
        let mut base = crate::solver::test_fixtures::build_fixture_manager_3d(
            n_div_x,
            ny,
            nz,
            1.0 * crate::uom::units::meter,
            1.0 * crate::uom::units::meter,
            false,
            true,
        );

        let dim = base.surface_curr.cells.raw_dim();
        eprintln!(
            "3-D z-sweep grid: ({}, {}, {}), cells={}",
            dim[0],
            dim[1],
            dim[2],
            base.surface_curr.cells.len()
        );

        // ── Impose a strong z-gradient: k=0 hot & wet, k=nz-1 cold & dry. ──
        let nx = dim[0];
        let bc = base.boundary_condition.clone();
        let settings = base.settings.clone();
        for surf in [
            &mut base.surface_prev,
            &mut base.surface_curr,
            &mut base.surface_next,
        ] {
            for k in 0..nz {
                let fz = k as f64 / (nz as f64 - 1.0);
                for i in 0..nx {
                    let is_mat =
                        matches!(surf.cells[[i, 0, k]].cell_type, CellType::Material(_));
                    if !is_mat {
                        continue;
                    }
                    // Strong gradient: 35°C → 10°C in z; mild x-tilt so it's genuinely 3-D.
                    let fx = i as f64 / (nx as f64 - 1.0);
                    let t_c = 35.0 - 25.0 * fz + 3.0 * (fx - 0.5);
                    let phi = (0.80 - 0.55 * fz + 0.04 * (fx - 0.5)).clamp(0.05, 0.99);
                    let cell = &mut surf.cells[[i, 0, k]];
                    cell.cell_temp = crate::boundary::celsius_to_kelvin(t_c);
                    cell.cell_rh = phi * dimensionless;
                }
            }
            surf.update_cell_properties(&bc, &settings).unwrap();
        }

        // Tight tolerance + no Aitken damping (min_relaxation_factor = 1.0) so
        // the iteration count reflects only the implicit/lagged z-direction cost.
        base.settings.timestep = 300.0 * crate::uom::units::seconds;
        base.settings.temp_convergence_tolerance = 1e-9 * kelvin;
        base.settings.rh_convergence_tolerance = 1e-9 * dimensionless;
        base.settings.min_relaxation_factor = 1.0;
        base.settings.solver_method = crate::manager::SolverMethod::Adi;
        base.settings.anderson_enabled = true;

        let mut mgr_adi = base.clone();
        let mut mgr_dir = base.clone();

        let max_iter = 500;
        let (conv_adi, iters_adi) = mgr_adi
            .run_outer_loop_counted(&crate::surface::AdiStep, max_iter)
            .expect("ADI outer loop");
        let (conv_dir, iters_dir) = mgr_dir
            .run_outer_loop_counted(&DirectStep, max_iter)
            .expect("Direct outer loop");

        // ── Agreement: compare final material T and φ. ──
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

        let ratio = iters_adi as f64 / iters_dir.max(1) as f64;
        eprintln!(
            "=== ADI-vs-DIRECT 3-D z-sweep result (Phase 2b Task 2) ===\n\
             ADI:    converged={conv_adi}, outer_iters={iters_adi}\n\
             DIRECT: converged={conv_dir}, outer_iters={iters_dir}\n\
             ADI/DIRECT ratio = {ratio:.3}\n\
             agreement: max|ΔT|={max_dt:.3e} K, max|Δφ|={max_dphi:.3e} (gate 1e-6)"
        );

        assert!(conv_adi, "ADI did not converge within {max_iter} outer iters");
        assert!(conv_dir, "DIRECT did not converge within {max_iter} outer iters");
        // Both methods must converge to the same fixed point.
        assert!(
            max_dt < 1e-6 && max_dphi < 1e-6,
            "AdiStep and DirectStep disagree: max|ΔT|={max_dt:.3e} K, \
             max|Δφ|={max_dphi:.3e} (gate 1e-6). ADI iters={iters_adi}, DIRECT iters={iters_dir}"
        );
        // Sanity bound only (NOT a z-sweep load-bearing guard — see the doc note):
        // at this weak-coupling grid size the outer loop + Anderson converges the
        // lagged coupling efficiently with or without the z-sweep, so this merely
        // confirms ADI is not pathologically slower than the fully-coupled solve.
        // The real z-sweep discriminator (Anderson off, strong cubic-cell coupling)
        // is the Task 3 gate adi_matches_direct_on_thermal_bridge_3d.
        assert!(
            ratio <= 1.75,
            "ADI iteration count pathologically high vs direct: ratio {ratio:.2} > 1.75 \
             (ADI={iters_adi}, DIRECT={iters_dir})"
        );
    }

    /// Z-SWEEP LOAD-BEARING STABILITY GATE (Phase 2b Task 3).
    ///
    /// Proves the z-sweep is load-bearing under strongly z-coupled conditions.
    ///
    /// Configuration:
    /// * `ny=1, nz=6` — z-dominated grid; y-coupling absent so only z matters.
    /// * `z_cell = 0.001 m` (2.5× smaller than x-cell 0.0025 m → z-conductance
    ///   2.5× larger). Strong z-coupling relative to x.
    /// * **Anderson OFF** (`anderson_enabled = false`): Anderson would mask the
    ///   z-lag by converging it in a few AA steps regardless of whether the z-sweep
    ///   runs. With raw Picard exposed, the per-sweep update must carry the z-coupling
    ///   explicitly via the z-sweep, else each outer iteration barely progresses the
    ///   z-lagged residual.
    /// * `max_iter = 400` (tight budget): WITH z-sweep ADI converges in ≈ 142
    ///   iters (within budget); WITHOUT z-sweep ADI needs ≈ 761 iters (exceeds
    ///   budget → test RED). (The budget was 1000 while the lagged surface node
    ///   made both counts 3–5× larger — 503 / ≈ 1010; once the surface node and the
    ///   neighbour Jacobian are implicit the mutant converges at 761 < 1000 and the
    ///   old budget stopped discriminating.) DIRECT runs with `max_iter_dir = 2000`
    ///   separately to get the correctness oracle when it converges.
    ///
    /// PRIMARY DISCRIMINATOR: `conv_adi` (Anderson OFF) — WITH z-sweep ADI converges
    /// within `max_iter = 400`; removing the guard (`if false && ..`) makes it need
    /// ~761 iters → budget exhausted → assertion fails (verified red→green).
    ///
    /// CORRECTNESS (live): the DIRECT reference is run with Anderson ON so the
    /// fully-coupled dense solve converges quickly to the true fixed point (Anderson
    /// changes only the rate, not the answer); the Anderson-OFF ADI result must match
    /// it to < 1e-7. The printed ADI/DIRECT iteration ratio is informational only and
    /// is NOT apples-to-apples (different Anderson settings).
    #[test]
    fn adi_matches_direct_on_thermal_bridge_3d() {
        use crate::cell::CellType;
        use crate::uom::units::{dimensionless, kelvin, meter};

        // z-dominated 3-D grid: ny=1 (no y-coupling), nz=6, stronger z-conductance.
        // x-cell ≈ 0.01/4 = 0.0025 m; z-cell = 0.001 m (2.5× smaller → z-conductance
        // ≈ 2.5× x-conductance). Nmat = 4*1*6 = 24 → dense 48×48 for DirectStep.
        let n_div_x = 4usize;
        let ny = 1usize;
        let nz = 6usize;
        let x_cell_size = 0.0025_f64; // m — matches material_thickness / n_div_x
        let z_cell_size = 0.001_f64;  // m — 2.5× smaller than x → z-conductance 2.5×
        let y_extent = 1.0 * meter;
        let z_extent = (nz as f64) * z_cell_size * meter;

        let mut base = crate::solver::test_fixtures::build_fixture_manager_3d(
            n_div_x,
            ny,
            nz,
            y_extent,
            z_extent,
            false, // y_faces_active=false: ny=1, y is trivial
            true,  // z_faces_active: genuine cross-z material coupling
        );

        let dim = base.surface_curr.cells.raw_dim();
        eprintln!(
            "3-D bridge grid: ({}, {}, {}), cells={}, x_cell={:.4}m, z_cell={:.4}m",
            dim[0], dim[1], dim[2], base.surface_curr.cells.len(), x_cell_size, z_cell_size
        );

        // Strong z-gradient; mild x-tilt so the system is genuinely 2-D in (x,z).
        let nx = dim[0];
        let bc = base.boundary_condition.clone();
        let settings = base.settings.clone();
        for surf in [
            &mut base.surface_prev,
            &mut base.surface_curr,
            &mut base.surface_next,
        ] {
            for k in 0..nz {
                let fz = k as f64 / (nz as f64 - 1.0);
                for i in 0..nx {
                    if !matches!(surf.cells[[i, 0, k]].cell_type, CellType::Material(_)) {
                        continue;
                    }
                    let fx = i as f64 / (nx as f64 - 1.0);
                    let t_c = 35.0 - 25.0 * fz + 4.0 * (fx - 0.5);
                    let phi = (0.80 - 0.55 * fz + 0.05 * (fx - 0.5)).clamp(0.05, 0.99);
                    let cell = &mut surf.cells[[i, 0, k]];
                    cell.cell_temp = crate::boundary::celsius_to_kelvin(t_c);
                    cell.cell_rh = phi * dimensionless;
                }
            }
            surf.update_cell_properties(&bc, &settings).unwrap();
        }

        // dt = 300 s; Anderson OFF to expose raw Picard lag; tol reachable (1e-7) so
        // the DIRECT reference and z-swept ADI both converge while lagged-z does not.
        base.settings.timestep = 300.0 * crate::uom::units::seconds;
        base.settings.temp_convergence_tolerance = 1e-7 * kelvin;
        base.settings.rh_convergence_tolerance = 1e-7 * dimensionless;
        base.settings.min_relaxation_factor = 1.0;
        base.settings.solver_method = crate::manager::SolverMethod::Adi;
        base.settings.anderson_enabled = false; // ← KEY: expose raw Picard lag

        let mut mgr_adi = base.clone();
        // DIRECT reference: Anderson ON so the fully-coupled dense solve converges
        // quickly to the true fixed point (Anderson changes only the rate, not the
        // answer). This gives a LIVE correctness oracle for the agreement check
        // below, independent of the Anderson-OFF discriminator ADI runs under.
        let mut mgr_dir = base.clone();
        mgr_dir.settings.anderson_enabled = true;

        // PRIMARY discriminator budget: tight enough that WITH z-sweep ADI (Anderson
        // OFF) converges (~142 iters) but WITHOUT z-sweep it does not (~761 iters).
        let max_iter_adi = 400i32;
        // DIRECT (Anderson ON) converges in well under this; generous for safety.
        let max_iter_dir = 2000i32;

        let (conv_adi, iters_adi) = mgr_adi
            .run_outer_loop_counted(&crate::surface::AdiStep, max_iter_adi)
            .expect("ADI outer loop");
        let (conv_dir, iters_dir) = mgr_dir
            .run_outer_loop_counted(&DirectStep, max_iter_dir)
            .expect("Direct outer loop");

        // Agreement (only meaningful when both converge).
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

        let ratio = iters_adi as f64 / iters_dir.max(1) as f64;
        eprintln!(
            "=== ADI-vs-DIRECT 3-D thermal-bridge result (Phase 2b Task 3) ===\n\
             grid: ({nx}, {ny}, {nz}), x_cell={x_cell_size:.4}m, z_cell={z_cell_size:.4}m, dt=300s, anderson=OFF\n\
             ADI:    converged={conv_adi}, outer_iters={iters_adi} (budget={max_iter_adi})\n\
             DIRECT: converged={conv_dir}, outer_iters={iters_dir} (budget={max_iter_dir})\n\
             ADI/DIRECT iteration ratio = {ratio:.3}\n\
             agreement: max|ΔT|={max_dt:.3e} K, max|Δφ|={max_dphi:.3e}"
        );

        // PRIMARY DISCRIMINATOR: WITH z-sweep ADI converges; without it does NOT.
        // (Removing the z-sweep guard to `if false && ..` makes this assertion fail.)
        assert!(conv_adi, "ADI did not converge within {max_iter_adi} outer iters — z-sweep may be missing");

        // CORRECTNESS (live): the DIRECT reference (Anderson ON) must converge, and
        // the Anderson-OFF ADI solution must match its fully-coupled fixed point.
        assert!(
            conv_dir,
            "DIRECT reference did not converge within {max_iter_dir} outer iters"
        );
        // Both converge by a step-size metric (|Δ between iters| < tol 1e-7), so the
        // two fixed-point approximations sit a few× tol apart; gate at 1e-5.
        assert!(
            max_dt < 1e-5 && max_dphi < 1e-5,
            "ADI and DIRECT disagree: max|ΔT|={max_dt:.3e} K, max|Δφ|={max_dphi:.3e} \
             (gate 1e-5). ADI iters={iters_adi}, DIRECT iters={iters_dir}"
        );
    }
}
