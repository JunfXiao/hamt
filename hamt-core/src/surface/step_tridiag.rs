use super::step_jacobi::relax_write_cell;
use super::*;

use anyhow::Result;

/// Direct block-tridiagonal inner step: block-Thomas solve of the implicit-Euler
/// system. See [`Surface::calculate_iteration_block_tridiag`].
pub struct BlockTridiagStep;

impl IterativeStep for BlockTridiagStep {
    fn step(
        &self,
        next: &mut Surface,
        settings: &crate::config::HamtSettings,
        bc: &crate::boundary::BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> anyhow::Result<()> {
        next.calculate_iteration_block_tridiag(settings, bc, surface_prev, surface_curr)
    }
}

impl Surface {
    /// Single outer BlockJacobi step using the direct block-tridiagonal solver.
    ///
    /// `self` is `surface_next` (writable). The method:
    ///   1. Sync `self` from `surface_curr`.
    ///   2. Apply BCs and refresh dependent state.
    ///   3. Assemble the block-tridiag system.
    ///   4. Solve via Thomas.
    ///   5. Scatter the solution into `self`'s material cells (with α
    ///      relaxation per `cell.relaxation_factor`).
    ///
    /// Boundary cells are left as-is (apply_boundary_coefficients
    /// owns them).
    pub fn calculate_iteration_block_tridiag(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        debug_assert!(
            self.cells.raw_dim()[1] == 1 && self.cells.raw_dim()[2] == 1,
            "1-D step invoked on a multi-dimensional grid; use AdiStep"
        );

        // Shared prologue: sync from `surface_curr`, apply BCs, refresh Virtual
        // cells against the current iterate, and update dependent properties.
        // Identical to the ADI path (the 1-D `(N,1,1)` grid is the single-line
        // special case of the [i,j,k]-generalised prologue, so the virtual-cell
        // refresh produces the same result as the former x-only inline block).
        self.prepare_iteration_prologue(settings, bc, surface_curr)?;

        // Assemble from `self`, not `surface_curr`: the BC and Virtual cells on
        // `self` were just refreshed against the current iterate's surface
        // temperature, while the same cells on `surface_curr` still reflect the
        // previous iterate. Material-cell coefficients are identical between the
        // two (sync_state_from copied them and update_cell_properties is idempotent
        // on that data), so this only swaps lagged BC/Virtual coefficients for
        // current-iter ones — restoring near-Newton convergence.
        let system = crate::solver::assembly::assemble_1d(self, surface_prev, bc, settings);
        let solution = crate::solver::block_tridiag::solve(
            &system.lower,
            &system.diag,
            &system.upper,
            &system.rhs,
        )
        // Propagate as a typed `SolverError` (not a stringified anyhow) so
        // `run_outer_loop` can downcast to `SolverError::Singular` and route
        // it into the adaptive substepper's shrink-dt-retry path. `.context()`
        // adds the call-site label without breaking the downcast chain.
        .map_err(|e| anyhow::Error::from(e).context("block-tridiag solve failed"))?;

        // Scatter solution back, applying per-cell α relaxation
        // (consistent with the BlockJacobi write-back behaviour).
        for (row, &triplet) in system.material_indices.iter().enumerate() {
            let (t_new_k, phi_new_raw) = solution[row];
            let curr_temp = surface_curr
                .cells
                .get(triplet)
                .ok_or_else(|| anyhow::anyhow!("invalid cell index {:?}", triplet))?
                .cell_temp;
            let curr_rh = surface_curr
                .cells
                .get(triplet)
                .ok_or_else(|| anyhow::anyhow!("invalid cell index {:?}", triplet))?
                .cell_rh;
            let cell = self
                .cells
                .get_mut(triplet)
                .ok_or_else(|| anyhow::anyhow!("invalid cell index {:?}", triplet))?;

            // re-tag at kernel boundary
            let t_new_typed = t_new_k * crate::uom::units::kelvin;
            let phi_new_typed = phi_new_raw * crate::uom::units::dimensionless;

            relax_write_cell(
                cell,
                t_new_typed,
                phi_new_typed,
                curr_temp,
                curr_rh,
                bc,
                settings,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod block_tridiag_tests {
    fn fixture(n_div: usize) -> crate::manager::HamtManager {
        crate::solver::test_fixtures::build_fixture_manager(n_div)
    }

    /// Property: a single call to `calculate_iteration_block_tridiag`
    /// from a state ALREADY at equilibrium with the BCs is a fixed
    /// point — no material cell moves more than 1e-5 K or 1e-5 RH.
    /// Catches sign errors in the assembly→solve→write-back chain.
    #[test]
    fn one_iteration_at_equilibrium_is_no_op() {
        let mut mgr = fixture(6);
        // Drive cells to BCs (same shape as solver::assembly::tests).
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
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
        // Apply BCs and refresh dependent state in surface_curr/prev so
        // boundary cells (incl. virtual) reflect the symmetric BC.
        mgr.surface_curr
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_prev
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_curr
            .update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        mgr.surface_prev
            .update_cell_properties(&mgr.boundary_condition, &mgr.settings)
            .unwrap();

        // surface_next is scratch; clone from surface_curr.
        let mut surface_next = mgr.surface_curr.clone();
        surface_next
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &mgr.surface_curr,
            )
            .unwrap();

        for (cell_n, cell_c) in surface_next.cells.iter().zip(mgr.surface_curr.cells.iter()) {
            if !matches!(cell_n.cell_type, crate::cell::CellType::Material(_)) {
                continue;
            }
            let dt_k = (cell_n.cell_temp - cell_c.cell_temp)
                .value_in(crate::uom::units::kelvin)
                .abs();
            let drh = (cell_n.cell_rh - cell_c.cell_rh).value().abs();
            assert!(
                dt_k < 1e-5 && drh < 1e-5,
                "ΔT={dt_k}, Δφ={drh} at supposed equilibrium"
            );
        }
    }

    /// Property: a single call away from equilibrium produces a
    /// physically-plausible direction. With cooler BC than wall, every
    /// material cell's T_next ≤ T_curr (within iteration noise).
    /// Verifies the solve goes the right direction; sign-flipped
    /// off-diagonals would make T_next *increase* instead.
    #[test]
    fn one_iteration_moves_in_correct_direction() {
        use crate::boundary::celsius_to_kelvin;
        let mut mgr = fixture(6);
        // Wall at 25 °C / 50 %; air at 15 °C / 30 % (cooler & drier).
        let t_wall = celsius_to_kelvin(25.0);
        let phi_wall = 0.5 * crate::uom::units::dimensionless;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                    cell.cell_temp = t_wall;
                    cell.cell_rh = phi_wall;
                }
            }
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }
        mgr.boundary_condition.temp_inside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.rh_inside_air = 0.3 * crate::uom::units::dimensionless;
        mgr.boundary_condition.rh_outside_air = 0.3 * crate::uom::units::dimensionless;

        let mut surface_next = mgr.surface_curr.clone();
        surface_next
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &mgr.surface_curr,
            )
            .unwrap();

        for cell in surface_next.cells.iter() {
            if !matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                continue;
            }
            let t_new = cell.cell_temp.value_in(crate::uom::units::kelvin);
            assert!(
                t_new <= t_wall.value_in(crate::uom::units::kelvin) + 1e-3,
                "T_next {t_new} K > T_curr 298.15 K — wrong direction"
            );
        }
    }

    /// Property: one orchestrator iteration must read BC and Virtual cell
    /// state from `self` (just refreshed via `apply_boundary_coefficients`
    /// and the virtual-cell update against the current-iterate material
    /// state) — NOT from `surface_curr` (where those cells still reflect
    /// the previous outer iteration). At convergence the two are equal,
    /// but mid-iteration the lag matters: if the assembly reads
    /// `surface_curr`, stale BC/Virtual coefficients leak into the linear
    /// system and convergence drops from near-Newton to slow BlockJacobi.
    ///
    /// Test: build two runs that differ ONLY in the Virtual(Interior)
    /// cell state of `surface_curr` (the Virtual cell sits between
    /// material and the Convection/Radiation BC cells; its `cell_temp`
    /// enters the resistance walk via the assembler's neighbour stencil
    /// over the interior-most material cell). With the fix, the
    /// orchestrator's virtual-cell update step rewrites the Virtual cell
    /// on `self` from current-iterate neighbours, so what was on
    /// `surface_curr` is irrelevant — the two runs must agree to
    /// floating-point noise. Without the fix, the assembler reads the
    /// poked Virtual cell from `surface_curr` and the runs disagree.
    #[test]
    fn iteration_uses_current_iter_state_not_stale_surface_curr() {
        use crate::boundary::celsius_to_kelvin;
        let mut mgr = fixture(6);
        mgr.boundary_condition.has_separate_radiant_int = true;
        // 5 K gap between wall and TMR / air, so a stale BC/Virtual state
        // would visibly disagree with a fresh one.
        let t_wall = celsius_to_kelvin(25.0);
        mgr.boundary_condition.temp_inside_mean_radiant = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_inside_air = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_outside_air = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(20.0);

        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                    cell.cell_temp = t_wall;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }

        // Baseline: surface_curr is consistent with its material-cell state.
        let mut next_fresh = mgr.surface_curr.clone();
        next_fresh
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &mgr.surface_curr,
            )
            .unwrap();

        // Stale variant: poke surface_curr's BC + Virtual interior cells
        // to deliberately-wrong values. The orchestrator must overwrite
        // them on `self` before assembly, so the result must be
        // unchanged.
        let mut surface_curr_stale = mgr.surface_curr.clone();
        let mut poked_rad = false;
        let mut poked_conv = false;
        let mut poked_virt = false;
        let bogus_t = celsius_to_kelvin(80.0);
        for cell in surface_curr_stale.cells.iter_mut() {
            match &cell.cell_type {
                crate::cell::CellType::Boundary(
                    crate::cell::Boundary::Radiation,
                    crate::cell::BoundaryLocation::Interior,
                ) => {
                    cell.htc = cell.htc * 10.0;
                    cell.cell_temp = bogus_t;
                    poked_rad = true;
                }
                crate::cell::CellType::Boundary(
                    crate::cell::Boundary::Convection,
                    crate::cell::BoundaryLocation::Interior,
                ) => {
                    cell.htc = cell.htc * 10.0;
                    cell.cell_temp = bogus_t;
                    poked_conv = true;
                }
                crate::cell::CellType::Virtual(crate::cell::BoundaryLocation::Interior) => {
                    cell.cell_temp = bogus_t;
                    poked_virt = true;
                }
                _ => {}
            }
        }
        assert!(
            poked_rad && poked_conv && poked_virt,
            "fixture missing one of: interior Rad / Conv / Virtual cell"
        );

        let mut next_stale = mgr.surface_curr.clone();
        next_stale
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &surface_curr_stale,
            )
            .unwrap();

        for (cf, cs) in next_fresh.cells.iter().zip(next_stale.cells.iter()) {
            if !matches!(cf.cell_type, crate::cell::CellType::Material(_)) {
                continue;
            }
            let dt = (cf.cell_temp - cs.cell_temp)
                .value_in(crate::uom::units::kelvin)
                .abs();
            let drh = (cf.cell_rh - cs.cell_rh).value().abs();
            assert!(
                dt < 1e-9 && drh < 1e-9,
                "stale surface_curr leaked into orchestrator result: ΔT={dt} K, Δφ={drh}"
            );
        }
    }
}
