//! The main orchestrator for the HAMT simulation.

mod adaptive;
mod settings;
mod solve;

pub use adaptive::*;
pub use settings::*;

use crate::boundary::BoundaryCondition;
use crate::cell::CellType;
use crate::material::{Construction, Material};
use crate::surface::Surface;
use crate::uom::Quantity;
use crate::uom::dimensions::{Area, Dimensionless, Length, Time};
use crate::uom::units;
use anyhow::{Context, Ok, Result};
use std::mem;
use std::sync::Arc;
use tracing::debug;

/// Manages the state and execution of the HAMT simulation for a surface.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// Materials registry for this self.
    pub materials: Vec<Arc<Material>>,

    /// Construction for the surface.
    pub construction: Arc<Construction>,

    /// The surface's state of last converged time step (t-1).
    pub surface_prev: Surface,
    /// The surface's state of current time step (t) in **last iteration**.
    pub surface_curr: Surface,
    /// The surface's state of current time step (t) in **current iteration**.
    pub surface_next: Surface,

    /// Boundary conditions for the simulation.
    pub boundary_condition: BoundaryCondition,

    /// Simulation-wide settings
    pub settings: HamtSettings,

    /// Converged state from two macro-steps ago (t-2). Used together with
    /// `surface_prev` (t-1) to seed the next macro-step's iteration with a
    /// linear time extrapolation instead of the previous-step state.
    /// `None` until two successful macro-steps have completed.
    pub surface_prev_prev_macro: Option<Surface>,

    /// Macro-timestep duration that just finished. Pairs with
    /// `surface_prev_prev_macro` to scale the time-extrapolation guess to the
    /// upcoming macro-step's Δt. `None` until at least one macro-step has been
    /// solved successfully.
    pub dt_prev_macro: Option<Time<f64>>,
}

impl HamtManager {
    /// Creates a new `HamtManager` and initializes the cell structure for a single surface.
    /// This is a Rust translation of the `InitHeatBalHAMT` subroutine logic.
    pub fn new(
        settings: Option<HamtSettings>,
        initial_boundary: Option<BoundaryCondition>,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
    ) -> Result<Self> {
        let (settings, boundary) =
            Self::prepare_settings_and_boundary(settings, initial_boundary, &construction, &surface_name);

        let surface = Surface::from_construction(
            &settings,
            construction.clone(),
            surface_name,
            surface_area,
            &boundary,
        )?;

        Ok(Self::from_surface(settings, boundary, construction, surface))
    }

    /// 2-D variant of [`HamtManager::new`]: builds the three surfaces via
    /// [`Surface::from_construction_2d`] so the manager owns a `(nx, ny, 1)`
    /// grid (`ny` x-columns stacked along y, each `y_extent / ny` tall) instead
    /// of the 1-D `(N, 1, 1)` stack. Everything else — settings, boundary,
    /// vapor-coefficient propagation, the three-surface (prev/curr/next) layout —
    /// is identical to `new`. With `ny = 1` and `y_extent = sqrt(area)` it is
    /// equivalent to `new`.
    pub fn new_2d(
        settings: Option<HamtSettings>,
        initial_boundary: Option<BoundaryCondition>,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: Area<f64>,
        ny: usize,
        y_extent: Length<f64>,
    ) -> Result<Self> {
        let (settings, boundary) = Self::prepare_settings_and_boundary(
            settings,
            initial_boundary,
            &construction,
            &surface_name,
        );

        let surface = Surface::from_construction_2d(
            &settings,
            construction.clone(),
            surface_name,
            surface_area,
            ny,
            y_extent,
            &boundary,
        )?;

        Ok(Self::from_surface(settings, boundary, construction, surface))
    }

    /// Propagate constant vapor-transfer coefficients from the construction onto
    /// the boundary condition and unwrap the optional settings/boundary. Shared
    /// by [`HamtManager::new`] and [`HamtManager::new_2d`].
    fn prepare_settings_and_boundary(
        settings: Option<HamtSettings>,
        initial_boundary: Option<BoundaryCondition>,
        construction: &Arc<Construction>,
        surface_name: &str,
    ) -> (HamtSettings, BoundaryCondition) {
        let settings = settings.unwrap_or_default();
        let mut boundary = initial_boundary.unwrap_or_default();
        debug!(
            "Initializing HamtManager for surface {}. Boundary conditions: {:#?}",
            surface_name, boundary
        );
        // Propagate constant vapor transfer coefficients from construction definition
        if construction.surface_vapor_coef.is_external_const {
            boundary.vapor_transport_coeff_ext =
                construction.surface_vapor_coef.external_value * units::kg_per_m2_s_pa;
        }
        if construction.surface_vapor_coef.is_internal_const {
            boundary.vapor_transport_coeff_int =
                construction.surface_vapor_coef.internal_value * units::kg_per_m2_s_pa;
        }
        (settings, boundary)
    }

    /// Assemble the manager from a fully-built surface (`surface_curr` =
    /// `surface_prev` = `surface_next` clones of it). Shared by `new`/`new_2d`.
    fn from_surface(
        settings: HamtSettings,
        boundary: BoundaryCondition,
        construction: Arc<Construction>,
        surface: Surface,
    ) -> Self {
        Self {
            settings,
            construction: construction.clone(),
            surface_curr: surface.clone(),
            surface_prev: surface.clone(),
            surface_next: surface,
            boundary_condition: boundary,
            materials: construction.layers.clone(),
            surface_prev_prev_macro: None,
            dt_prev_macro: None,
        }
    }

    /// Linear time-extrapolation warm start for the macro-step initial guess.
    ///
    /// Replaces material-cell `cell_temp` and `cell_rh` in `surface_curr` and
    /// `surface_next` with `value_prev + (value_prev - value_prev_prev) * Δt_ratio`,
    /// where `Δt_ratio = dt_macro_now / dt_prev_macro`. Boundary cells are left
    /// untouched — `apply_boundary_coefficients` overwrites their state at the
    /// start of every iteration. After the extrapolation, dependent state
    /// (vp_sat, d_psat_d_t, volumetric_water, ...) is refreshed via
    /// `update_cell_properties` so it stays consistent with the new (T, φ).
    pub(super) fn apply_warm_start(
        &mut self,
        prev_prev: &Surface,
        dt_macro_now: Time<f64>,
        dt_prev_macro: Time<f64>,
    ) -> Result<()> {
        let ratio: Dimensionless<f64> = dt_macro_now / dt_prev_macro;

        // surface_prev is the t-1 anchor; prev_prev is t-2. surface_curr was
        // just sync'd to surface_prev by tick_timestep, so it carries t-1 too.
        for ((cell_curr, cell_next), (cell_prev, cell_prev_prev)) in self
            .surface_curr
            .cells
            .iter_mut()
            .zip(self.surface_next.cells.iter_mut())
            .zip(self.surface_prev.cells.iter().zip(prev_prev.cells.iter()))
        {
            if !matches!(cell_prev.cell_type, CellType::Material(_)) {
                continue;
            }
            let dt_extrap = (cell_prev.cell_temp - cell_prev_prev.cell_temp) * ratio;
            let drh_extrap = (cell_prev.cell_rh - cell_prev_prev.cell_rh) * ratio;
            let t_init = cell_prev.cell_temp + dt_extrap;
            let rh_init = (cell_prev.cell_rh + drh_extrap).clamp(Quantity(0.0), Quantity(1.01));
            cell_curr.cell_temp = t_init;
            cell_curr.cell_rh = rh_init;
            cell_next.cell_temp = t_init;
            cell_next.cell_rh = rh_init;
        }

        self.surface_curr
            .update_cell_properties(&self.boundary_condition, &self.settings)?;
        self.surface_next
            .update_cell_properties(&self.boundary_condition, &self.settings)?;
        Ok(())
    }

    /// Resets all per-cell relaxation factors to 1.0.
    pub fn reset_relaxation_factors(&mut self) {
        for cell in self.surface_curr.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
        for cell in self.surface_next.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
        for cell in self.surface_prev.cells.iter_mut() {
            cell.relaxation_factor = 1.0;
        }
    }

    /// Executes after the timestep has been calculated.
    pub fn tick_timestep(&mut self) -> Result<()> {
        // Capture the about-to-be-replaced t-1 state as the warm-start anchor
        // for the *next* macro call (where it becomes t-2). Done here, before
        // the swap, because afterwards `surface_prev` carries the just-converged
        // state and the t-1 snapshot is no longer accessible.
        match self.surface_prev_prev_macro {
            Some(ref mut s) => {
                s.sync_state_from(&self.surface_prev)?;
            }
            None => {
                self.surface_prev_prev_macro = Some(self.surface_prev.clone());
            }
        }

        // The converged state is in `surface_curr`. This becomes the previous state for the next timestep.
        mem::swap(&mut self.surface_prev, &mut self.surface_curr);

        // After swapping, the previous becomes the current state.
        self.surface_curr
            .sync_state_from(&self.surface_prev)
            .with_context(|| "Syncing current and previous states")?;
        // `surface_next` is scratch space, so its contents don't matter between steps.
        Ok(())
    }
}

#[cfg(test)]
mod block_tridiag_dispatch_tests {
    use super::*;
    use crate::cell::CellType;
    use crate::uom::units::kelvin;

    /// Property: with `solver_method = BlockTridiagonal` and a benign
    /// macro-step (BC matches initial wall state), `solve_timestep`
    /// returns `Ok(true)` and produces a small-residual converged state.
    /// Catches dispatch bugs (BlockJacobi path getting hit instead) and
    /// outer-loop loop-condition bugs.
    #[test]
    fn block_tridiag_converges_on_benign_step() {
        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        mgr.settings.solver_method = SolverMethod::BlockTridiagonal;
        // Drive both states to BCs so the step is a no-op (benign).
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp = bc_temp;
                    cell.cell_rh = bc_rh;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }

        let result = mgr
            .solve_timestep()
            .expect("solve must not error on benign step");
        assert!(result, "solve must return Ok(true) on benign step");

        // Verify the converged state is at the BC.
        for cell in mgr.surface_curr.cells.iter() {
            if !matches!(cell.cell_type, CellType::Material(_)) {
                continue;
            }
            let dt_k = (cell.cell_temp - bc_temp).value_in(kelvin).abs();
            let drh = (cell.cell_rh - bc_rh).value().abs();
            assert!(
                dt_k < 1e-3 && drh < 1e-3,
                "post-step ΔT={dt_k}, Δφ={drh} — block-tridiag did not stay at BC"
            );
        }
    }

    /// Property: with `coupled_heat_moisture_enabled = false` AND
    /// `solver_method = BlockTridiagonal`, the dispatch falls back to
    /// BlockJacobi (via the warn! path). The solve completes without error.
    #[test]
    fn block_tridiag_falls_back_when_coupling_disabled() {
        let mut mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        mgr.settings.solver_method = SolverMethod::BlockTridiagonal;
        mgr.settings.coupled_heat_moisture_enabled = false;
        // Drive both states to BCs so the step is benign.
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp = bc_temp;
                    cell.cell_rh = bc_rh;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }
        // Should not panic; the dispatcher must fall back to BlockJacobi
        // because BlockTridiagonal requires coupled_heat_moisture_enabled.
        let _ = mgr
            .solve_timestep()
            .expect("fallback to BlockJacobi must not error");
    }
}
