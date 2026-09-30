//! Adaptive sub-stepping state and flux accumulator.

use crate::cell::{BoundaryLocation, CellType};
use crate::error;
use crate::surface::Surface;
use crate::uom::dimensions::{
    ArealEnergyDensity, ArealMassDensity, Dimensionless, EnergyFluxDensity, MassFluxDensity, Time,
};
use crate::uom::units::{
    dimensionless, hours, j_per_m2, kelvin, kg_per_m2, kg_per_m2_s, seconds, w_per_m2,
};
use crate::uom::{Quantity, units};
use anyhow::Result;
use std::mem;

use super::HamtManager;
use super::settings::{PiControllerSettings, RhRate, TempRate};

/// Accumulates time-weighted boundary fluxes across adaptive substeps.
/// After all substeps, dividing each `*_dt` field by `total_time` yields the
/// time-averaged flux density over the macro-timestep.
///
/// Each `*_dt` is the END-POINT (right-Riemann) sum `Σ F_end·dt` over the accepted
/// substeps — exactly what the backward-Euler update stores in the wall (see
/// `advance_accepted_substep`), so `∫ m_int dt · A` equals the change of the water
/// stored in the wall.
#[derive(Debug, Clone)]
pub struct SubstepFluxAccumulator {
    /// ∫ q_int dt  [J/m²] — the TOTAL interior film heat flux (convective + longwave,
    /// [`Surface::heat_flux`]), not the convective film alone.
    pub q_int_dt: ArealEnergyDensity<f64>,
    /// ∫ q_ext dt  [J/m²] — the exterior film heat flux (the single convective film).
    pub q_ext_dt: ArealEnergyDensity<f64>,
    /// ∫ m_int dt  [kg/m²]
    pub m_int_dt: ArealMassDensity<f64>,
    /// ∫ m_ext dt  [kg/m²]
    pub m_ext_dt: ArealMassDensity<f64>,
    /// Total substep time accumulated
    pub total_time: Time<f64>,
}

impl Default for SubstepFluxAccumulator {
    fn default() -> Self {
        Self {
            q_int_dt: 0.0 * j_per_m2,
            q_ext_dt: 0.0 * j_per_m2,
            m_int_dt: 0.0 * kg_per_m2,
            m_ext_dt: 0.0 * kg_per_m2,
            total_time: 0.0 * seconds,
        }
    }
}

impl SubstepFluxAccumulator {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Time-averaged TOTAL interior film heat flux [W/m²] (convective film to the air +
    /// longwave film to TMR; `qWG1` excluded). Type 395 output 8 is this × area.
    pub fn avg_q_int(&self) -> EnergyFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.q_int_dt / self.total_time
        } else {
            0.0 * w_per_m2
        }
    }

    /// Time-averaged exterior film heat flux [W/m²] (`qWG2` excluded). Type 395 output 9 is
    /// this × area.
    pub fn avg_q_ext(&self) -> EnergyFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.q_ext_dt / self.total_time
        } else {
            0.0 * w_per_m2
        }
    }

    /// Time-averaged interior moisture flux [kg/(m²·s)].
    pub fn avg_m_int(&self) -> MassFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.m_int_dt / self.total_time
        } else {
            0.0 * kg_per_m2_s
        }
    }

    /// Time-averaged exterior moisture flux [kg/(m²·s)].
    pub fn avg_m_ext(&self) -> MassFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.m_ext_dt / self.total_time
        } else {
            0.0 * kg_per_m2_s
        }
    }
}

pub struct AdaptiveState<T> {
    pub substep_time: Time<f64>,
    pub current_time: Time<f64>,

    /// The current timestep size proposed by the PI controller.
    pub proposed_dt: Option<Time<f64>>,

    /// The local truncation error (LTE) from the previous accepted step, used for the Integral term.
    pub prev_lte: Option<f64>,

    // Temperature per Time
    pub rate_t: Option<ndarray::Array3<TempRate<f64>>>,
    pub rate_rh: Option<ndarray::Array3<RhRate<f64>>>,
    pub data: T,
    pub prev_surface_state: Option<Surface>,
    pub predicted_surface_state: Option<Surface>,
    pub flux_accumulator: SubstepFluxAccumulator,
}

impl<T> Default for AdaptiveState<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            substep_time: 0.0 * units::seconds,
            current_time: 0.0 * units::seconds,
            data: T::default(),
            proposed_dt: None,
            prev_lte: None,
            rate_t: None,
            rate_rh: None,
            prev_surface_state: None,
            predicted_surface_state: None,
            flux_accumulator: SubstepFluxAccumulator::default(),
        }
    }
}

impl HamtManager {
    /// Solves for the current timestep using an adaptive stepping strategy based on a PI controller.
    ///
    /// Thin driver: orchestrates the per-substep PREDICT → solve+LTE → ACCEPT/REJECT
    /// loop, delegating each phase to a named helper. Behaviour is identical to the
    /// previous monolithic implementation; the `self.settings.timestep` set/restore
    /// dt-flow is preserved unchanged.
    pub fn solve_timestep_adaptive<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        mut callback_state_change: Option<Box<dyn FnMut(&AdaptiveState<T>) -> Result<()>>>,
    ) -> Result<()> {
        let (macro_dt, mut current_dt) = self.init_adaptive_substepping(adaptive_state)?;

        // Use adaptive_state.substep_time as the accumulator condition
        while adaptive_state.substep_time < macro_dt {
            let time_remaining = macro_dt - adaptive_state.substep_time;
            let is_boundary_step = current_dt >= time_remaining;

            // Boundary Clamping
            let dt_step: Time<f64> = current_dt.min(time_remaining);

            // PREDICT: linearly extrapolate the predicted surface and seed surface_curr.
            self.extrapolate_predicted_surface(adaptive_state, dt_step)?;

            self.settings.timestep = dt_step;
            let converged = self.solve_timestep()?;

            if !converged {
                // Failed to converge, rollback to the previous state and reduce the timestep more aggressively for the next try.
                current_dt *= 0.25;

                if current_dt.value_in(seconds) < 1e-3 {
                    return Err(error::Error::Other(format!(
                        "Solver irrecoverably stalled at {:.2} hours with dt < 1ms",
                        adaptive_state.current_time.value_in(hours)
                    ))
                    .into());
                }

                // Clean up any partial updates to avoid contaminating the next iteration
                self.surface_curr.sync_state_from(&self.surface_prev)?;
                self.surface_next.sync_state_from(&self.surface_prev)?;
                continue;
            }

            // solve + LTE: compute the per-cell local truncation error for this trial dt.
            let max_lte = self.compute_substep_lte(
                adaptive_state
                    .predicted_surface_state
                    .as_ref()
                    .expect("Predicted Surface shouldn't be None"),
            );

            // Accept/Reject Logic
            if max_lte <= 1.0 {
                // ACCEPT: commit the substep (rates, times, fluxes, surface swap).
                self.advance_accepted_substep(adaptive_state, dt_step)?;

                if let Some(ref mut cb) = callback_state_change {
                    cb(adaptive_state)?;
                }

                // PI Controller
                if !is_boundary_step {
                    current_dt = Self::pi_predict_dt(
                        &self.settings.pi,
                        adaptive_state.prev_lte,
                        max_lte,
                        dt_step,
                    );
                    adaptive_state.prev_lte = Some(max_lte);
                }
            } else {
                // REJECT: shrink dt and restore the pre-substep state.
                current_dt = self.shrink_rejected_substep(adaptive_state.prev_lte, max_lte, dt_step)?;
            }
        }

        self.finalize_macro_step(adaptive_state, current_dt, macro_dt)?;

        Ok(())
    }

    /// Macro-level warm start plus adaptive-state / accumulator initialisation.
    ///
    /// Performs the time-extrapolation warm start, syncs `prev_surface_state`,
    /// resets the flux accumulator, cold-starts the PI extrapolation rate arrays
    /// and the predicted surface, and returns `(macro_dt, current_dt)` — the macro
    /// timestep and the optimistic starting Δt for the first substep.
    fn init_adaptive_substepping<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
    ) -> Result<(Time<f64>, Time<f64>)> {
        // --- 1. Macro-level Warm Start ---
        // Uses previous macro-steps to provide a better initial guess for the non-linear solver.
        if let (Some(prev_prev), Some(dt_prev)) =
            (self.surface_prev_prev_macro.clone(), self.dt_prev_macro)
        {
            self.apply_warm_start(&prev_prev, self.settings.timestep, dt_prev)?;
        }

        if let Some(ref mut prev_surface_state) = adaptive_state.prev_surface_state {
            prev_surface_state.sync_state_from(&self.surface_prev)?;
        } else {
            adaptive_state.prev_surface_state = Some(self.surface_prev.clone());
        }

        // --- 2. State & Accumulator Initialization ---
        adaptive_state.flux_accumulator.reset();
        let macro_dt = self.settings.timestep;
        adaptive_state.substep_time = 0.0 * hours;

        // Optimistic Start with the full macro Δt. The PI controller will adjust down if it detects instability.
        let current_dt: Time<f64> = adaptive_state.proposed_dt.unwrap_or(macro_dt);

        // Historical change rates for PI linear extrapolation. Cold start with 0.
        if adaptive_state.rate_t.is_none() {
            adaptive_state.rate_t = Some(ndarray::Array3::<Quantity<f64, _>>::from_shape_fn(
                self.surface_curr.cells.dim(),
                |_| 0.0 * kelvin / hours,
            ));
        }
        if adaptive_state.rate_rh.is_none() {
            adaptive_state.rate_rh = Some(ndarray::Array3::<Quantity<f64, _>>::from_shape_fn(
                self.surface_curr.cells.dim(),
                |_| 0.0 * (dimensionless / hours),
            ));
        }

        if adaptive_state.predicted_surface_state.is_none() {
            adaptive_state.predicted_surface_state = Some(self.surface_prev.clone());
        }

        Ok((macro_dt, current_dt))
    }

    /// PREDICT: build the predicted surface by linearly extrapolating from the
    /// previous converged state with the stored per-cell change rates over `dt_step`,
    /// then seed `surface_curr` with it as the predictor warm-start initial guess.
    fn extrapolate_predicted_surface<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        dt_step: Time<f64>,
    ) -> Result<()> {
        let rate_t = adaptive_state.rate_t.as_ref().unwrap();
        let rate_rh = adaptive_state.rate_rh.as_ref().unwrap();

        let predicted_surface = adaptive_state
            .predicted_surface_state
            .as_mut()
            .expect("Predicted Surface shouldn't be None");

        predicted_surface.sync_state_from(&self.surface_prev)?;
        for (idx, cell) in predicted_surface.cells.indexed_iter_mut() {
            if matches!(cell.cell_type, CellType::Material(_)) {
                cell.cell_temp += rate_t[idx] * dt_step;
                cell.cell_rh += rate_rh[idx] * dt_step;
                cell.calc_dependent_state(&self.boundary_condition, &self.settings);
            }
        }

        // Predictor Warm-Start
        // Use the predicted surface as the initial guess for the current step.
        //This can significantly reduce iterations for small steps, especially in the early phase of a macro-step where the state changes rapidly.
        self.surface_curr.sync_state_from(predicted_surface)?;

        Ok(())
    }

    /// LTE: compare the just-converged `surface_curr` against `predicted_surface`
    /// to estimate the local truncation error, normalised by the decoupled
    /// integration tolerances. Returns the per-cell maximum (init floor `1e-12`).
    fn compute_substep_lte(&self, predicted_surface: &Surface) -> f64 {
        // Decouple Integration Tolerance
        // Instead of using the global convergence tolerances, we define separate integration tolerances for temperature and relative humidity that are more lenient.
        // This allows the PI controller to make informed decisions based on the physical changes rather than strict solver convergence, which can be too tight for adaptive stepping.
        let integ_tol_t: crate::uom::dimensions::Temperature<f64> = self
            .settings
            .temp_convergence_tolerance
            .max(self.settings.pi.integ_tol_t_floor);
        let integ_tol_rh: Dimensionless<f64> = self
            .settings
            .rh_convergence_tolerance
            .max(self.settings.pi.integ_tol_rh_floor);

        let mut max_lte: f64 = 1e-12;
        for (cell_pred, cell_curr) in predicted_surface
            .cells
            .iter()
            .zip(self.surface_curr.cells.iter())
        {
            if matches!(cell_curr.cell_type, CellType::Material(_)) {
                let lte_t: Dimensionless<f64> =
                    (cell_pred.cell_temp - cell_curr.cell_temp).abs() / integ_tol_t;
                let lte_rh: Dimensionless<f64> =
                    (cell_pred.cell_rh - cell_curr.cell_rh).abs() / integ_tol_rh;

                let cell_lte = lte_t.max(lte_rh).value();
                if cell_lte > max_lte {
                    max_lte = cell_lte;
                }
            }
        }

        max_lte
    }

    /// ACCEPT: commit the substep. Refreshes the per-cell extrapolation rates from
    /// the accepted state, advances `substep_time` / `current_time`, accumulates the
    /// boundary fluxes into the `SubstepFluxAccumulator` by the END-POINT
    /// (right-Riemann) rule — the quadrature that is exact for the backward-Euler
    /// update the solver performs — and rotates the accepted state into
    /// `surface_prev`.
    fn advance_accepted_substep<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        dt_step: Time<f64>,
    ) -> Result<()> {
        let rate_t = adaptive_state.rate_t.as_mut().unwrap();
        let rate_rh = adaptive_state.rate_rh.as_mut().unwrap();

        // Update the prediction rates.
        for (idx, cell_prev) in self.surface_prev.cells.indexed_iter() {
            if matches!(cell_prev.cell_type, CellType::Material(_)) {
                rate_t[idx] =
                    (self.surface_curr.cells[idx].cell_temp - cell_prev.cell_temp) / dt_step;
                rate_rh[idx] =
                    (self.surface_curr.cells[idx].cell_rh - cell_prev.cell_rh) / dt_step;
            }
        }

        adaptive_state.substep_time += dt_step;
        adaptive_state.current_time += dt_step;

        {
            // Boundary fluxes at the END of this substep (surface_curr, just converged).
            // The heat fluxes are the TOTAL film fluxes (interior: convective + longwave film).
            let q_int_end = self.surface_curr.heat_flux(BoundaryLocation::Interior)?;
            let q_ext_end = self.surface_curr.heat_flux(BoundaryLocation::Exterior)?;
            let m_int_end = self
                .surface_curr
                .moisture_flux(BoundaryLocation::Interior, &self.boundary_condition)?;
            let m_ext_end = self
                .surface_curr
                .moisture_flux(BoundaryLocation::Exterior, &self.boundary_condition)?;

            // END-POINT (right-Riemann) quadrature for EVERY substep. The implicit-Euler
            // update stores exactly `dt·F_end` in the wall: summed over all cells the
            // interior fluxes cancel pairwise and what is left of the balance is the
            // boundary flux evaluated at the END of the step. Reporting the composite
            // trapezoid `dt·(F_start + F_end)/2` instead hands the zone `½·dt·(F_start −
            // F_end)` more (or less) water/energy per substep than the wall actually
            // took up — first order in dt, with the sign of the flux decay, so it
            // creates water in every drying transient and destroys it in every wetting
            // one (measured: up to 1.3 % of the exchanged mass on the production lining
            // under a sharp heating step, 6.5 % on a harsh outdoor climate). The
            // end-point sum makes ∫ flux dt equal the wall's stored-water change, so
            // the wall↔zone coupling conserves mass. Solver state is unaffected.
            adaptive_state.flux_accumulator.q_int_dt += q_int_end * dt_step;
            adaptive_state.flux_accumulator.q_ext_dt += q_ext_end * dt_step;
            adaptive_state.flux_accumulator.m_int_dt += m_int_end * dt_step;
            adaptive_state.flux_accumulator.m_ext_dt += m_ext_end * dt_step;
            adaptive_state.flux_accumulator.total_time += dt_step;
        }

        std::mem::swap(&mut self.surface_prev, &mut self.surface_curr);
        self.surface_curr.sync_state_from(&self.surface_prev)?;

        Ok(())
    }

    /// PI-controller next-dt prediction. Amplifies `dt_step` by the integral term
    /// `(1/LTE)^k_i`, optionally the proportional term `(prev_LTE/LTE)^k_p`, applies
    /// the safety factor, and clamps asymmetrically (more aggressive on the first
    /// successful step after a failure). Returns the proposed next Δt.
    fn pi_predict_dt(
        pi: &PiControllerSettings,
        prev_lte: Option<f64>,
        max_lte: f64,
        dt_step: Time<f64>,
    ) -> Time<f64> {
        let k_i = pi.k_i;
        let k_p = pi.k_p;
        let safety = pi.safety;
        let amp_min = pi.amp_min;
        let amp_max_first = pi.amp_max_first;
        let amp_max = pi.amp_max;

        let mut amplification = (1.0 / max_lte).powf(k_i);
        if let Some(pe) = prev_lte {
            amplification *= (pe / max_lte).powf(k_p);
        }

        // Asymmetric Clamping
        // Allow more aggressive amplification on the first successful step after a failure to quickly recover,
        // but clamp more conservatively after that to ensure stability.
        let max_amp = if prev_lte.is_none() {
            amp_max_first
        } else {
            amp_max
        };
        amplification = (amplification * safety).clamp(amp_min, max_amp);

        dt_step * amplification
    }

    /// REJECT: shrink `dt_step` (asymmetric floor in cold start vs. afterwards) and
    /// restore `surface_curr` / `surface_next` to the pre-substep state so the next
    /// attempt is not contaminated. Returns the shrunk next Δt.
    fn shrink_rejected_substep(
        &mut self,
        prev_lte: Option<f64>,
        max_lte: f64,
        dt_step: Time<f64>,
    ) -> Result<Time<f64>> {
        // In cold start, be more forgiving to allow the controller to gather data. After that, be more conservative to ensure stability.
        let min_shrink = if prev_lte.is_none() { 0.5 } else { 0.1 };
        let shrink_factor = (0.8 / max_lte.sqrt()).clamp(min_shrink, 0.9);

        let current_dt = dt_step * shrink_factor;

        // Clean up any partial updates to avoid contaminating the next iteration
        self.surface_curr.sync_state_from(&self.surface_prev)?;
        self.surface_next.sync_state_from(&self.surface_prev)?;

        Ok(current_dt)
    }

    /// FINALIZE: record the proposed Δt for the next macro-step's warm start, rotate
    /// the macro-step's converged result back into `surface_prev`, and restore the
    /// macro timestep (`self.settings.timestep = macro_dt`) plus bookkeeping fields.
    fn finalize_macro_step<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        current_dt: Time<f64>,
        macro_dt: Time<f64>,
    ) -> Result<()> {
        // Save the successfully proposed macro Δt for the next step's warm start extrapolation,
        // but only if we made actual progress (substep_time > 0) to avoid poisoning the next step with a failed attempt.
        adaptive_state.proposed_dt = Some(current_dt);

        mem::swap(&mut self.surface_prev, &mut self.surface_curr);
        self.surface_prev
            .sync_state_from(adaptive_state.prev_surface_state.as_ref().unwrap())?;

        self.settings.timestep = macro_dt;
        self.dt_prev_macro = Some(macro_dt);
        adaptive_state.substep_time = macro_dt;

        Ok(())
    }
}
