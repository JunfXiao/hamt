//! Adaptive sub-stepping state and flux accumulator.

use crate::cell::{BoundaryLocation, CellType};
use crate::surface::Surface;
use crate::uom::dimensions::{
    ArealEnergyDensity, ArealMassDensity, Dimensionless, EnergyFluxDensity, MassFluxDensity, Time,
};
use crate::uom::units::{
    dimensionless, hours, j_per_m2, kelvin, kg_per_m2, kg_per_m2_s, seconds, w_per_m2,
};
use crate::uom::{Quantity, units};
use crate::error;
use anyhow::Result;
use std::mem;

use super::settings::{RhRate, TempRate};
use super::HamtManager;

/// Accumulates time-weighted boundary fluxes across adaptive substeps.
/// After all substeps, dividing each `*_dt` field by `total_time` yields the
/// time-averaged flux density over the macro-timestep.
#[derive(Debug, Clone)]
pub struct SubstepFluxAccumulator {
    /// ∫ q_int dt  [J/m²]
    pub q_int_dt: ArealEnergyDensity<f64>,
    /// ∫ q_ext dt  [J/m²]
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

    /// Time-averaged interior heat flux [W/m²].
    pub fn avg_q_int(&self) -> EnergyFluxDensity<f64> {
        if self.total_time > 0.0 * seconds {
            self.q_int_dt / self.total_time
        } else {
            0.0 * w_per_m2
        }
    }

    /// Time-averaged exterior heat flux [W/m²].
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
/// Solves for the current timestep using an adaptive stepping strategy based on a PI controller.
    pub fn solve_timestep_adaptive<T>(
        &mut self,
        adaptive_state: &mut AdaptiveState<T>,
        mut callback_state_change: Option<Box<dyn FnMut(&AdaptiveState<T>) -> Result<()>>>,
    ) -> Result<()> {
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
        let mut current_dt: Time<f64> = adaptive_state.proposed_dt.unwrap_or(macro_dt);

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

        // Use adaptive_state.substep_time as the accumulator condition
        while adaptive_state.substep_time < macro_dt {
            let time_remaining = macro_dt - adaptive_state.substep_time;
            let is_boundary_step = current_dt >= time_remaining;
            let rate_t = adaptive_state.rate_t.as_mut().unwrap();
            let rate_rh = adaptive_state.rate_rh.as_mut().unwrap();

            // Boundary Clamping
            let dt_step: Time<f64> = current_dt.min(time_remaining);

            let predicted_surface = adaptive_state
                .predicted_surface_state
                .as_mut()
                .expect("Predicted Surface shouldn't be None");

            predicted_surface.sync_state_from(&self.surface_prev)?;
            for (idx, cell) in predicted_surface.cells.indexed_iter_mut() {
                if matches!(cell.cell_type, CellType::Material(_)) {
                    cell.cell_temp += (*rate_t)[idx] * dt_step;
                    cell.cell_rh += (*rate_rh)[idx] * dt_step;
                    cell.calc_dependent_state(&self.boundary_condition, &self.settings);
                }
            }


            // Predictor Warm-Start
            // Use the predicted surface as the initial guess for the current step.
            //This can significantly reduce iterations for small steps, especially in the early phase of a macro-step where the state changes rapidly.
            self.surface_curr.sync_state_from(predicted_surface)?;

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


            // Decouple Integration Tolerance
            // Instead of using the global convergence tolerances, we define separate integration tolerances for temperature and relative humidity that are more lenient.
            // This allows the PI controller to make informed decisions based on the physical changes rather than strict solver convergence, which can be too tight for adaptive stepping.
            let integ_tol_t: crate::uom::dimensions::Temperature<f64> = self.settings.temp_convergence_tolerance.max(self.settings.pi.integ_tol_t_floor);
            let integ_tol_rh: Dimensionless<f64> = self.settings.rh_convergence_tolerance.max(self.settings.pi.integ_tol_rh_floor);

            let mut max_lte: f64 = 1e-12;
            for (cell_pred, cell_curr) in predicted_surface
                .cells
                .iter()
                .zip(self.surface_curr.cells.iter())
            {
                if matches!(cell_curr.cell_type, CellType::Material(_)) {
                    let lte_t:Dimensionless<f64> = (cell_pred.cell_temp - cell_curr.cell_temp).abs() / integ_tol_t;
                    let lte_rh:Dimensionless<f64> = (cell_pred.cell_rh - cell_curr.cell_rh).abs() / integ_tol_rh;

                    let cell_lte = lte_t.max(lte_rh).value();
                    if cell_lte > max_lte {
                        max_lte = cell_lte;
                    }
                }
            }

            // Accept/Reject Logic
            if max_lte <= 1.0 {
                // ACCEPT

                // Update the prediction rates.
                for (idx, cell_prev) in self.surface_prev.cells.indexed_iter() {
                    if matches!(cell_prev.cell_type, CellType::Material(_)) {
                        (*rate_t)[idx] = (self.surface_curr.cells[idx].cell_temp - cell_prev.cell_temp) / dt_step;
                        (*rate_rh)[idx] = (self.surface_curr.cells[idx].cell_rh - cell_prev.cell_rh) / dt_step;
                    }
                }

                adaptive_state.substep_time += dt_step;
                adaptive_state.current_time += dt_step;

                // Accumulate fluxes.
                {
                    let q_int = self.surface_curr.heat_flux(BoundaryLocation::Interior)?;
                    let q_ext = self.surface_curr.heat_flux(BoundaryLocation::Exterior)?;
                    let m_int = self
                        .surface_curr
                        .moisture_flux(BoundaryLocation::Interior, &self.boundary_condition)?;
                    let m_ext = self
                        .surface_curr
                        .moisture_flux(BoundaryLocation::Exterior, &self.boundary_condition)?;

                    adaptive_state.flux_accumulator.q_int_dt += q_int * dt_step;
                    adaptive_state.flux_accumulator.q_ext_dt += q_ext * dt_step;
                    adaptive_state.flux_accumulator.m_int_dt += m_int * dt_step;
                    adaptive_state.flux_accumulator.m_ext_dt += m_ext * dt_step;
                    adaptive_state.flux_accumulator.total_time += dt_step;
                }

                std::mem::swap(&mut self.surface_prev, &mut self.surface_curr);
                self.surface_curr.sync_state_from(&self.surface_prev)?;

                if let Some(ref mut cb) = callback_state_change {
                    cb(adaptive_state)?;
                }

                // PI Controller
                if !is_boundary_step {
                    let k_i = self.settings.pi.k_i;
                    let k_p = self.settings.pi.k_p;
                    let safety = self.settings.pi.safety;
                    let amp_min = self.settings.pi.amp_min;
                    let amp_max_first = self.settings.pi.amp_max_first;
                    let amp_max = self.settings.pi.amp_max;

                    let mut amplification = (1.0 / max_lte).powf(k_i);
                    if let Some(pe) = adaptive_state.prev_lte {
                        amplification *= (pe / max_lte).powf(k_p);
                    }

                    // Asymmetric Clamping
                    // Allow more aggressive amplification on the first successful step after a failure to quickly recover,
                    // but clamp more conservatively after that to ensure stability.
                    let max_amp = if adaptive_state.prev_lte.is_none() { amp_max_first } else { amp_max };
                    amplification = (amplification * safety).clamp(amp_min, max_amp);

                    current_dt = dt_step * amplification;
                    adaptive_state.prev_lte = Some(max_lte);
                }
            } else {
                // REJECT

                // In cold start, be more forgiving to allow the controller to gather data. After that, be more conservative to ensure stability.
                let min_shrink = if adaptive_state.prev_lte.is_none() { 0.5 } else { 0.1 };
                let shrink_factor = (0.8 / max_lte.sqrt()).clamp(min_shrink, 0.9);

                current_dt = dt_step * shrink_factor;

                // Clean up any partial updates to avoid contaminating the next iteration
                self.surface_curr.sync_state_from(&self.surface_prev)?;
                self.surface_next.sync_state_from(&self.surface_prev)?;
            }
        }

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
