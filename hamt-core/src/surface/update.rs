use crate::{
    parallel::{par_or_seq, par_or_seq_mut}, uom::{
        Quantity, constants::{WATER_LATENT_HEAT, WATER_SPECIFIC_HEAT}, dimensions::{
            HeatTransferCoeff, MassFlux, MassFluxPerTemperature, Power, SpecificEnergy, ThermalConductance, VaporConductance, VaporTransportCoeff
        }, units::{
            dimensionless, j_per_kg, k_per_w, kelvin, kg_per_m2_s_pa, kg_per_m3, kg_per_s,
            kilograms, pa_s_per_kg, pascals, seconds, w_per_k, w_per_m2_k, watts
        }
    }
};

use super::*;

use anyhow::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;


/// Blend (next, curr) by the cell's relaxation factor, clamp RH to [0, 1.01],
/// and recompute dependent state. Shared by the BlockJacobi write-back and the
/// BlockTridiag scatter.
fn relax_write_cell(
    cell: &mut Cell,
    next_t: Temperature<f64>,
    next_rh: Dimensionless<f64>,
    curr_t: Temperature<f64>,
    curr_rh: Dimensionless<f64>,
    bc: &BoundaryCondition,
    settings: &HamtSettings,
) {
    let alpha = cell.relaxation_factor;
    let relaxed_t = if next_t.is_finite() { next_t * alpha + curr_t * (1.0 - alpha) } else { curr_t };
    let relaxed_rh = if next_rh.is_finite() { next_rh * alpha + curr_rh * (1.0 - alpha) } else { curr_rh };
    cell.cell_temp = relaxed_t;
    cell.cell_rh = relaxed_rh.clamp(Quantity(0.0), Quantity(1.01));
    cell.calc_dependent_state(bc, settings);
}

impl Surface {
    /// Executes a single iteration of the HAMT simulation for a single timestep.
    /// `self` is the **only** writable `surface_next`.
    /// Reads from `surface_prev` (t-1) and `surface_curr` (last iteration state),
    /// and never writes to them.
    pub fn calculate_iteration(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        // --- Sync `surface_next` with current iteration state ---
        self.sync_state_from(surface_curr)?;

        // --- Apply latest boundary coefficients/conditions ---
        self.apply_boundary_coefficients(bc, settings)?;

        // --- Combined Heat and Moisture Step ---
        // Note: a pre-step update_cell_properties call would be redundant.
        // The solver reads dependent state (vp_sat, d_psat_d_t, w_thermal_c, mu, dw,
        // dwd_phi, h_sorp_diff) from `surface_curr`, not from `self`. `surface_curr`
        // was left in a consistent state by the previous iteration's post-step call
        // (or by tick_timestep at iter=0). Boundary cells overwritten in
        // apply_boundary_coefficients above are not read from `self` during the step.
        self.calculate_physics_step(settings, bc, surface_prev, surface_curr)?;

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        Ok(())
    }

    /// Copy the current iteration state into `surface_next` before property updates.
    pub fn sync_state_from(&mut self, surface_curr: &Surface) -> Result<()> {
        if let (Some(dst), Some(src)) = (self.cells.as_slice_mut(), surface_curr.cells.as_slice()) {
            par_or_seq_mut(dst)
                .zip(par_or_seq(src))
                .for_each(|(cell, ref_cell)| {
                    *cell = ref_cell.clone();
                    cell.sync_state_from(ref_cell);
                });
        } else {
            for (cell, ref_cell) in self.cells.iter_mut().zip(surface_curr.cells.iter()) {
                *cell = ref_cell.clone();
                cell.sync_state_from(ref_cell);
            }
        }
        Ok(())
    }

    /// Update dependent cell properties based on the current iteration's Temp and RH.
    pub fn update_cell_properties(
        &mut self,
        bc: &BoundaryCondition,
        settings: &HamtSettings,
    ) -> Result<()> {
        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).for_each(|cell| cell.calc_dependent_state(bc, settings));
        } else {
            for cell in self.cells.iter_mut() {
                cell.calc_dependent_state(bc, settings);
            }
        }
        Ok(())
    }

    /// Push boundary htc/vtc/temp/rh/vp into boundary cells before property updates.
    pub(crate) fn apply_boundary_coefficients(
        &mut self,
        bc: &BoundaryCondition,
        _settings: &HamtSettings,
    ) -> Result<()> {
        let nx = self.cells.raw_dim()[0];
        let interior_mat_idx = self.interior_surface_material_idx;
        let interior_mat_temp: Option<Temperature<f64>> =
            interior_mat_idx.map(|i| self.cells[[i, 0, 0]].cell_temp);
        for i in 0..nx {
            let cell = &mut self.cells[[i, 0, 0]];
            if cell.is_boundary() {
                if cell.is_exterior() {
                    cell.htc = bc.heat_transfer_coeff_ext;
                    cell.cell_temp = bc.temp_outside_air;
                    cell.cell_rh = bc.rh_outside_air;
                    // Use the dedicated moisture temperature (actual outside air T) for VP_bc,
                    // so the VP boundary condition is physically correct even when cell_temp
                    // is set to the star-node temperature for the heat convection path.
                    let psat = physics::psy_psat_fn_temp(bc.temp_outside_air_moisture);
                    let rh = cell.cell_rh;
                    let vtc = if matches!(cell.cell_type, CellType::Boundary(crate::cell::Boundary::Convection, _)) {
                        if bc.vapor_transport_coeff_ext > VaporTransportCoeff::<f64>::zero() {
                            bc.vapor_transport_coeff_ext
                        } else {
                            let rho_v = physics::vd_from_t_rh(cell.cell_temp, rh);
                            let denom = psat * rh;
                            if denom > Quantity(1.0e-9) {
                                bc.mass_convection_coeff_ext * rho_v / denom
                            } else {
                                Quantity(0.0)
                            }
                        }
                    } else {
                        Quantity(0.0)
                    };
                    cell.vtc = vtc;
                    cell.cell_vp = rh * psat;
                } else {
                    match &cell.cell_type {
                        CellType::Boundary(crate::cell::Boundary::Convection, BoundaryLocation::Interior) => {
                            // Pure convective HTC (HCONVI). Paired with TAIR on the heat side,
                            // shared with VP_bc on the moisture side. In separated-conv-rad mode
                            // (14-input deck) temp_inside_air_moisture = TAIR (NType 1);
                            // in legacy 11-input mode it falls back to temp_inside_air = TSTAR,
                            // preserving byte-identical single-potential behaviour.
                            cell.htc = bc.heat_transfer_coeff_conv_int;
                            cell.cell_temp = bc.temp_inside_air_moisture;
                            cell.cell_rh = bc.rh_inside_air;
                            let psat = physics::psy_psat_fn_temp(bc.temp_inside_air_moisture);
                            let rh = cell.cell_rh;
                            let vtc = if bc.vapor_transport_coeff_int > VaporTransportCoeff::<f64>::zero() {
                                bc.vapor_transport_coeff_int
                            } else {
                                let rho_v = physics::vd_from_t_rh(cell.cell_temp, rh);
                                let denom = psat * rh;
                                if denom > Quantity(1.0e-9) {
                                    bc.mass_convection_coeff_int * rho_v / denom
                                } else {
                                    0.0 * kg_per_m2_s_pa
                                }
                            };
                            cell.vtc = vtc;
                            cell.cell_vp = rh * psat;
                        }
                        CellType::Boundary(crate::cell::Boundary::Radiation, BoundaryLocation::Interior) => {
                            // Linearised longwave exchange against mean radiant temperature.
                            // h_rad = 4 σ ε T_m³, T_m ≈ mean of surface and TMR.
                            // Use the adjacent interior material cell's current temperature as the
                            // surface proxy.
                            // Legacy decks (<14 Type395 inputs) keep using the combined
                            // REQV-derived h_conv on the Convection cell; the Radiation cell
                            // must then be inert (h_rad = 0) so the refactor is byte-identical
                            // to the pre-refactor single-potential behaviour.
                            let h_rad = if bc.has_separate_radiant_int {
                                let t_s = interior_mat_temp.unwrap_or(bc.temp_inside_mean_radiant);
                                let t_m_k = (t_s + bc.temp_inside_mean_radiant) * 0.5;
                                // strip: `.powi(3)` is only defined on the inner f64; the
                                // result is re-tagged as HeatTransferCoeff via `* w_per_m2_k`.
                                let t_m_cubed = t_m_k.value_in(kelvin).powi(3);
                                4.0 * physics::SIGMA * physics::EPSILON * t_m_cubed * w_per_m2_k
                            } else {
                                HeatTransferCoeff::<f64>::zero()
                            };
                            cell.htc = h_rad;
                            cell.cell_temp = bc.temp_inside_mean_radiant;
                            cell.vtc = VaporTransportCoeff::<f64>::zero();
                            cell.cell_vp = 0.0 * pascals;
                            cell.cell_rh = bc.rh_inside_air; // unused, kept valid
                        }
                        _ => {
                            // Fallback: legacy combined behaviour for any other interior
                            // boundary cell type not explicitly handled above.
                            cell.htc = bc.heat_transfer_coeff_int;
                            cell.cell_temp = bc.temp_inside_air;
                            cell.cell_rh = bc.rh_inside_air;
                            let psat = physics::psy_psat_fn_temp(bc.temp_inside_air_moisture);
                            let rh = cell.cell_rh;
                            cell.vtc = VaporTransportCoeff::<f64>::zero();
                            cell.cell_vp = rh * psat;
                        }
                    }
                }
            } else {
                cell.vtc = VaporTransportCoeff::<f64>::zero();
                cell.htc = HeatTransferCoeff::<f64>::zero();
            }
        }
        Ok(())
    }

    /// Combined Heat and Moisture step: compute next T and RH for all cells.
    fn calculate_physics_step(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        let dt_s = settings.timestep;

        let nx = self.cells.raw_dim()[0];
        let indices: Vec<usize> = (0..nx).collect();

        let results: Vec<(Temperature<f64>, Dimensionless<f64>)> = par_or_seq(&indices)
            .map(|&i| {
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                if !cell_curr.is_boundary() {
                    let cell_prev = &surface_prev.cells[[i, 0, 0]];
                    let is_material = matches!(cell_curr.cell_type, CellType::Material(_));

                    // =========================================================
                    // Build all neighbor sums in a single pass
                    //   Heat:     oorsum, torsum, vapor_diffusion_net_flux
                    //   Vapor:    vpoosum, vporsum
                    //   Liquid:   phioosum, phiorsum
                    // Merging the two former loops avoids calling
                    // calculate_resistance_pair twice per neighbor per cell.
                    // =========================================================
                    let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
                    let mut torsum: Power<f64> = 0.0 * watts;
                    let mut vapor_diffusion_net_flux: MassFlux<f64> = 0.0 * kilograms / seconds;
                    let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
                    let mut vporsum: MassFlux<f64> = 0.0 * kg_per_s;
                    let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s;
                    let mut phiorsum: MassFlux<f64> = 0.0 * kg_per_s;

                    let curr_vp_pa = cell_curr.cell_vp;

                    for nbr_i in surface_curr.neighbor_indices(i, nx) {
                        let nbr_curr = &surface_curr.cells[[nbr_i, 0, 0]];

                        let r = Self::calculate_resistance_triple(cell_curr, nbr_curr, bc.pressure_air);

                        if r.therm > 1e-12 * k_per_w {
                            oorsum += 1.0 / r.therm;
                            torsum += nbr_curr.cell_temp / r.therm;
                        }

                        if r.vapor > 1e-12 * pa_s_per_kg {
                            let nbr_vp = nbr_curr.cell_vp;
                            let vc = 1.0 / r.vapor;
                            vapor_diffusion_net_flux += (nbr_vp - curr_vp_pa) / r.vapor;
                            vpoosum += vc;
                            vporsum += nbr_vp * vc;
                        }

                        let r_total: Quantity<f64, _> = r.liquid * dimensionless;
                        if r_total < f64::MAX * dimensionless / kg_per_s
                            && r_total > 1e-13 * dimensionless / kg_per_s
                        {
                            let cond: MassFlux<f64> = 1.0 / r_total;
                            phioosum += cond;
                            phiorsum += nbr_curr.cell_rh * cond;
                        }
                    }

                    // =========================================================
                    // Cell-level scalars (independent of solver path)
                    // =========================================================
                    let tcap_j_per_k = (cell_curr.density * cell_curr.spech
                        + cell_curr.volumetric_water * WATER_SPECIFIC_HEAT)
                        * cell_curr.volume;

                    let wcap = if cell_curr.dwd_phi > 1e-10 * kg_per_m3 {
                        cell_curr.dwd_phi * cell_curr.volume
                    } else {
                        debug!(
                            "Cell {} has very low dwd_phi, using fallback moisture capacity to prevent instability",
                            i
                        );
                        1.0e-10 * kilograms
                    };

                    let q_weight = surface_curr
                        .user_defined_cell_idx
                        .as_ref()
                        .map(|weights| {
                            weights
                                .iter()
                                .filter(|(idx, _)| idx[0] == i)
                                .map(|(_, w)| w)
                                .sum::<f64>()
                        })
                        .unwrap_or(0.0);
                    let q_adds = bc.q_addition_userdef * q_weight;

                    // Deposit absorbed radiation into the boundary-adjacent material cell.
                    let q_boundary_energy: Power<f64> =
                        if Some(i) == surface_curr.interior_surface_material_idx {
                            bc.additional_energy_int
                        } else if Some(i) == surface_curr.exterior_surface_material_idx {
                            bc.additional_energy_ext
                        } else {
                            0.0 * watts
                        };

                    let t_curr = cell_curr.cell_temp;
                    let vpsat_curr = cell_curr.vp_sat;
                    // dPsat/dT [Pa/K] pre-cached in calc_dependent_state (Magnus formula)
                    let d_psat_d_t = cell_curr.d_psat_d_t;

                    // =========================================================
                    // Solve for (T_next, RH_next)
                    // =========================================================
                    let (t_next, rh_next);

                    if settings.coupled_heat_moisture_enabled && is_material && settings.latent_heat_enabled {
                        // ── Symmetric 2×2 block solve (typed) ─────────────────────────────────
                        //
                        // Linearise  P_v,i^{k+1} ≈ Psat(T^k)·φ^{k+1} + φ^k·(dPsat/dT)·(T^{k+1}−T^k)
                        //
                        //  ┌ A_tt  A_tp ┐ ┌ T_next ┐   ┌ b_t ┐
                        //  └ A_pt  A_pp ┘ └ φ_next ┘ = └ b_p ┘
                        //
                        //   A_tt : ThermalConductance      [W/K]   = oorsum + tcap/dt + L_eff·cross
                        //   A_tp : Power                   [W]     = L_eff · vpoosum · Psat
                        //   A_pt : MassFluxPerTemperature  [kg/(s·K)] = cross
                        //   A_pp : MassFlux                [kg/s]  = phioosum + vpoosum·Psat + wcap/dt
                        //   b_t  : Power                   [W]
                        //   b_p  : MassFlux                [kg/s]
                        //   cross = vpoosum · φ_k · dPsat/dT       [kg/(s·K)]
                        //   L_eff = L_v + H_diff(φ)                [J/kg]
                        //
                        // Cramer's rule produces composite intermediate dimensions
                        // (W·kg/s and W·kg/(s·K)); diman tracks them anonymously, and
                        // the divisions resolve to Temperature and Dimensionless.

                        // WATER_LATENT_HEAT is declared as a `constant` in unit_system!
                        // and is therefore a Unit, not a Quantity — coerce via `1.0 *`.
                        let l_eff: SpecificEnergy<f64> = 1.0 * WATER_LATENT_HEAT
                            + if settings.sorption_enthalpy_enabled {
                                cell_curr.h_sorp_diff
                            } else {
                                0.0 * j_per_kg
                            };

                        let phi_k    = cell_curr.cell_rh;          // Dimensionless
                        let phi_prev = cell_prev.cell_rh;          // Dimensionless
                        let t_prev   = cell_prev.cell_temp;        // Temperature

                        // cross = vpoosum · φ_k · dPsat/dT  →  MassFluxPerTemperature
                        let cross: MassFluxPerTemperature<f64> = vpoosum * phi_k * d_psat_d_t;

                        // Explicit annotations force diman to reduce composite dimension
                        // expressions to their canonical Dimension struct, which is what
                        // the subsequent Cramer subtractions need to type-check.
                        let a_tt: ThermalConductance<f64> =
                            oorsum + tcap_j_per_k / dt_s + l_eff * cross;
                        let a_tp: Power<f64> = l_eff * vpoosum * vpsat_curr;
                        let a_pt: MassFluxPerTemperature<f64> = cross;
                        let a_pp: MassFlux<f64> =
                            phioosum + vpoosum * vpsat_curr + wcap / dt_s;

                        let b_t: Power<f64> = torsum
                            + q_adds
                            + q_boundary_energy
                            + tcap_j_per_k * t_prev / dt_s
                            + l_eff * (vporsum + cross * t_curr);
                        let b_p: MassFlux<f64> = phiorsum
                            + vporsum
                            + wcap * phi_prev / dt_s
                            + cross * t_curr;

                        // ── Linear algebra kernel ─────────────────────────────────────
                        // Cramer's rule mixes composite dimensions (W·kg/s and W·kg/(s·K))
                        // that diman cannot canonicalise across the subtractions in the
                        // determinant and numerators. We treat the 2×2 solve like a call
                        // into a BLAS routine: enter with typed magnitudes (.0 = SI value),
                        // do unit-agnostic arithmetic, re-tag the typed outputs on exit.
                        // Inputs and outputs above this block are fully typed.
                        let a_tt_v = a_tt.0;
                        let a_tp_v = a_tp.0;
                        let a_pt_v = a_pt.0;
                        let a_pp_v = a_pp.0;
                        let b_t_v  = b_t.0;
                        let b_p_v  = b_p.0;

                        let det_v = a_tt_v * a_pp_v - a_tp_v * a_pt_v;

                        let (t_n_typed, p_n_typed): (Temperature<f64>, Dimensionless<f64>) =
                            if det_v.abs() > 1e-30 {
                                let t_n_v = (b_t_v * a_pp_v - a_tp_v * b_p_v) / det_v;
                                let p_n_v = (a_tt_v * b_p_v - a_pt_v * b_t_v) / det_v;
                                (t_n_v * kelvin, p_n_v * dimensionless)
                            } else {
                                // Near-singular (e.g. no vapor coupling): fall back to current
                                // state; the outer relaxation loop will converge next iteration.
                                (t_curr, phi_k)
                            };

                        t_next = if t_n_typed.is_finite() {
                            t_n_typed
                        } else {
                            t_curr
                        };
                        rh_next = if p_n_typed.is_finite() {
                            p_n_typed
                        } else {
                            cell_curr.cell_rh
                        };
                    } else {
                        // ── Legacy sequential scalar solve ────────────────────────────────────
                        // Reproduce the pre-coupling behaviour exactly when flag is false.
                        // Taylor Series Linearization for Vapor Pressure:
                        // P_v^{k+1} ≈ Psat(T^k)·φ^{k+1} + φ^k·(dPsat/dT)·(T^{k+1} − T^k)
                        let mut q_vapor: Power<f64> = 0.0 * watts;
                        if is_material && settings.latent_heat_enabled {
                            q_vapor = vapor_diffusion_net_flux * WATER_LATENT_HEAT;
                            if settings.sorption_enthalpy_enabled {
                                q_vapor += vapor_diffusion_net_flux * cell_curr.h_sorp_diff;
                            }
                            if q_vapor.abs() > settings.max_latent_heat {
                                // strip: signum() is only defined on the inner f64;
                                // we want a raw ±1.0 to re-tag the clamped magnitude.
                                q_vapor =
                                    q_vapor.value_in(watts).signum() * settings.max_latent_heat;
                            }
                        }

                        let denom_t: ThermalConductance<f64> = oorsum + tcap_j_per_k / dt_s;
                        let t_next_s: Temperature<f64> = if denom_t > 1e-10 * w_per_k {
                            let inertial = tcap_j_per_k * cell_prev.cell_temp / dt_s;
                            (torsum + q_vapor + q_adds + q_boundary_energy + inertial) / denom_t
                        } else {
                            t_curr
                        };
                        t_next = if t_next_s.is_finite() {
                            t_next_s
                        } else {
                            t_curr
                        };

                        // Moisture: thermal correction uses the just-computed t_next.
                        // delta_t : Temperature, d_psat_d_t : Pa/K, cell_rh : Dimensionless
                        // → p_sat_thermal_correction : Pressure (Pa)
                        // → thermal_source_term      : MassFlux (kg/s)
                        let delta_t = t_next - t_curr;
                        let p_sat_thermal_correction =
                            cell_curr.cell_rh * d_psat_d_t * delta_t;
                        let thermal_source_term = vpoosum * p_sat_thermal_correction;
                        let denom_rh: MassFlux<f64> =
                            phioosum + vpoosum * vpsat_curr + wcap / dt_s;

                        let rh_next_s: Dimensionless<f64> = if denom_rh > 1e-20 * kg_per_s {
                            (phiorsum + vporsum + (wcap * cell_prev.cell_rh) / dt_s
                                - thermal_source_term)
                                / denom_rh
                        } else {
                            debug!(
                                "Cell {} has very low moisture conductance {:?}, using fallback RH to prevent instability",
                                i, denom_rh
                            );
                            cell_curr.cell_rh
                        };
                        rh_next = if rh_next_s.is_finite() {
                            rh_next_s
                        } else {
                            debug!(
                                "Cell {} has non-finite RH_next {:?}, using fallback RH to prevent instability",
                                i, rh_next_s
                            );
                            cell_curr.cell_rh
                        };
                    }

                    (t_next, rh_next)
                } else {
                    // Boundary cells: return the same T and RH that
                    // apply_boundary_coefficients() already set, so the
                    // post-step write loop does not overwrite them.
                    if cell_curr.is_exterior() {
                        (bc.temp_outside_air, bc.rh_outside_air)
                    } else {
                        match &cell_curr.cell_type {
                            CellType::Boundary(crate::cell::Boundary::Convection, BoundaryLocation::Interior) => {
                                (bc.temp_inside_air_moisture, bc.rh_inside_air)
                            }
                            CellType::Boundary(crate::cell::Boundary::Radiation, BoundaryLocation::Interior) => {
                                (bc.temp_inside_mean_radiant, bc.rh_inside_air)
                            }
                            _ => {
                                (bc.temp_inside_air, bc.rh_inside_air)
                            }
                        }
                    }
                }
            })
            .collect();

        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).enumerate().for_each(|(i, cell)| {
                let next_temp = results[i].0;
                let next_rh = results[i].1;
                let curr_temp = surface_curr.cells[[i, 0, 0]].cell_temp;
                let curr_rh = surface_curr.cells[[i, 0, 0]].cell_rh;
                relax_write_cell(cell, next_temp, next_rh, curr_temp, curr_rh, bc, settings);
            });
        } else {
            for i in 0..nx {
                let next_temp = results[i].0;
                let next_rh = results[i].1;
                let curr_temp = surface_curr.cells[[i, 0, 0]].cell_temp;
                let curr_rh = surface_curr.cells[[i, 0, 0]].cell_rh;
                relax_write_cell(
                    &mut self.cells[[i, 0, 0]],
                    next_temp, next_rh, curr_temp, curr_rh,
                    bc, settings,
                );
            }
        }
        Ok(())
    }

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
        self.sync_state_from(surface_curr)?;
        self.apply_boundary_coefficients(bc, settings)?;

        // Virtual cells aren't updated by `apply_boundary_coefficients`. BlockJacobi's
        // per-cell 2×2 solve covers them as a side-effect of iterating over all
        // non-boundary cells, but BlockTridiag's scatter writes only material
        // cells. Compute Virtual cells here as the conductance-weighted average
        // of their non-Virtual neighbours' state — the implicit-Euler answer
        // for a cell with negligible capacity. This produces a non-zero
        // Virtual ↔ Conv gradient that the moisture-flux query relies on.
        let nx = self.cells.raw_dim()[0];
        // Collect (virtual_li, [nbr_li, ...]) before any cell borrows to avoid
        // simultaneous &self and &self.cells conflicts in the inner loop.
        let virtual_nbr_pairs: Vec<(usize, Vec<usize>)> = (0..self.cells.len())
            .filter(|&li| matches!(self.cells[[li, 0, 0]].cell_type, crate::cell::CellType::Virtual(_)))
            .map(|li| {
                let nbrs: Vec<usize> = self.neighbor_indices(li, nx).collect();
                (li, nbrs)
            })
            .collect();

        let mut virtual_updates: Vec<(usize, f64, f64, f64)> = Vec::new();
        for (li, nbr_indices) in virtual_nbr_pairs {
            let cell = &self.cells[[li, 0, 0]];

            // strip: linear-algebra kernel boundary
            let mut sum_inv_v: f64 = 0.0;
            let mut sum_inv_v_vp: f64 = 0.0;
            let mut sum_inv_t: f64 = 0.0;
            let mut sum_inv_t_t: f64 = 0.0;

            for nbr_li in nbr_indices {
                let nbr = &self.cells[[nbr_li, 0, 0]];
                let (therm_r, vapor_r) = Surface::calculate_resistance_pair(
                    cell, nbr, bc.pressure_air,
                );
                if vapor_r > 1e-12 * pa_s_per_kg {
                    let inv_v = 1.0 / vapor_r.value_in(pa_s_per_kg);
                    sum_inv_v += inv_v;
                    sum_inv_v_vp += inv_v * nbr.cell_vp.value_in(pascals);
                }
                if therm_r > 1e-12 * k_per_w {
                    let inv_t = 1.0 / therm_r.value_in(k_per_w);
                    sum_inv_t += inv_t;
                    sum_inv_t_t += inv_t * nbr.cell_temp.value_in(kelvin);
                }
            }

            // Fall back to current values when no neighbour has non-zero coupling
            // (degenerate; should not happen for the standard 1-D wall layout).
            let new_t = if sum_inv_t > 0.0 {
                sum_inv_t_t / sum_inv_t
            } else {
                cell.cell_temp.value_in(kelvin)
            };
            let new_vp = if sum_inv_v > 0.0 {
                sum_inv_v_vp / sum_inv_v
            } else {
                cell.cell_vp.value_in(pascals)
            };
            let psat = crate::physics::psy_psat_fn_temp(new_t * kelvin)
                .value_in(pascals);
            let new_rh = if psat > 0.0 { (new_vp / psat).max(0.0).min(1.01) } else { 0.0 };
            virtual_updates.push((li, new_vp, new_t, new_rh));
        }

        for (li, vp, t, rh) in virtual_updates {
            let cell = &mut self.cells[[li, 0, 0]];
            // re-tag at kernel boundary
            cell.cell_vp = vp * pascals;
            cell.cell_temp = t * kelvin;
            cell.cell_rh = rh * dimensionless;
        }

        self.update_cell_properties(bc, settings)?;

        // Assemble from `self`, not `surface_curr`: the BC and Virtual cells on
        // `self` were just refreshed against the current iterate's surface
        // temperature, while the same cells on `surface_curr` still reflect the
        // previous iterate. Material-cell coefficients are identical between the
        // two (sync_state_from copied them and update_cell_properties is idempotent
        // on that data), so this only swaps lagged BC/Virtual coefficients for
        // current-iter ones — restoring near-Newton convergence.
        let system = crate::solver::assembly::assemble_1d(
            self, surface_prev, bc, settings,
        );
        let solution = crate::solver::block_tridiag::solve(
            &system.lower, &system.diag, &system.upper, &system.rhs,
        )
        .map_err(|e| anyhow::anyhow!("block-tridiag solve failed: {:?}", e))?;

        // Scatter solution back, applying per-cell α relaxation
        // (consistent with the BlockJacobi write-back behaviour).
        for (row, &li) in system.material_indices.iter().enumerate() {
            let (t_new_k, phi_new_raw) = solution[row];
            let curr_temp = surface_curr
                .cells
                .get([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?
                .cell_temp;
            let curr_rh = surface_curr
                .cells
                .get([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?
                .cell_rh;
            let cell = self
                .cells
                .get_mut([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?;

            // re-tag at kernel boundary
            let t_new_typed = t_new_k * crate::uom::units::kelvin;
            let phi_new_typed = phi_new_raw * crate::uom::units::dimensionless;

            relax_write_cell(cell, t_new_typed, phi_new_typed, curr_temp, curr_rh, bc, settings);
        }

        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    pub(crate) fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
        let self_cell = self.cells.get([i, 0, 0]).unwrap();

        let left: Vec<usize> = if i > 0 {
            if matches!(
                self_cell.cell_type,
                CellType::Virtual(BoundaryLocation::Exterior)
            ) {
                (0..i).collect()
            } else if i > 1 {
                vec![i - 1]
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        let right = if matches!(
            self_cell.cell_type,
            CellType::Virtual(BoundaryLocation::Interior)
        ) {
            (i + 1..nx).collect()
        } else if i + 1 < nx {
            vec![i + 1]
        } else {
            vec![]
        };

        left.into_iter().chain(right)
    }
}

#[cfg(test)]
mod relax_write_cell_tests {
    use super::*;
    use crate::uom::units::{dimensionless, kelvin};

    fn make_material_cell() -> Cell {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(1);
        // Grab the first material cell from surface_curr.
        mgr.surface_curr
            .cells
            .iter()
            .find(|c| matches!(c.cell_type, crate::cell::CellType::Material(_)))
            .cloned()
            .expect("fixture must have at least one material cell")
    }

    #[test]
    fn relax_write_cell_blends_and_clamps() {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(1);
        let bc = &mgr.boundary_condition;
        let settings = &mgr.settings;

        // ── Case 1: normal blend ────────────────────────────────────────
        let mut cell = make_material_cell();
        cell.relaxation_factor = 0.5;
        let next_t   = 300.0 * kelvin;
        let curr_t   = 290.0 * kelvin;
        let next_rh  = 0.8 * dimensionless;
        let curr_rh  = 0.4 * dimensionless;
        relax_write_cell(&mut cell, next_t, next_rh, curr_t, curr_rh, bc, settings);
        let t_out  = cell.cell_temp.value_in(kelvin);
        let rh_out = cell.cell_rh.value();
        assert!((t_out - 295.0).abs() < 1e-9,  "expected 295 K, got {t_out}");
        assert!((rh_out - 0.6).abs() < 1e-9,   "expected 0.6 RH, got {rh_out}");

        // ── Case 2: next_rh = 2.0 → clamped to 1.01 ───────────────────
        let mut cell2 = make_material_cell();
        cell2.relaxation_factor = 0.5;
        let next_rh_high = 2.0 * dimensionless;   // blended = 0.5*2.0 + 0.5*0.4 = 1.2 > 1.01
        relax_write_cell(&mut cell2, next_t, next_rh_high, curr_t, curr_rh, bc, settings);
        let rh_clamped = cell2.cell_rh.value();
        assert!(
            (rh_clamped - 1.01).abs() < 1e-9,
            "expected clamp to 1.01, got {rh_clamped}"
        );

        // ── Case 3: next_t = NaN → cell_temp stays at curr_t ───────────
        let mut cell3 = make_material_cell();
        cell3.relaxation_factor = 0.5;
        let nan_t = f64::NAN * kelvin;
        relax_write_cell(&mut cell3, nan_t, next_rh, curr_t, curr_rh, bc, settings);
        let t_nan_fallback = cell3.cell_temp.value_in(kelvin);
        assert!(
            (t_nan_fallback - 290.0).abs() < 1e-9,
            "expected fallback to 290 K, got {t_nan_fallback}"
        );
    }
}

#[cfg(test)]
mod block_tridiag_tests {
    use super::*;

    fn fixture(n_div: usize) -> crate::hamt_manager::HamtManager {
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
        mgr.surface_curr.apply_boundary_coefficients(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_prev.apply_boundary_coefficients(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_curr.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_prev.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();

        // surface_next is scratch; clone from surface_curr.
        let mut surface_next = mgr.surface_curr.clone();
        surface_next.calculate_iteration_block_tridiag(
            &mgr.settings, &mgr.boundary_condition,
            &mgr.surface_prev, &mgr.surface_curr,
        ).unwrap();

        for (cell_n, cell_c) in surface_next.cells.iter().zip(mgr.surface_curr.cells.iter()) {
            if !matches!(cell_n.cell_type, crate::cell::CellType::Material(_)) { continue; }
            let dt_k = (cell_n.cell_temp - cell_c.cell_temp)
                .value_in(crate::uom::units::kelvin).abs();
            let drh = (cell_n.cell_rh - cell_c.cell_rh).value().abs();
            assert!(dt_k < 1e-5 && drh < 1e-5,
                    "ΔT={dt_k}, Δφ={drh} at supposed equilibrium");
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
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings).unwrap();
        }
        mgr.boundary_condition.temp_inside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.rh_inside_air = 0.3 * crate::uom::units::dimensionless;
        mgr.boundary_condition.rh_outside_air = 0.3 * crate::uom::units::dimensionless;

        let mut surface_next = mgr.surface_curr.clone();
        surface_next.calculate_iteration_block_tridiag(
            &mgr.settings, &mgr.boundary_condition,
            &mgr.surface_prev, &mgr.surface_curr,
        ).unwrap();

        for cell in surface_next.cells.iter() {
            if !matches!(cell.cell_type, crate::cell::CellType::Material(_)) { continue; }
            let t_new = cell.cell_temp.value_in(crate::uom::units::kelvin);
            assert!(t_new <= t_wall.value_in(crate::uom::units::kelvin) + 1e-3,
                    "T_next {t_new} K > T_curr 298.15 K — wrong direction");
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
                crate::cell::CellType::Virtual(
                    crate::cell::BoundaryLocation::Interior,
                ) => {
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
            if !matches!(cf.cell_type, crate::cell::CellType::Material(_)) { continue; }
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
