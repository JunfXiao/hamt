use crate::{
    parallel::{par_or_seq, par_or_seq_mut}, uom::{
        Quantity, constants::{WATER_LATENT_HEAT, WATER_SPECIFIC_HEAT}, dimensions::{
            HeatTransferCoeff, MassFlux, MassResistance, Power, ThermalConductance, VaporConductance, VaporTransportCoeff
        }, units::{
            dimensionless, j_per_k, j_per_kg, k_per_w, kelvin, kg_per_m2_s_pa, kg_per_m3, kg_per_s,
            kilograms, m2_per_s, pa_s_per_kg, pascals, s_per_kg, seconds, w_per_k, w_per_m2_k, watts
        }
    }
};

use super::*;

use anyhow::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;


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

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        // --- Combined Heat and Moisture Step ---
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
    fn apply_boundary_coefficients(
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
                            // Pure convective HTC (HCONVI). Paired with TAIR on the heat side;
                            // TAIR is also used for VP_bc on the moisture side.
                            cell.htc = bc.heat_transfer_coeff_conv_int;
                            cell.cell_temp = bc.temp_inside_air;
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

                        let (therm_r, vapor_r) =
                            Self::calculate_resistance_pair(cell_curr, nbr_curr, bc.pressure_air);

                        if therm_r > 1e-12 * k_per_w {
                            oorsum += 1.0 / therm_r;
                            torsum += nbr_curr.cell_temp / therm_r;
                        }

                        if vapor_r > 1e-12 * pa_s_per_kg {
                            let nbr_vp = nbr_curr.cell_vp;
                            let vc = 1.0 / vapor_r;
                            vapor_diffusion_net_flux += (nbr_vp - curr_vp_pa) / vapor_r;
                            vpoosum += vc;
                            vporsum += nbr_vp * vc;
                        }

                        let overlap_area = cell_curr.size.area_along(0).min(nbr_curr.size.area_along(0));
                        let rh_r1: MassResistance<f64> =
                            if cell_curr.dw > 1e-20 * m2_per_s && cell_curr.dwd_phi > 1e-20 * kg_per_m3 {
                                (cell_curr.size.x().abs() * 0.5)
                                    / (overlap_area * cell_curr.dw * cell_curr.dwd_phi)
                            } else {
                                0.0f64 * s_per_kg
                            };
                        let rh_r2: MassResistance<f64> =
                            if nbr_curr.dw > 1e-20 * m2_per_s && nbr_curr.dwd_phi > 1e-20 * kg_per_m3 {
                                (nbr_curr.size.x().abs() * 0.5)
                                    / (overlap_area * nbr_curr.dw * nbr_curr.dwd_phi)
                            } else {
                                0.0f64 * s_per_kg
                            };
                        let r_total: Quantity<f64, _> = (rh_r1 + rh_r2) * dimensionless;
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
                        // ── Symmetric 2×2 block solve ─────────────────────────────────────────
                        //
                        // Linearise P_v,i^{k+1} ≈ Psat(T^k)·φ^{k+1} + φ^k·(dPsat/dT)·(T^{k+1}−T^k)
                        //
                        // System (all quantities in SI, raw f64):
                        //
                        //  ┌ A_tt  A_tp ┐ ┌ T_next ┐   ┌ b_t ┐
                        //  └ A_pt  A_pp ┘ └ φ_next ┘ = └ b_p ┘
                        //
                        // where:
                        //   A_tt = oorsum + tcap/dt + L_eff·vpoosum·φ_k·(dPsat/dT)  [W/K]
                        //   A_tp = L_eff·vpoosum·Psat                                [W]
                        //   A_pt = vpoosum·φ_k·(dPsat/dT)                           [kg/(s·K)]
                        //   A_pp = phioosum + vpoosum·Psat + wcap/dt                 [kg/s]
                        //   b_t  = torsum + q_adds + q_bnd + (tcap/dt)·T_prev
                        //          + L_eff·(vporsum + A_pt·T_k)                      [W]
                        //   b_p  = phiorsum + vporsum + (wcap/dt)·φ_prev + A_pt·T_k [kg/s]
                        //
                        // L_eff = L_v + H_diff(φ) — see uom.rs constant WATER_LATENT_HEAT
                        // (2,489,000 J/kg) plus the optional per-material sorption enthalpy.

                        // L_eff [J/kg] as a raw scalar
                        // WATER_LATENT_HEAT = 2_489_000 J/kg (from uom::constants)
                        let l_eff_f: f64 = 2_489_000.0
                            + if settings.sorption_enthalpy_enabled {
                                cell_curr.h_sorp_diff.value_in(j_per_kg)
                            } else {
                                0.0
                            };

                        let oorsum_f  = oorsum.value_in(w_per_k);
                        let torsum_f  = torsum.value_in(watts);
                        let vpoosum_f = vpoosum.value_in(kg_per_s / pascals);
                        let vporsum_f = vporsum.value_in(kg_per_s);
                        let phioosum_f = phioosum.value_in(kg_per_s);
                        let phiorsum_f = phiorsum.value_in(kg_per_s);
                        let tcap_f    = tcap_j_per_k.value_in(j_per_k);
                        let wcap_f    = wcap.value_in(kilograms);
                        let dt_f      = dt_s.value_in(seconds);
                        let psat_f    = vpsat_curr.value_in(pascals);
                        let dpsat_f   = d_psat_d_t.value_in(pascals); // Pa/K (as Pa)
                        let phi_k     = cell_curr.cell_rh.value();
                        let phi_prev  = cell_prev.cell_rh.value();
                        let t_k       = t_curr.value_in(kelvin);
                        let t_prev_f  = cell_prev.cell_temp.value_in(kelvin);
                        let q_adds_f  = q_adds.value_in(watts);
                        let q_b_f     = q_boundary_energy.value_in(watts);

                        // cross = vpoosum · φ_k · dPsat/dT  [kg/(s·K)]
                        let cross = vpoosum_f * phi_k * dpsat_f;

                        let a_tt = oorsum_f + tcap_f / dt_f + l_eff_f * cross;
                        let a_tp = l_eff_f * vpoosum_f * psat_f;
                        let a_pt = cross;
                        let a_pp = phioosum_f + vpoosum_f * psat_f + wcap_f / dt_f;

                        let b_t = torsum_f + q_adds_f + q_b_f
                            + (tcap_f / dt_f) * t_prev_f
                            + l_eff_f * (vporsum_f + cross * t_k);
                        let b_p = phiorsum_f + vporsum_f
                            + (wcap_f / dt_f) * phi_prev
                            + cross * t_k;

                        let det = a_tt * a_pp - a_tp * a_pt;

                        let (t_k_next, phi_next) = if det.abs() > 1e-30 {
                            let t_n = (b_t * a_pp - a_tp * b_p) / det;
                            let p_n = (a_tt * b_p - a_pt * b_t) / det;
                            (t_n, p_n)
                        } else {
                            // Near-singular (e.g. no vapor coupling): fall back to current state;
                            // the outer relaxation loop will converge on next iteration.
                            (t_k, phi_k)
                        };

                        t_next = if t_k_next.is_finite() {
                            t_k_next * kelvin
                        } else {
                            t_curr
                        };
                        rh_next = if phi_next.is_finite() {
                            phi_next * dimensionless
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
                                q_vapor =
                                    q_vapor.value_in(watts).signum() * settings.max_latent_heat;
                            }
                        }

                        let denom_t = oorsum + tcap_j_per_k / dt_s;
                        let t_next_s = if denom_t > 1e-10 * w_per_k {
                            let inertial = tcap_j_per_k * cell_prev.cell_temp / dt_s;
                            (torsum + q_vapor + q_adds + q_boundary_energy + inertial) / denom_t
                        } else {
                            t_curr
                        };
                        t_next = if t_next_s.value_in(kelvin).is_finite() {
                            t_next_s
                        } else {
                            t_curr
                        };

                        // Moisture: thermal correction uses the just-computed t_next
                        let delta_t_val = (t_next - t_curr).value_in(kelvin);
                        let p_sat_thermal_correction =
                            cell_curr.cell_rh.value() * d_psat_d_t * delta_t_val;
                        let thermal_source_term = vpoosum * p_sat_thermal_correction;
                        let denom_rh: MassFlux<f64> =
                            phioosum + vpoosum * vpsat_curr + wcap / dt_s;

                        let rh_next_s = if denom_rh > 1e-20 * kg_per_s {
                            (phiorsum + vporsum + (wcap * cell_prev.cell_rh.value()) / dt_s
                                - thermal_source_term)
                                / denom_rh
                        } else {
                            debug!(
                                "Cell {} has very low moisture conductance {:?}, using fallback RH to prevent instability",
                                i, denom_rh
                            );
                            cell_curr.cell_rh
                        };
                        rh_next = if rh_next_s.value().is_finite() {
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
                    if cell_curr.is_exterior() {
                        let t = bc.temp_outside_air;
                        (t, bc.rh_outside_air)
                    } else {
                        let t = bc.temp_inside_air;
                        (t, bc.rh_inside_air)
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
                let alpha = cell.relaxation_factor;

                let relaxed_temp = if next_temp.value_in(kelvin).is_finite() {
                    next_temp * alpha + curr_temp * (1.0 - alpha)
                } else {
                    curr_temp
                };

                let relaxed_rh = if next_rh.value().is_finite() {
                    next_rh * alpha + curr_rh * (1.0 - alpha)
                } else {
                    curr_rh
                };

                cell.cell_temp = relaxed_temp;
                cell.cell_rh = relaxed_rh.clamp(Quantity(0.0), Quantity(1.01));
                cell.calc_dependent_state(bc, settings);
            });
        } else {
            for i in 0..nx {
                let next_temp = results[i].0;
                let next_rh = results[i].1;
                let curr_temp = surface_curr.cells[[i, 0, 0]].cell_temp;
                let curr_rh = surface_curr.cells[[i, 0, 0]].cell_rh;
                let alpha = self.cells[[i, 0, 0]].relaxation_factor;

                let relaxed_temp = if next_temp.value_in(kelvin).is_finite() {
                    next_temp * alpha + curr_temp * (1.0 - alpha)
                } else {
                    curr_temp
                };

                let relaxed_rh = if next_rh.value().is_finite() {
                    next_rh * alpha + curr_rh * (1.0 - alpha)
                } else {
                    curr_rh
                };

                self.cells[[i, 0, 0]].cell_temp = relaxed_temp;
                self.cells[[i, 0, 0]].cell_rh = relaxed_rh.clamp(Quantity(0.0), Quantity(1.01));
                self.cells[[i, 0, 0]].calc_dependent_state(bc, settings);
            }
        }
        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
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
