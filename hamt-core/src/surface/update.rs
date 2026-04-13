use crate::{
    boundary::kelvin_to_celsius, parallel::{par_or_seq, par_or_seq_mut}, physics::ECLIPSE, uom::{
        Quantity, constants::{WATER_LATENT_HEAT, WATER_SPECIFIC_HEAT}, dimensions::{
            HeatTransferCoeff, LiquidTransportCoeff, MassFlux, MassResistance, Power, ThermalConductance, VaporConductance, VaporTransportCoeff
        }, units::{
            dimensionless, k_per_w, kelvin, kg_per_m2_s_pa, kg_per_m3, kg_per_s, kilograms, m2_per_s, pa_s_per_kg, pascals, s_per_kg, seconds, w_per_k, w_per_m2_k, watts
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
                            let t_s = interior_mat_temp.unwrap_or(bc.temp_inside_mean_radiant);
                            let t_m_k = (t_s + bc.temp_inside_mean_radiant) * 0.5;
                            let t_m_cubed = t_m_k.value_in(kelvin).powi(3);
                            let h_rad =
                                4.0 * physics::SIGMA * physics::EPSILON * t_m_cubed * w_per_m2_k;
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

                    // ==========================================
                    // 1. Heat Calculation (Calculate T_next)
                    // ==========================================
                    let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
                    let mut torsum: Power<f64> = 0.0 * watts;
                    let mut vapor_diffusion_net_flux: MassFlux<f64> = 0.0 * kilograms / seconds;

                    let curr_vp_pa = cell_curr.cell_vp;

                    for nbr_i in surface_curr.neighbor_indices(i, nx) {
                        let neighbor_curr = &surface_curr.cells[[nbr_i, 0, 0]];

                        let (therm_resistance, vapor_resistance_coeff) =
                            Self::calculate_resistance_pair(
                                cell_curr,
                                neighbor_curr,
                                bc.pressure_air,
                            );

                        if therm_resistance > 1e-12 * k_per_w {
                            let nbr_temp_k = neighbor_curr.cell_temp;
                            oorsum += 1.0 / therm_resistance;
                            torsum += nbr_temp_k / therm_resistance;
                        }

                        if vapor_resistance_coeff > 1e-12 * pa_s_per_kg {
                            let nbr_vp_pa = neighbor_curr.cell_vp;
                            vapor_diffusion_net_flux +=
                                (nbr_vp_pa - curr_vp_pa) / vapor_resistance_coeff;
                        }
                    }

                    let density = cell_curr.density;
                    let spech = cell_curr.spech;
                    let volumetric_water = cell_curr.volumetric_water;
                    let volume = cell_curr.volume;

                    let tcap_j_per_k =
                        (density * spech + volumetric_water * WATER_SPECIFIC_HEAT) * volume;

                    let mut q_vapor: Power<f64> = 0.0 * watts;
                    if is_material && settings.latent_heat_enabled {
                        q_vapor = vapor_diffusion_net_flux * WATER_LATENT_HEAT;

                        if q_vapor.abs() > settings.max_latent_heat {
                            q_vapor = q_vapor.value_in(watts).signum() * settings.max_latent_heat;
                        }
                    }

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

                    // Deposit absorbed radiation (QABSI + QABSILW - QWG from TRNBuild) into the
                    // boundary-adjacent material cell, not the massless Virtual cell.
                    // Real surfaces absorb shortwave/longwave flux in the outermost material layer;
                    // placing it in the Virtual cell had no thermal effect (zero mass, zero storage).
                    let q_boundary_energy: Power<f64> =
                        if Some(i) == surface_curr.interior_surface_material_idx {
                            bc.additional_energy_int
                        } else if Some(i) == surface_curr.exterior_surface_material_idx {
                            bc.additional_energy_ext
                        } else {
                            0.0 * watts
                        };

                    let denom_t = oorsum + (tcap_j_per_k / dt_s);

                    let t_next = if denom_t > 1e-10 * w_per_k {
                        let prev_temp_k = cell_prev.cell_temp;
                        let inertial_term: Power<f64> = tcap_j_per_k * prev_temp_k / dt_s;
                        let numer = torsum + q_vapor + q_adds + q_boundary_energy + inertial_term;
                        numer / denom_t
                    } else {
                        cell_curr.cell_temp
                    };
                    let t_next = if t_next.value_in(kelvin).is_finite() {
                        t_next
                    } else {
                        cell_curr.cell_temp
                    };

                    // ==========================================
                    // 2. Moisture Calculation (Calculate RH_next)
                    // ==========================================

                    let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s; // sum(1/R_liquid)
                    let mut phiorsum: MassFlux<f64> = 0.0 * kg_per_s; // sum(phi_nb/R_liquid)
                    let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals; // sum(1/R_vapor)
                    let mut vporsum: MassFlux<f64> = 0.0 * kg_per_s; // sum(P_v_nb/R_vapor)


                    // Taylor Series Linearization for Vapor Pressure:
                    // P_v^{n+1} \approx P_v^n + \left. \frac{\partial P_v}{\partial \phi} \right|^n (\phi^{n+1} - \phi^n) + \left. \frac{\partial P_v}{\partial T} \right|^n (T^{n+1} - T^n)
                    // gives:
                    // P_v^{n+1} \approx P_{sat}(T^n) \cdot \phi^{n+1} + \phi^n \cdot \frac{d P_{sat}}{d T}(T^n) \cdot (T^{n+1} - T^n)

                    let t_curr = cell_curr.cell_temp;
                    let t_curr_c = kelvin_to_celsius(t_curr);
                    let vpsat_curr = physics::psy_psat_fn_temp(t_curr);
                    // Approximate derivative using Magnus formula: dP_sat/dT ≈ P_sat * 17.27 * 237.3 / (T + 237.3)^2
                    let denom_magnus = t_curr_c + 237.3;
                    let d_psat_d_t = if denom_magnus > ECLIPSE {
                        vpsat_curr * 17.27 * 237.3 / (denom_magnus * denom_magnus)
                    } else {
                        0.0 * pascals
                    };

                    let delta_t_val = (t_next - t_curr).value_in(kelvin);

                    // Correction term for P_v due to temperature change: phi_curr * dPsat/dT * dt
                    let p_sat_thermal_correction = cell_curr.cell_rh.value() * d_psat_d_t * delta_t_val;

                    const RELAXATION_FACTOR: f64 = 1.0; // Empirical factor to prevent overshooting in RH update
                    let vpsat_old = physics::psy_psat_fn_temp(cell_prev.cell_temp);
                    let vpsat_pa_val = vpsat_curr * RELAXATION_FACTOR + vpsat_old * (1.0 - RELAXATION_FACTOR);

                    for nbr_i in surface_curr.neighbor_indices(i, nx) {
                        let nbr_curr = &surface_curr.cells[[nbr_i, 0, 0]];
                        let (_, vaporr) =
                        // Should be vaporr1 + vaporr2
                            Self::calculate_resistance_pair(cell_curr, nbr_curr, bc.pressure_air);

                        if vaporr > 1e-13 * pa_s_per_kg {
                            let vapor_conductance = 1.0 / vaporr;
                            vpoosum += vapor_conductance;

                            vporsum += nbr_curr.cell_vp * vapor_conductance;
                        }

                        let overlap_area = cell_curr
                            .size
                            .area_along(0)
                            .min(nbr_curr.size.area_along(0));

                        let dw1: LiquidTransportCoeff<f64> = cell_curr.dw;
                        let dwd_phi1: Density<f64> = cell_curr.dwd_phi;
                        let dw2: LiquidTransportCoeff<f64> = nbr_curr.dw;
                        let dwd_phi2: Density<f64> = nbr_curr.dwd_phi;

                        let rh_r1: MassResistance<f64> =
                            if dw1 > 1e-20 * m2_per_s && dwd_phi1 > 1e-20 * kg_per_m3 {
                                (cell_curr.size.x().abs() * 0.5) / (overlap_area * dw1 * dwd_phi1)
                            } else {
                                0.0f64 * s_per_kg
                            };

                        let rh_r2: MassResistance<f64> = if dw2 > 1e-20 * m2_per_s && dwd_phi2 > 1e-20 * kg_per_m3 {
                            (nbr_curr.size.x().abs() * 0.5) / (overlap_area * dw2 * dwd_phi2)
                        } else {
                            0.0f64 * s_per_kg
                        };

                        let r_total: Quantity<f64, _> = (rh_r1 + rh_r2) * dimensionless;

                        if r_total < f64::MAX * dimensionless / kg_per_s && r_total > 1e-13 * dimensionless / kg_per_s {
                            let conductance: MassFlux<f64> = 1.0 / r_total;
                            phioosum += conductance;
                            phiorsum += nbr_curr.cell_rh * conductance;
                        }
                    }

                    // Moisture Capacity
                    let wcap:Mass<f64> = if cell_curr.dwd_phi > 1e-10 * kg_per_m3 {
                        cell_curr.dwd_phi * cell_curr.volume
                    } else {
                        1.0e-10 * kilograms // 防止除零
                    };

                    // RH_next
                    // P_v_next = P_sat(T_next) * RH_next
                    // Eq: (phioosum + vpoosum * Psat) * RH_next = phiorsum + vporsum + (wcap/dt)*RH_prev

                    let inertia_term = (wcap * cell_prev.cell_rh.value()) / dt_s;
                    // subtract the thermal correction term from LHS P_v expansion
                    let thermal_source_term = vpoosum * p_sat_thermal_correction;
                    let denom_rh:MassFlux<f64> = phioosum + (vpoosum * vpsat_pa_val) + (wcap / dt_s);

                    let rh_next = if denom_rh > 1e-20 * kg_per_s {
                        (phiorsum + vporsum + inertia_term - thermal_source_term) / denom_rh
                    } else {
                        cell_curr.cell_rh
                    };
                    let rh_next = if rh_next.value().is_finite() {
                        rh_next
                    } else {
                        cell_curr.cell_rh
                    };

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
