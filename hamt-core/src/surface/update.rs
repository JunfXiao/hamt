use crate::{
    parallel::{par_or_seq, par_or_seq_mut},
    physics::ECLIPSE,
    uom::{
        self, DIM_POWER, Quantity,
        dimensions::{
            Area, HeatTransferCoeff, LiquidTransportCoeff, MassFlux, MassResistance, Power, Pressure, ThermalConductance, ThermalConductivity, ThermalResistance, VaporConductance, VaporResistance, VaporTransportCoeff
        },
        units::{
            dimensionless, joules, k_per_w, kelvin, kg_per_m2_s_pa, kg_per_m3, kg_per_s, kilograms, m2_per_s, pa_s_per_kg, pascals, s_per_kg, seconds, square_meter, w_per_k, watts
        },
    },
};

use super::*;
use crate::uom::units;
use anyhow::Result;
use diman::si::units::watt;
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
        for i in 0..nx {
            let cell = &mut self.cells[[i, 0, 0]];
            if cell.is_boundary() {
                if cell.is_exterior() {
                    cell.htc = bc.heat_transfer_coeff_ext;
                    cell.cell_temp = bc.temp_outside_air;
                    cell.cell_rh = bc.rh_outside_air;
                    let psat = physics::psy_psat_fn_temp(cell.cell_temp);
                    let rh = cell.cell_rh;
                    let vtc = if bc.vapor_transport_coeff_ext > VaporTransportCoeff::<f64>::zero() {
                        bc.vapor_transport_coeff_ext
                    } else {
                        let rho_v = physics::vd_from_t_rh(cell.cell_temp, rh);
                        let denom = psat * rh;
                        if denom > Quantity(1.0e-9) {
                            bc.mass_convection_coeff_ext * rho_v / denom
                        } else {
                            Quantity(0.0)
                        }
                    };
                    cell.vtc = vtc;
                    cell.cell_vp = rh * psat;
                } else {
                    cell.htc = bc.heat_transfer_coeff_int;
                    cell.cell_temp = bc.temp_inside_air;
                    cell.cell_rh = bc.rh_inside_air;
                    let psat = physics::psy_psat_fn_temp(cell.cell_temp);
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

                if matches!(cell_curr.cell_type, CellType::Material(_)) {
                    let cell_prev = &surface_prev.cells[[i, 0, 0]];

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
                        (density * spech + volumetric_water * physics::W_SPECH) * volume;

                    let mut q_vapor: Power<f64> = 0.0 * watts;
                    if settings.latent_heat_enabled {
                        q_vapor = vapor_diffusion_net_flux * physics::W_HV;

                        if q_vapor.abs() > settings.max_latent_heat {
                            q_vapor = q_vapor.value_in(watts).signum() * settings.max_latent_heat;
                        }
                    }

                    let q_adds = if surface_curr
                        .user_defined_cell_idx
                        .map(|idx| idx[0] == i)
                        .unwrap_or(false)
                    {
                        bc.q_addition_userdef
                    } else {
                        0.0 * watts
                    };

                    let denom_t = oorsum + (tcap_j_per_k / dt_s);

                    let t_next = if denom_t > 1e-10 * w_per_k {
                        let prev_temp_k = cell_prev.cell_temp;
                        let inertial_term: Power<f64> = tcap_j_per_k * prev_temp_k / dt_s;
                        let numer = torsum + q_vapor + q_adds + inertial_term;
                        numer / denom_t
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

                    let vpsat_pa_val = physics::psy_psat_fn_temp(t_next);

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
                    let denom_rh:MassFlux<f64> = phioosum + (vpoosum * vpsat_pa_val) + (wcap / dt_s);

                    let rh_next = if denom_rh > 1e-20 * kg_per_s {
                        (phiorsum + vporsum + inertia_term) / denom_rh
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
                cell.cell_temp = results[i].0;
                cell.cell_rh = results[i].1.clamp(Quantity(0.0), Quantity(1.01));
                cell.calc_dependent_state(bc, settings);
            });
        } else {
            for i in 0..nx {
                self.cells[[i, 0, 0]].cell_temp = results[i].0;
                self.cells[[i, 0, 0]].cell_rh = results[i].1.clamp(Quantity(0.0), Quantity(1.01));
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
