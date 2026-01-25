use crate::{
    parallel::{par_or_seq, par_or_seq_mut},
    physics::ECLIPSE,
    uom::{
        Quantity,
        dimensions::{
            Area, HeatTransferCoeff, Power, Pressure, ThermalConductivity, ThermalResistance,
            VaporResistance, VaporTransportCoeff,
        },
        units::{k_per_w, kg_per_m2_s_pa, pa_s_per_kg, watts},
    },
};

use super::*;
use crate::uom::units;
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
        let dt_s_val = dt_s.value_in(units::seconds);
        let nx = self.cells.raw_dim()[0];
        let indices: Vec<usize> = (0..nx).collect();

        let results: Vec<(Temperature<f64>, Dimensionless<f64>)> = par_or_seq(&indices)
            .map(|&i| {
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                if matches!(
                    cell_curr.cell_type,
                    CellType::Material(_) | CellType::Boundary(_, _)
                ) {
                    let cell_prev = &surface_prev.cells[[i, 0, 0]];

                    // ==========================================
                    // 1. Heat Calculation (Calculate T_next)
                    // ==========================================

                    let mut torsum = Quantity(0.0);
                    let mut oorsum = Quantity(0.0);
                    let mut vpdiff = Quantity(0.0);

                    // Fix C: Use current iteration vapor pressure for latent heat (Implicit)
                    let curr_vp_pa = cell_curr.cell_vp;

                    for nbr_i in surface_curr.neighbor_indices(i, nx) {
                        let neighbor_curr = &surface_curr.cells[[nbr_i, 0, 0]];

                        let (therm_resistance, vapor_resistance) = Self::calculate_resistance_pair(
                            cell_curr,
                            neighbor_curr,
                            bc.pressure_air,
                        );

                        if therm_resistance > 0.0 * k_per_w {
                            let nbr_temp_k = neighbor_curr.cell_temp;
                            oorsum += 1.0 / therm_resistance;
                            torsum += nbr_temp_k / therm_resistance;
                        }
                        if vapor_resistance > 0.0 * pa_s_per_kg {
                            // Fix C: Use current iteration vapor pressure
                            let nbr_vp_pa = neighbor_curr.cell_vp;
                            vpdiff += (nbr_vp_pa - curr_vp_pa) / vapor_resistance;
                        }
                    }

                    let density = cell_curr.density;
                    let spech = cell_curr.spech;
                    let volumetric_water = cell_curr.volumetric_water;
                    let volume = cell_curr.volume;

                    let tcap_j_per_k =
                        (density * spech + volumetric_water * physics::W_SPECH) * volume;

                    let mut qvp = Quantity(0.0);
                    if settings.latent_heat_enabled
                        && matches!(cell_curr.cell_type, CellType::Material(_))
                    {
                        qvp = vpdiff * physics::W_HV;
                        if qvp.abs() > settings.max_latent_heat {
                            qvp = Quantity(0.0);
                        }
                    }

                    let q_adds = match cell_curr.cell_type {
                        CellType::Virtual(BoundaryLocation::Exterior) => bc.additional_energy_ext,
                        CellType::Virtual(BoundaryLocation::Interior) => bc.additional_energy_int,
                        CellType::Material(_) => {
                            if surface_curr
                                .user_defined_cell_idx
                                .map(|idx| idx[0] == i)
                                .unwrap_or(false)
                            {
                                bc.q_addition_userdef
                            } else {
                                0.0 * watts
                            }
                        }
                        _ => 0.0 * watts,
                    };

                    let denom_t = oorsum + (tcap_j_per_k / dt_s);
                    let t_next = if denom_t > Quantity(ECLIPSE) {
                        let prev_temp_k = cell_prev.cell_temp;
                        let other: Power<f64> = tcap_j_per_k * prev_temp_k / dt_s;
                        let numer = torsum + qvp + q_adds + other;
                        numer / denom_t
                    } else {
                        cell_curr.cell_temp
                    };

                    // ==========================================
                    // 2. Moisture Calculation (Calculate RH_next)
                    // ==========================================

                    let mut phioosum = 0.0;
                    let mut phiorsum = 0.0;
                    let mut vpoosum = 0.0;
                    let mut vporsum = 0.0;

                    // Fix B: Use the NEW temperature for Psat calculation (Tight Coupling)
                    let vpsat_pa = physics::psy_psat_fn_temp(t_next).value_in(units::pascals);

                    for nbr_i in surface_curr.neighbor_indices(i, nx) {
                        let nbr_curr = &surface_curr.cells[[nbr_i, 0, 0]];

                        // Re-calculate resistances as needed by physics (simplification: reuse same state as heat step)
                        let (_, vaporr) =
                            Self::calculate_resistance_pair(cell_curr, nbr_curr, bc.pressure_air);
                        let vaporr_ps_per_kg = vaporr.value_in(crate::uom::units::pa_s_per_kg);

                        if vaporr_ps_per_kg > 0.0 {
                            vpoosum += 1.0 / vaporr_ps_per_kg;
                            let nbr_vp_pa = (nbr_curr.cell_rh
                                * physics::psy_psat_fn_temp(nbr_curr.cell_temp))
                            .value_in(units::pascals);
                            vporsum += nbr_vp_pa / vaporr_ps_per_kg;
                        }

                        let overlap: Area<f64> = Quantity(
                            cell_curr
                                .size
                                .area_along(0)
                                .value_in(units::square_meter)
                                .min(nbr_curr.size.area_along(0).value_in(units::square_meter)),
                        );
                        let dist1 = cell_curr.size.x().abs() * 0.5;
                        let dist2 = nbr_curr.size.x().abs() * 0.5;

                        let overlap_m2 = overlap.value_in(crate::uom::units::square_meter);
                        let dist1_m = dist1.value_in(crate::uom::units::meter);
                        let dist2_m = dist2.value_in(crate::uom::units::meter);

                        let dw1 = cell_curr.dw.value_in(crate::uom::units::m2_per_s);
                        let dwd_phi1 = cell_curr.dwd_phi.value_in(crate::uom::units::kg_per_m3);

                        let rhr1 = if dw1 > 0.0 && dwd_phi1 > 0.0 {
                            dist1_m / (overlap_m2 * dw1 * dwd_phi1)
                        } else {
                            0.0
                        };

                        let dw2 = nbr_curr.dw.value_in(crate::uom::units::m2_per_s);
                        let dwd_phi2 = nbr_curr.dwd_phi.value_in(crate::uom::units::kg_per_m3);
                        let rhr2 = if dw2 > 0.0 && dwd_phi2 > 0.0 {
                            dist2_m / (overlap_m2 * dw2 * dwd_phi2)
                        } else {
                            0.0
                        };

                        if rhr1 > 0.0 && rhr2 > 0.0 {
                            phioosum += 1.0 / (rhr1 + rhr2);
                            phiorsum += nbr_curr.cell_rh.value() / (rhr1 + rhr2);
                        }
                    }

                    let dwd_phi = cell_curr.dwd_phi.value_in(crate::uom::units::kg_per_m3);
                    let volume_m3 = cell_curr.volume.value_in(crate::uom::units::cubic_meter);
                    let wcap = if dwd_phi > 0.0 {
                        dwd_phi * volume_m3
                    } else {
                        0.0
                    };

                    let denom_rh = phioosum + vpoosum * vpsat_pa + wcap / dt_s_val;

                    let rh_next = if denom_rh > 0.0 {
                        Quantity(
                            (phiorsum + vporsum + (wcap * cell_prev.cell_rh.value()) / dt_s_val)
                                / denom_rh,
                        )
                    } else {
                        cell_curr.cell_rh
                    };

                    (t_next, rh_next)
                } else {
                    // Boundary Logic
                    if cell_curr.is_exterior() {
                        let t = bc.temp_outside_air;
                        let rho = physics::vd_from_t_rh(t, bc.rh_outside_air);
                        (t, physics::rh_from_t_vd(t, rho))
                    } else {
                        let t = bc.temp_inside_air;
                        let rho = physics::vd_from_t_rh(t, bc.rh_inside_air);
                        (t, physics::rh_from_t_vd(t, rho))
                    }
                }
            })
            .collect();

        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).enumerate().for_each(|(i, cell)| {
                cell.cell_temp = results[i].0;
                cell.cell_rh = results[i].1.clamp(Quantity(0.0), Quantity(1.01));
            });
        } else {
            for i in 0..nx {
                self.cells[[i, 0, 0]].cell_temp = results[i].0;
                self.cells[[i, 0, 0]].cell_rh = results[i].1.clamp(Quantity(0.0), Quantity(1.01));
            }
        }
        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
        let self_cell = self.cells.get([i, 0, 0]).unwrap();

        let left: Vec<usize> = if i > 0 { (i - 1..i).collect() } else { vec![] };
        let right = if i + 1 < nx {
            (i + 1..i + 2).collect()
        } else {
            vec![]
        };

        left.into_iter().chain(right)
    }
}
