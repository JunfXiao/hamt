use crate::{
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

        // --- Pre-calculate next-step temperatures for all cells ---
        self.calculate_temp_next(settings, bc, surface_prev, surface_curr)?;

        // --- Pre-calculate next-step relative humidity for all cells ---
        self.calculate_rh_next(settings, bc, surface_prev, surface_curr)?;

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        Ok(())
    }

    /// Copy the current iteration state into `surface_next` before property updates.
    pub fn sync_state_from(&mut self, surface_curr: &Surface) -> Result<()> {

        for (cell, ref_cell) in self.cells.iter_mut().zip(surface_curr.cells.iter()) {
            *cell = ref_cell.clone();
            cell.sync_state_from(ref_cell);
        }
        Ok(())
    }

    /// Update dependent cell properties based on the current iteration's Temp and RH.
    pub fn update_cell_properties(
        &mut self,
        bc: &BoundaryCondition,
        settings: &HamtSettings,
    ) -> Result<()> {
        for cell in self.cells.iter_mut() {
            cell.calc_dependent_state(bc, settings);
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
                if cell.is_exterior_boundary() {
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

    /// Heat step: compute next temperatures for all cells.
    fn calculate_temp_next(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        let dt_s = settings.timestep;
        let nx = self.cells.raw_dim()[0];
        for i in 0..nx {
            if self.cells[[i, 0, 0]].is_boundary() {
                let bc_temp = if self.cells[[i, 0, 0]].is_exterior_boundary() {
                    bc.temp_outside_air
                } else {
                    bc.temp_inside_air
                };
                self.cells[[i, 0, 0]].cell_temp = bc_temp;
                continue;
            }

            let t_new = {
                // use scalar Kelvin for calculation
                let cell_next = &self.cells[[i, 0, 0]];
                let cell_prev = &surface_prev.cells[[i, 0, 0]];
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                let mut torsum = Quantity(0.0);
                let mut oorsum = Quantity(0.0);
                let mut vpdiff = Quantity(0.0);

                // let curr_temp_k = cell_curr.cell_temp;
                let curr_vp_pa = cell_prev.cell_vp;

                for nbr_i in self.neighbor_indices(i, nx) {
                    let neighbor_next = &self.cells[[nbr_i, 0, 0]];
                    let neighbor_curr = &self.cells[[nbr_i, 0, 0]];
                    let neighbor_prev = &surface_prev.cells[[nbr_i, 0, 0]];

                    let (therm_resistance, vapor_resistance) =
                        self.calculate_resistance_pair(cell_next, neighbor_next, bc.pressure_air);

                    if therm_resistance > 0.0 * k_per_w {
                        let nbr_temp_k = neighbor_curr.cell_temp;
                        oorsum += 1.0 / therm_resistance;
                        torsum += nbr_temp_k / therm_resistance;
                    }
                    if vapor_resistance > 0.0 * pa_s_per_kg {
                        let nbr_vp_pa = neighbor_prev.cell_vp;
                        vpdiff += (nbr_vp_pa - curr_vp_pa) / vapor_resistance;
                    }
                }

                let density = cell_next.density;
                let spech = cell_next.spech;
                let volumetric_water = cell_next.volumetric_water;
                let volume = cell_next.volume;

                let tcap_j_per_k = (density * spech + volumetric_water * physics::W_SPECH) * volume;

                let mut qvp = Quantity(0.0);
                if settings.latent_heat_enabled
                    && let CellType::Material(_) = cell_next.cell_type
                {
                    qvp = vpdiff * physics::W_HV;
                    if qvp.abs() > settings.max_latent_heat {
                        qvp = Quantity(0.0);
                    }
                }

                let q_adds = match cell_next.cell_type {
                    CellType::Virtual(BoundaryLocation::Exterior) => bc.additional_energy_ext,
                    CellType::Virtual(BoundaryLocation::Interior) => bc.additional_energy_int,
                    CellType::Material(_) => {
                        if self
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

                let denom = oorsum + (tcap_j_per_k / dt_s);
                if denom > Quantity(ECLIPSE) {
                    let prev_temp_k = cell_prev.cell_temp;
                    let other: Power<f64> = tcap_j_per_k * prev_temp_k / dt_s;
                    let numer = torsum + qvp + q_adds + other;
                    numer / denom
                } else {
                    cell_curr.cell_temp
                }
            };

            self.cells[[i, 0, 0]].cell_temp = t_new;
        }
        Ok(())
    }

    /// Moisture step: compute next RH for all cells.
    fn calculate_rh_next(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        let nx = self.cells.raw_dim()[0];
        let dt_s = settings.timestep.value_in(units::seconds);
        for i in 0..nx {
            if self.cells[[i, 0, 0]].is_boundary() {
                let bc_rh = if self.cells[[i, 0, 0]].is_exterior_boundary() {
                    let rho_vapor = physics::vd_from_t_rh(bc.temp_outside_air, bc.rh_outside_air);
                    physics::rh_from_t_vd(bc.temp_outside_air, rho_vapor)
                } else {
                    let rho_vapor = physics::vd_from_t_rh(bc.temp_inside_air, bc.rh_inside_air);
                    physics::rh_from_t_vd(bc.temp_inside_air, rho_vapor)
                };
                self.cells[[i, 0, 0]].cell_rh = bc_rh;
                continue;
            }

            let rh_new = {
                // cell_props: writable buffer (surface_next) properties (dw, dwd_phi, mu, etc.) and T
                let cell_props = &self.cells[[i, 0, 0]];
                // cell_prev_ts: last converged timestep (t-1) RH
                let cell_prev_ts = &surface_prev.cells[[i, 0, 0]];
                // cell_curr: current iteration RH/VP/T
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                // phi one over sum
                // phioosum: Σ(1 / R_liq) liquid transport conductance (Fortran: phioosum)
                // Unit: kg/(m²·s)
                let mut phioosum = 0.0;

                // phi over resistance sum
                // phiorsum: Σ(φ_neighbor / R_liq) neighbor RH weighted by liquid conductance
                // Unit: kg/(m²·s)
                let mut phiorsum = 0.0;

                // vapor pressure one over sum
                // vioosum: Σ(1 / R_v) vapor transport conductance (Fortran: vioosum)
                // Unit: kg/(m²·s·Pa)
                let mut vpoosum = 0.0;

                // vapor pressure over resistance sum
                // vporsum: Σ(VP_neighbor / R_v) vapor pressure driving term
                // Units: kg/(m²·s)
                let mut vporsum = 0.0;

                // vpsat: saturation vapor pressure at updated temperature (Fortran: vpsat)
                let vpsat_pa =
                    physics::psy_psat_fn_temp(cell_props.cell_temp).value_in(units::pascals);

                for nbr_i in self.neighbor_indices(i, nx) {
                    let nbr_props = &self.cells[[nbr_i, 0, 0]];
                    let nbr_curr = &self.cells[[nbr_i, 0, 0]];

                    // overlap: shared face area (Fortran: overlap)
                    let overlap: Area<f64> = Quantity(
                        cell_props
                            .size
                            .area_along(0)
                            .value_in(units::square_meter)
                            .min(nbr_props.size.area_along(0).value_in(units::square_meter)),
                    );

                    // dist: each cell's half-length to the shared face (Fortran: dist)
                    let dist1 = cell_props.size.x().abs() * 0.5;
                    let dist2 = nbr_props.size.x().abs() * 0.5;

                    let (_, vaporr) =
                        self.calculate_resistance_pair(cell_props, nbr_props, bc.pressure_air);

                    let vaporr_ps_per_kg = vaporr.value_in(crate::uom::units::pa_s_per_kg);

                    if vaporr_ps_per_kg > 0.0 {
                        vpoosum += 1.0 / vaporr_ps_per_kg;

                        let nbr_vp_pa = (nbr_curr.cell_rh
                            * physics::psy_psat_fn_temp(nbr_curr.cell_temp))
                        .value_in(units::pascals);

                        vporsum += nbr_vp_pa / vaporr_ps_per_kg;
                    }

                    // rhr1/rhr2: liquid transport resistances for each cell (Fortran: rhr1/rhr2)
                    let overlap_m2 = overlap.value_in(crate::uom::units::square_meter);
                    let dist1_m = dist1.value_in(crate::uom::units::meter);
                    let dist2_m = dist2.value_in(crate::uom::units::meter);

                    let dw1 = cell_props.dw.value_in(crate::uom::units::m2_per_s);
                    let dwd_phi1 = cell_props.dwd_phi.value_in(crate::uom::units::kg_per_m3);
                    let rhr1 = if dw1 > 0.0 && dwd_phi1 > 0.0 {
                        dist1_m / (overlap_m2 * dw1 * dwd_phi1)
                    } else {
                        0.0
                    };

                    let dw2 = nbr_props.dw.value_in(crate::uom::units::m2_per_s);
                    let dwd_phi2 = nbr_props.dwd_phi.value_in(crate::uom::units::kg_per_m3);
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

                // Water Capacity
                // wcap: storage term dθ/dφ * volume (Fortran: wcap)
                // Unit: kg
                let dwd_phi = cell_props.dwd_phi.value_in(crate::uom::units::kg_per_m3);
                let volume = cell_props.volume.value_in(crate::uom::units::cubic_meter);
                let wcap = if dwd_phi > 0.0 { dwd_phi * volume } else { 0.0 };

                // (moisture):
                // denom: Σ(1/R_liq) + Σ(1/R_v)*vpsat + wcap/dt (Fortran: denom)
                // phioosum + vpoosum·vpsat + wcap/d
                // kg/(m²·s)
                let denom = phioosum + vpoosum * vpsat_pa + wcap / dt_s;

                if denom > 0.0 {
                    // numerator: liquid + vapor driving + storage from previous timestep
                    (phiorsum + vporsum + (wcap * cell_prev_ts.cell_rh.value()) / dt_s) / denom
                } else {
                    cell_curr.cell_rh.value()
                }
            };

            self.cells[[i, 0, 0]].cell_rh = Quantity(rh_new).clamp(Quantity(0.0), Quantity(1.01));
        }

        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
        let self_cell = self.cells.get([i, 0, 0]).unwrap();

        let left: Vec<usize> = if matches!(self_cell.cell_type, CellType::Boundary(_, _)) {
            (0..i).collect()
        } else if i > 0 {
            (i - 1..i).collect()
        } else {
            vec![]
        };

        let right = if matches!(self_cell.cell_type, CellType::Boundary(_, _)) {
            (i + 1..nx).collect()
        } else if i + 1 < nx {
            (i + 1..i + 2).collect()
        } else {
            vec![]
        };

        left.into_iter().chain(right)
    }

    /// Calculate thermal and vapor resistance between two cells.
    ///
    /// # Returns
    /// A tuple containing the thermal and vapor resistances.
    /// - Thermal resistance [m²K/W] // TODO Check Unit
    /// - Vapor resistance [m²Pa/W] // TODO Check Unit
    fn calculate_resistance_pair(
        &self,
        cell1: &Cell,
        cell2: &Cell,
        pressure_air: Pressure<f64>,
    ) -> (ThermalResistance<f64>, VaporResistance<f64>) {
        let overlap: Area<f64> = Quantity(
            cell1
                .size
                .area_along(0)
                .value_unchecked()
                .min(cell2.size.area_along(0).value_unchecked()),
        );
        let dist1 = cell1.size.x().abs() * 0.5;
        let dist2 = cell2.size.x().abs() * 0.5;

        if overlap <= Quantity(ECLIPSE) || dist1 <= Quantity(ECLIPSE) || dist2 <= Quantity(ECLIPSE)
        {
            tracing::warn!(
                "calc_resistance_pair: degenerate geometry overlap={:?}, dist1={:?}, dist2={:?}, cell1={:?}, cell2={:?}",
                overlap,
                dist1,
                dist2,
                cell1.cell_type,
                cell2.cell_type
            );
            return (Quantity(0.0), Quantity(0.0));
        }

        let htc1 = cell1.htc;
        let k1 = cell1.w_thermal_c;
        let thermr1: ThermalResistance<f64> = if htc1 > HeatTransferCoeff::<f64>::zero() {
            1.0 / (htc1 * overlap)
        } else if k1 > ThermalConductivity::<f64>::zero() {
            dist1 / (k1 * overlap)
        } else {
            Quantity(0.0)
        };
        let htc2 = cell2.htc;
        let k2 = cell2.w_thermal_c;
        let thermr2: ThermalResistance<f64> = if htc2 > HeatTransferCoeff::<f64>::zero() {
            1.0 / (htc2 * overlap)
        } else if k2 > ThermalConductivity::<f64>::zero() {
            dist2 / (k2 * overlap)
        } else {
            Quantity(0.0)
        };

        let vtc1 = cell1.vtc;
        let vaporr1: VaporResistance<f64> = if vtc1 > VaporTransportCoeff::<f64>::zero() {
            1.0 / (overlap * vtc1)
        } else if cell1.mu > Quantity(ECLIPSE) {
            (dist1 * cell1.mu) / (overlap * physics::wvdc(cell1.cell_temp, pressure_air))
        } else {
            Quantity(0.0)
        };
        let vtc2 = cell2.vtc;
        let vaporr2: VaporResistance<f64> = if vtc2 > VaporTransportCoeff::<f64>::zero() {
            1.0 / (overlap * vtc2)
        } else if cell2.mu > Quantity(ECLIPSE) {
            (dist2 * cell2.mu) / (overlap * physics::wvdc(cell2.cell_temp, pressure_air))
        } else {
            Quantity(0.0)
        };

        (thermr1 + thermr2, vaporr1 + vaporr2)
    }
}
