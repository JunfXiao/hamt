use super::*;
use anyhow::Result;
use rayon::iter::IntoParallelRefMutIterator;

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
        let dt = settings.timestep;
        let dims = self.cells.raw_dim();
        let nx = dims[0];

        // --- Sync `surface_next` with current iteration state ---
        self.sync_state_from(surface_curr)?;

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        // --- Pre-calculate next-step temperatures for all cells ---
        self.calculate_temp_next(settings, surface_prev, surface_curr, dt, nx, bc)?;

        // --- Pre-calculate next-step relative humidity for all cells ---
        self.calculate_rh_next(settings, surface_prev, surface_curr, dt, nx, bc)?;

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        Ok(())
    }

    /// Copy the current iteration state into `surface_next` before property updates.
    fn sync_state_from(&mut self, surface_curr: &Surface) -> Result<()> {
        for (cell, ref_cell) in self.cells.iter_mut().zip(surface_curr.cells.iter()) {
            *cell = ref_cell.clone();
            cell.sync_state_from(ref_cell);
        }
        Ok(())
    }

    /// Update dependent cell properties based on the current iteration's Temp and RH.
    fn update_cell_properties(
        &mut self,
        bc: &BoundaryCondition,
        settings: &HamtSettings,
    ) -> Result<()> {
        for cell in self.cells.iter_mut() {
            cell.calc_dependent_state(bc, settings);
        }
        Ok(())
    }

    /// Heat step: compute next temperatures for all cells.
    fn calculate_temp_next(
        &mut self,
        settings: &HamtSettings,
        surface_prev: &Surface,
        surface_curr: &Surface,
        dt: f64,
        nx: usize,
        bc: &BoundaryCondition,
    ) -> Result<()> {
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
                // cell_next: writable buffer (surface_next)
                let cell_next = &self.cells[[i, 0, 0]];
                // cell_prev: last converged timestep (t-1)
                let cell_prev = &surface_prev.cells[[i, 0, 0]];
                // cell_curr: current iteration state (t, previous iter)
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                // temperature over resistance sum:
                // Σ(T_neighbor / R_th) from previous iteration neighbors (Fortran: torsum)
                // Unit: W/m²
                let mut torsum = 0.0;

                // One over resistance sum:
                // Σ(1 / R_th) over neighbors (Fortran: oorsum)
                // Unit: W/m²K
                let mut oorsum = 0.0;

                // vapor pressure difference weighted by vapor resistance
                // vpdiff: Σ(ΔVP / R_v) latent coupling source term (Fortran: vpdiff)
                // kg/(m²·s
                let mut vpdiff = 0.0;

                for nbr_i in self.neighbor_indices(i, nx) {
                    // neighbor_next: writable buffer props (w_thermal_c/mu) for R calculations
                    let neighbor_next = &self.cells[[nbr_i, 0, 0]];

                    // neighbor_curr: state (T, RH, VP) from previous iteration
                    let neighbor_curr = &surface_curr.cells[[nbr_i, 0, 0]];
                    let (therm_resistance, vapor_resistance) =
                        self.calculate_resistance_pair(cell_next, neighbor_next, bc.pressure_air);

                    if therm_resistance > 0.0 {
                        oorsum += 1.0 / therm_resistance;
                        torsum += neighbor_curr.cell_temp / therm_resistance;
                    }
                    if vapor_resistance > 0.0 {
                        vpdiff += (neighbor_curr.cell_vp - cell_curr.cell_vp) / vapor_resistance;
                    }
                }

                // tcap: thermal capacity term [J/K] = (ρ*c + θ_w * c_w) * volume (Fortran: tcap)
                let tcap = (cell_next.density * cell_next.spech
                    + cell_next.volumetric_water * physics::W_SPECH)
                    * cell_next.volume;

                // qvp: latent source term [W] = Σ(ΔVP/R_v) * H_vap if enabled (Fortran: qvp)
                let mut qvp = 0.0;
                if settings.latent_heat_enabled
                    && let CellType::Material(_) = cell_next.cell_type
                {
                    qvp = vpdiff * physics::W_HV;
                }

                // denom: Σ(1/R_th) + tcap/dt (Fortran: denom)
                let denom = oorsum + (tcap / dt);
                if denom > 0.0 {
                    // numerator: torsum + qvp + internal gains + previous-step storage term
                    (torsum + qvp + cell_next.q_adds + (tcap * cell_prev.cell_temp / dt)) / denom
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
        _settings: &HamtSettings,
        surface_prev: &Surface,
        surface_curr: &Surface,
        dt: f64,
        nx: usize,
        bc: &BoundaryCondition,
    ) -> Result<()> {
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
                let vpsat = physics::psy_psat_fn_temp(cell_props.cell_temp);

                for nbr_i in self.neighbor_indices(i, nx) {
                    let nbr_props = &self.cells[[nbr_i, 0, 0]];
                    let nbr_curr = &surface_curr.cells[[nbr_i, 0, 0]];

                    // overlap: shared face area (Fortran: overlap)
                    let overlap = cell_props
                        .size
                        .area_along(0)
                        .min(nbr_props.size.area_along(0));

                    // dist: center-to-center distance (Fortran: dist)
                    let dist = (cell_props.size.x().abs() + nbr_props.size.x().abs()) * 0.5;

                    let (_, vaporr) =
                        self.calculate_resistance_pair(cell_props, nbr_props, bc.pressure_air);

                    if vaporr > 0.0 {
                        vpoosum += 1.0 / vaporr;

                        let nbr_vp =
                            nbr_curr.cell_rh * physics::psy_psat_fn_temp(nbr_curr.cell_temp);

                        vporsum += nbr_vp / vaporr;
                    }

                    // rhr1/rhr2: liquid transport resistances for each cell (Fortran: rhr1/rhr2)
                    let rhr1 = if cell_props.dw > 0.0 && cell_props.dwd_phi > 0.0 {
                        dist / (overlap * cell_props.dw * cell_props.dwd_phi)
                    } else {
                        0.0
                    };

                    let rhr2 = if nbr_props.dw > 0.0 && nbr_props.dwd_phi > 0.0 {
                        dist / (overlap * nbr_props.dw * nbr_props.dwd_phi)
                    } else {
                        0.0
                    };

                    if rhr1 > 0.0 && rhr2 > 0.0 {
                        phioosum += 1.0 / (rhr1 + rhr2);

                        phiorsum += nbr_curr.cell_rh / (rhr1 + rhr2);
                    }
                }

                // Water Capacity
                // wcap: storage term dθ/dφ * volume (Fortran: wcap)
                // Unit: kg
                let wcap = if cell_props.dwd_phi > 0.0 {
                    cell_props.dwd_phi * cell_props.volume
                } else {
                    0.0
                };

                // (moisture):
                // denom: Σ(1/R_liq) + Σ(1/R_v)*vpsat + wcap/dt (Fortran: denom)
                // phioosum + vpoosum·vpsat + wcap/d
                // kg/(m²·s)
                let denom = phioosum + vpoosum * vpsat + wcap / dt;

                if denom > 0.0 {
                    // numerator: liquid + vapor driving + storage from previous timestep
                    (phiorsum + vporsum + (wcap * cell_prev_ts.cell_rh) / dt) / denom
                } else {
                    cell_curr.cell_rh
                }
            };

            self.cells[[i, 0, 0]].cell_rh = rh_new.clamp(0.0, 1.01);
        }

        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
        // 1. 检查左边 (使用 bool::then_some 生成 Option)
        let left = (i > 0).then_some(i - 1);

        // 2. 检查右边
        let right = (i + 1 < nx).then_some(i + 1);

        // 3. 将两个 Option 链接起来
        // Option 也是一种迭代器（由 0 或 1 个元素组成）
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
        pressure_air: f64,
    ) -> (f64, f64) {
        let overlap = cell1.size.area_along(0).min(cell2.size.area_along(0));
        let dist = (cell1.size.x().abs() + cell2.size.x().abs()) * 0.5;

        let thermr1 = if cell1.htc > 0.0 {
            1.0 / (overlap * cell1.htc)
        } else if cell1.w_thermal_c > 0.0 {
            dist / (overlap * cell1.w_thermal_c)
        } else {
            0.0
        };
        let thermr2 = if cell2.htc > 0.0 {
            1.0 / (overlap * cell2.htc)
        } else if cell2.w_thermal_c > 0.0 {
            dist / (overlap * cell2.w_thermal_c)
        } else {
            0.0
        };

        let vaporr1 = if cell1.vtc > 0.0 {
            1.0 / (overlap * cell1.vtc)
        } else if cell1.mu > 0.0 {
            (dist * cell1.mu) / (overlap * physics::wvdc(cell1.cell_temp, pressure_air))
        } else {
            0.0
        };
        let vaporr2 = if cell2.vtc > 0.0 {
            1.0 / (overlap * cell2.vtc)
        } else if cell2.mu > 0.0 {
            (dist * cell2.mu) / (overlap * physics::wvdc(cell2.cell_temp, pressure_air))
        } else {
            0.0
        };

        (thermr1 + thermr2, vaporr1 + vaporr2)
    }
}
