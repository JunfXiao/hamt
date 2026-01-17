use super::*;
use anyhow::{Context, Result};

impl Surface {
    /// Executes a single-timestep update assuming `self` is the writable `surface_next`.
    /// Reads from `surface_prev` (t-1) and `surface_curr` (current iteration state).
    ///
    /// Returns `true` if no further iterations are needed and the convergence criteria is met.
    pub fn calculate_timestep(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<bool> {
        let surface_prev_cells = surface_prev.cells;
        let surface_curr_cells = surface_curr.cells;

        // TODO: Recheck
        let dt = settings.timestep;
        let dims = self.cells.raw_dim();
        let nx = dims[0];
        assert_eq!(dims[1], 1, "Only 1D surfaces are supported");
        assert_eq!(dims[2], 1, "Only 1D surfaces are supported");

        // 0) Copy current iteration state into the writable buffer
        // TODO: Is this necessary?
        self.override_cell_state(&surface_curr_cells);

        // 1) Apply boundary conditions to boundary cells in the working (next) buffer
        // Exterior
        let exterior_idxs: Vec<[usize; 3]> = self
            .boundary_cells(BoundaryLocation::Exterior)
            .map(|v| *v)
            .collect();
        for idx in exterior_idxs {
            let cell = self.cells.get_mut(idx).with_context(|| {
                format!("Cell index {:?} of surface_next is out of bounds", idx)
            })?;
            cell.cell_temp = bc.temp_outside_air;
            let rho_vapor = physics::vd_from_t_rh(bc.temp_outside_air, bc.rh_outside_air);
            cell.cell_rh = physics::rh_from_t_vd(bc.temp_outside_air, rho_vapor);
        }

        // Interior
        let interior_idxs: Vec<[usize; 3]> = self
            .boundary_cells(BoundaryLocation::Interior)
            .map(|v| *v)
            .collect();
        for idx in interior_idxs {
            let cell = self.cells.get_mut(idx).with_context(|| {
                format!("Cell index {:?} of surface_next is out of bounds", idx)
            })?;
            cell.cell_temp = bc.temp_inside_air;
            let rho_vapor = physics::vd_from_t_rh(bc.temp_inside_air, bc.rh_inside_air);
            cell.cell_rh = physics::rh_from_t_vd(bc.temp_inside_air, rho_vapor);
        }

        // Flattened scratch arrays (previous iteration state for the nonlinear solver)
        let mut temp_prev: Vec<f64> = self.cells.iter().map(|c| c.cell_temp).collect();
        let mut rh_prev: Vec<f64> = self.cells.iter().map(|c| c.cell_rh).collect();

        // Storage for updated values each iteration
        let mut temp_next = vec![0.0f64; self.cells.len()];
        let mut rh_next = vec![0.0f64; self.cells.len()];

        let mut converged = false;

        for _iter in 0..settings.max_iterations {
            // --- Update moisture-dependent properties for material cells using prev state ---
            for (idx, cell) in self.cells.iter_mut().enumerate() {
                let t_prev = temp_prev[idx];
                let rh_p = rh_prev[idx];

                // Saturation vapor pressure and vapor pressure
                cell.vp_sat = physics::psy_psat_fn_temp(t_prev);
                cell.cell_vp = rh_p * cell.vp_sat;

                if let Some(mat) = &cell.material {
                    // Sorption isotherm: water content (kg/m3) and derivative dwdphi
                    let (water, dwdphi) = physics::interp(&mat.sorption_isotherm, rh_p);
                    cell.volumetric_water = water;
                    cell.dwd_phi = dwdphi;

                    // Liquid transport coefficient
                    let (dw, _) = if bc.is_raining && settings.rain_enabled {
                        physics::interp(&mat.suction, water)
                    } else {
                        physics::interp(&mat.redistribution, water)
                    };
                    cell.dw = dw;

                    // Vapor diffusion resistance factor (mu) from RH
                    let (mu, _) = physics::interp(&mat.vapor_diffusion_resistance, rh_p);
                    cell.mu = mu;

                    // Moisture-dependent thermal conductivity
                    let (k, _) = physics::interp(&mat.thermal_conductivity, water);
                    cell.w_thermal_c = if k > 0.0 { k } else { mat.conductivity };
                }
            }

            // --- Heat step: compute next temperatures based on temp_prev/rh_prev ---
            for i in 0..nx {
                let cell = &self.cells[[i, 0, 0]];

                let mut torsum = 0.0;
                let mut oorsum = 0.0;
                let mut vpdiff = 0.0;

                // Neighbors: left (i-1) and right (i+1) along x
                let mut neighbors: Vec<usize> = Vec::with_capacity(2);
                if i > 0 {
                    neighbors.push(i - 1);
                }
                if i + 1 < nx {
                    neighbors.push(i + 1);
                }

                for nbr_i in neighbors {
                    let nbr_cell = &self.cells[[nbr_i, 0, 0]];

                    // Geometric data (simple 1D face overlap)
                    let overlap = cell.size.area_along(0).min(nbr_cell.size.area_along(0));
                    let dist = (cell.size.x().abs() + nbr_cell.size.x().abs()) * 0.5;

                    // Thermal resistances
                    let thermr1 = if cell.htc > 0.0 {
                        1.0 / (overlap * cell.htc)
                    } else if cell.w_thermal_c > 0.0 {
                        dist / (overlap * cell.w_thermal_c)
                    } else {
                        0.0
                    };

                    let thermr2 = if nbr_cell.htc > 0.0 {
                        1.0 / (overlap * nbr_cell.htc)
                    } else if nbr_cell.w_thermal_c > 0.0 {
                        dist / (overlap * nbr_cell.w_thermal_c)
                    } else {
                        0.0
                    };

                    if thermr1 + thermr2 > 0.0 {
                        oorsum += 1.0 / (thermr1 + thermr2);
                        torsum += temp_prev[nbr_i] / (thermr1 + thermr2);
                    }

                    // Vapor resistances for latent coupling
                    let vaporr1 = if cell.vtc > 0.0 {
                        1.0 / (overlap * cell.vtc)
                    } else if cell.mu > 0.0 {
                        (dist * cell.mu) / (overlap * physics::wvdc(temp_prev[i], bc.pressure_air))
                    } else {
                        0.0
                    };

                    let vaporr2 = if nbr_cell.vtc > 0.0 {
                        1.0 / (overlap * nbr_cell.vtc)
                    } else if nbr_cell.mu > 0.0 {
                        (dist * nbr_cell.mu)
                            / (overlap * physics::wvdc(temp_prev[nbr_i], bc.pressure_air))
                    } else {
                        0.0
                    };

                    if vaporr1 + vaporr2 > 0.0 {
                        vpdiff += (nbr_cell.cell_vp - cell.cell_vp) / (vaporr1 + vaporr2);
                    }
                }

                // Heat capacitance
                let tcap = (cell.density * cell.spech + cell.volumetric_water * physics::W_SPECH)
                    * cell.volume;

                // Latent coupling term
                let mut qvp = 0.0;
                if settings.latent_heat_enabled && cell.material.is_some() {
                    qvp = vpdiff * physics::W_HV;
                }

                // Temperature update
                let denom = oorsum + (tcap / dt);
                temp_next[i] = if denom > 0.0 {
                    (torsum + qvp + cell.q_adds + (tcap * temp_prev[i] / dt)) / denom
                } else {
                    temp_prev[i]
                };
            }

            // --- Moisture step: compute next RH based on rh_prev and updated temps ---
            for i in 0..nx {
                let cell = &self.cells[[i, 0, 0]];

                let mut phioosum = 0.0;
                let mut phiorsum = 0.0;
                let mut vpoosum = 0.0;
                let mut vporsum = 0.0;

                let vpsat = physics::psy_psat_fn_temp(temp_next[i]);

                let mut neighbors: Vec<usize> = Vec::with_capacity(2);
                if i > 0 {
                    neighbors.push(i - 1);
                }
                if i + 1 < nx {
                    neighbors.push(i + 1);
                }

                for nbr_i in neighbors {
                    let nbr_cell = &self.cells[[nbr_i, 0, 0]];

                    let overlap = cell.size.area_along(0).min(nbr_cell.size.area_along(0));
                    let dist = (cell.size.x().abs() + nbr_cell.size.x().abs()) * 0.5;

                    // Vapor resistance
                    let vaporr1 = if cell.vtc > 0.0 {
                        1.0 / (overlap * cell.vtc)
                    } else if cell.mu > 0.0 {
                        (dist * cell.mu) / (overlap * physics::wvdc(temp_next[i], bc.pressure_air))
                    } else {
                        0.0
                    };

                    let vaporr2 = if nbr_cell.vtc > 0.0 {
                        1.0 / (overlap * nbr_cell.vtc)
                    } else if nbr_cell.mu > 0.0 {
                        (dist * nbr_cell.mu)
                            / (overlap * physics::wvdc(temp_next[nbr_i], bc.pressure_air))
                    } else {
                        0.0
                    };

                    if vaporr1 + vaporr2 > 0.0 {
                        vpoosum += 1.0 / (vaporr1 + vaporr2);
                        let nbr_vp = rh_prev[nbr_i] * physics::psy_psat_fn_temp(temp_next[nbr_i]);
                        vporsum += nbr_vp / (vaporr1 + vaporr2);
                    }

                    // Liquid transport resistance (only if both sides defined)
                    let rhr1 = if cell.dw > 0.0 && cell.dwd_phi > 0.0 {
                        dist / (overlap * cell.dw * cell.dwd_phi)
                    } else {
                        0.0
                    };
                    let rhr2 = if nbr_cell.dw > 0.0 && nbr_cell.dwd_phi > 0.0 {
                        dist / (overlap * nbr_cell.dw * nbr_cell.dwd_phi)
                    } else {
                        0.0
                    };

                    if rhr1 > 0.0 && rhr2 > 0.0 {
                        phioosum += 1.0 / (rhr1 + rhr2);
                        phiorsum += rh_prev[nbr_i] / (rhr1 + rhr2);
                    }
                }

                // Moisture capacitance
                let wcap = if cell.dwd_phi > 0.0 {
                    cell.dwd_phi * cell.volume
                } else {
                    0.0
                };

                let denom = phioosum + vpoosum * vpsat + wcap / dt;
                rh_next[i] = if denom > 0.0 {
                    (phiorsum + vporsum + (wcap * rh_prev[i]) / dt) / denom
                } else {
                    rh_prev[i]
                };

                // Clamp RH
                if rh_next[i] > 1.01 {
                    rh_next[i] = 1.01;
                }
                if rh_next[i] < 0.0 {
                    rh_next[i] = 0.0;
                }

                // Store updated vapor pressure for reporting consistency
                let cell_mut = &mut self.cells[[i, 0, 0]];
                cell_mut.cell_vp = rh_next[i] * vpsat;
            }

            // Check convergence against previous iteration state
            let mut max_dt: f64 = 0.0;
            let mut max_drh: f64 = 0.0;
            for i in 0..nx {
                max_dt = f64::max(max_dt, (temp_next[i] - temp_prev[i]).abs());
                max_drh = f64::max(max_drh, (rh_next[i] - rh_prev[i]).abs());
            }
            if max_dt <= settings.temp_convergence_tolerance
                && max_drh <= settings.rh_convergence_tolerance
            {
                converged = true;
                break;
            }

            // Prepare for next iteration
            temp_prev.clone_from_slice(&temp_next);
            rh_prev.clone_from_slice(&rh_next);
        }

        // Commit results into the writable surface_next buffer
        for i in 0..nx {
            let cell = &mut self.cells[[i, 0, 0]];
            cell.cell_temp = temp_next[i];
            cell.cell_rh = rh_next[i];
            cell.rh_p = cell.cell_rh * 100.0;
            // Water content already stored as volumetric_water during property update
            cell.w_report = if cell.density > 0.0 {
                cell.volumetric_water / cell.density
            } else {
                0.0
            };
        }

        Ok(converged && self.check_convergence(surface_curr_cells, &self.cells, settings))
    }

    pub fn override_cell_state(&mut self, ref_cells: &ArcArray3<Cell>) {
        for (cell_curr, cell_ref) in self.cells.iter_mut().zip(ref_cells) {
            cell_curr.cell_temp = cell_ref.cell_temp;
            cell_curr.cell_rh = cell_ref.cell_rh;
        }
    }
}
