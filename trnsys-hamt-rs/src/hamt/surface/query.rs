use super::*;
use anyhow::{Context, Result};

impl Surface {
    /// Compatibility helper for callers that previously expected a vector of surfaces.
    pub fn get(&self, idx: usize) -> Option<&Self> {
        if idx == 0 {
            Some(self)
        } else {
            None
        }
    }

    /// Compute all reporting values on demand.
    pub fn report(&self) -> Result<SurfaceReport> {
        let temp_surf_out = self
            .avg_boundary(BoundaryLocation::Exterior, |c| c.cell_temp)
            .with_context(|| "Failed to calculate exterior surface temperature")?;
        let temp_surf_in = self
            .avg_boundary(BoundaryLocation::Interior, |c| c.cell_temp)
            .with_context(|| "Failed to calculate interior surface temperature")?;
        let rh_surf_out = self
            .avg_boundary(BoundaryLocation::Exterior, |c| c.cell_rh)
            .with_context(|| "Failed to calculate exterior surface relative humidity")?;
        let rh_surf_in = self
            .avg_boundary(BoundaryLocation::Interior, |c| c.cell_rh)
            .with_context(|| "Failed to calculate interior surface relative humidity")?;

        let (temp_surf_user_def, rh_surf_user_def, water_surf_user_def) =
            self.user_defined_values();

        let q_surf_ext = self
            .heat_flux(BoundaryLocation::Exterior)
            .with_context(|| "Failed to calculate exterior surface heat flux")?;
        let q_surf_int = self
            .heat_flux(BoundaryLocation::Interior)
            .with_context(|| "Failed to calculate interior surface heat flux")?;
        let m_surf_ext = self
            .moisture_flux(BoundaryLocation::Exterior)
            .with_context(|| "Failed to calculate exterior surface moisture flux")?;
        let m_surf_int = self
            .moisture_flux(BoundaryLocation::Interior)
            .with_context(|| "Failed to calculate interior surface moisture flux")?;

        let water_total = self
            .cells_idx_map
            .iter()
            .filter_map(|(ctype, idxs)| match ctype {
                CellType::Material(_) => Some(idxs),
                _ => None,
            })
            .flatten()
            .map(|[x, y, z]| {
                let cell = &self.cells[[*x, *y, *z]];
                cell.volumetric_water * cell.volume
            })
            .sum();

        Ok(SurfaceReport {
            temp_surf_out,
            temp_surf_in,
            rh_surf_out,
            rh_surf_in,
            temp_surf_user_def,
            rh_surf_user_def,
            water_surf_user_def,
            q_surf_ext,
            q_surf_int,
            m_surf_ext,
            m_surf_int,
            water_total,
        })
    }

    /// Find the boundary cells for a given location.
    pub fn boundary_cells(
        &self,
        location: BoundaryLocation,
    ) -> impl Iterator<Item = &[usize; 3]> + '_ {
        self.cells_idx_map
            .iter()
            .filter(
                move |(ctype, _)| matches!(ctype, CellType::Boundary(_, loc) if *loc == location),
            )
            .flat_map(|(_, idxs)| idxs)
    }

    /// Calculates the average value of a function over the boundary cells of a surface.
    /// The "average value" is calculated as the sum of the function values divided by the total area of the boundary cells perpendicular to the surface depth direction (x axis).
    pub fn avg_boundary<F>(&self, location: BoundaryLocation, f: F) -> Result<f64>
    where
        F: Fn(&Cell) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_area = 0.0;
        {
            for idx in self.boundary_cells(location) {
                let cell = self
                    .cells
                    .get(*idx)
                    .with_context(|| format!("Cell index {:?} is invalid", idx))?;
                sum += f(cell);
                sum_area += cell.size.area_along(0);
            }
        }
        if sum_area > 0.0 {
            Ok(sum / sum_area)
        } else {
            Ok(0.0)
        }
    }

    /// This function returns the values of the user-defined depth.
    /// If no user-defined depth is provided, it returns the mid-depth of the surface.
    /// # Returns
    /// A tuple containing the (cell_temp, cell_rh, water) of the depth.
    pub fn user_defined_values(&self) -> (f64, f64, f64) {
        let target_depth = self.user_defined_depth.unwrap_or_else(|| {
            // default to mid-depth if not provided
            let dims = self.cells.raw_dim();
            if dims[0] > 0 {
                (dims[0] as f64) / 2.0
            } else {
                0.0
            }
        });

        let mut best: Option<([usize; 3], f64)> = None;
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            let depth = cell.origin.x();
            let dist = (depth - target_depth).abs();
            match best {
                Some((_, best_dist)) if dist >= best_dist => {}
                _ => best = Some(([x, y, z], dist)),
            }
        }

        if let Some(([x, y, z], _)) = best {
            let cell = &self.cells[[x, y, z]];
            (cell.cell_temp, cell.cell_rh, cell.volumetric_water)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Returns the neighbor cell index toward the center of the surface along the x-axis.
    pub fn neighbor_toward_center(
        &self,
        idx: [usize; 3],
        location: &BoundaryLocation,
    ) -> Option<[usize; 3]> {
        let dims = self.cells.raw_dim();
        let x = idx[0];
        match location {
            BoundaryLocation::Exterior => {
                if x + 1 < dims[0] {
                    Some([x + 1, idx[1], idx[2]])
                } else {
                    None
                }
            }
            BoundaryLocation::Interior => x.checked_sub(1).map(|nx| [nx, idx[1], idx[2]]),
        }
    }

    /// Calculates the heat flux across the surface boundary.
    ///
    /// ## Input
    /// - `location`: boundary location (exterior or interior).
    ///
    /// ## Output
    /// - `avg_flux`: average heat flux across the surface boundary [W/m²].
    pub fn heat_flux(&self, location: BoundaryLocation) -> Result<f64> {
        // Total Heat Flux in W
        let mut total_flux = 0.0;
        let mut total_area = 0.0;

        for idx in self.boundary_cells(location.clone()) {
            if let Some(n_idx) = self.neighbor_toward_center(*idx, &location) {
                let b = self
                    .cells
                    .get(*idx)
                    .with_context(|| format!("Failed to get boundary cell {:?}", idx))?;
                let n = self.cells.get(n_idx).with_context(|| {
                    format!(
                        "Failed to get neighbor cell {:?} for boundary cell {:?}",
                        n_idx, idx
                    )
                })?;
                let delta_t = b.cell_temp - n.cell_temp;

                // Prefer boundary HTC, fallback to conduction
                let flux_w_m2 = if b.htc > 0.0 {
                    b.htc * delta_t
                } else if b.w_thermal_c > 0.0 && b.size.x() > 0.0 {
                    (b.w_thermal_c / b.size.x()) * delta_t
                } else {
                    0.0
                };
                let cell_area = b.size.area_along(0);
                total_flux += flux_w_m2 * cell_area;
                total_area += cell_area;
            }
        }

        if total_area > 0.0 {
            Ok(total_flux / total_area)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate the average moisture flux across the surface boundary.
    ///
    /// Units:
    /// - `vtc`: vapor transfer coefficient [kg/(m²·s·Pa)] (boundary override or cell.vtc).
    /// - `flux`: instantaneous moisture flux per boundary face [kg/(m²·s)].
    /// - `delta_vp`: vapor pressure difference across boundary face [Pa].
    /// - `accum`: running sum of flux values used for averaging [kg/(m²·s)].
    ///
    /// ## Input
    /// - `location`: boundary location (exterior or interior).
    ///
    /// ## Output
    /// - `avg_flux`: average moisture flux across the surface boundary [kg/(m²·s)].
    pub fn moisture_flux(&self, location: BoundaryLocation) -> Result<f64> {
        // Accumulated flux in [kg/s]
        let mut accum_flux = 0.0;
        let mut total_area = 0.0;

        for idx in self.boundary_cells(location.clone()) {
            let b = self
                .cells
                .get(*idx)
                .with_context(|| format!("Cannot get boundary cell index {:?}", idx))?;
            if let Some(n_idx) = self.neighbor_toward_center(*idx, &location) {
                let n = self
                    .cells
                    .get(n_idx)
                    .with_context(|| format!("Cannot get neighbor cell index {:?}", n_idx))?;
                let delta_vp = b.cell_vp - n.cell_vp;

                let flux = if let Some(vtc) = match &location {
                    BoundaryLocation::Exterior => self.ext_vtc.or(Some(b.vtc)),
                    BoundaryLocation::Interior => self.int_vtc.or(Some(b.vtc)),
                } {
                    // flux = vtc * Δvp → kg/(m²·s)
                    vtc * delta_vp
                } else {
                    0.0
                };

                let cell_area = b.size.area_along(0);

                accum_flux += flux * cell_area;
                total_area += cell_area;
            }
        }

        if total_area > 0.0 {
            Ok(accum_flux / total_area)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates thermal and vapor resistances for a cell's face.
    fn calculate_resistances(&self, cell: &Cell, face_idx: usize, pressure: f64) -> (f64, f64) {
        let overlap = cell.overlap[face_idx];
        let dist = cell.dist[face_idx];

        let therm_r = if cell.htc > 0.0 {
            1.0 / (overlap * cell.htc)
        } else if cell.material.is_some() && cell.w_thermal_c > 0.0 {
            dist / (overlap * cell.w_thermal_c)
        } else {
            0.0
        };

        let vapor_r = if cell.vtc > 0.0 {
            1.0 / (overlap * cell.vtc)
        } else if cell.material.is_some() {
            let wvdc = physics::wvdc(cell.cell_temp, pressure);
            (dist * cell.mu) / (overlap * wvdc)
        } else {
            0.0
        };

        (therm_r, vapor_r)
    }

    /// Checks the maximum change in temperature and RH across cells for convergence.
    pub fn check_convergence(
        &self,
        curr: &ArcArray3<Cell>,
        next: &ArcArray3<Cell>,
        settings: &HamtSettings,
    ) -> bool {
        for (((x, y, z), cell_curr), cell_next) in curr.indexed_iter().zip(next) {
            let delta_t = (cell_next.cell_temp - cell_curr.cell_temp).abs();

            if delta_t > settings.temp_convergence_tolerance {
                debug!(
                    "Temperature change at ({}, {}, {}) exceeded tolerance: {}",
                    x, y, z, delta_t
                );
                return false;
            }

            let delta_rh = cell_next.cell_rh - cell_curr.cell_rh;

            if delta_rh.abs() > settings.rh_convergence_tolerance {
                debug!(
                    "Relative humidity change at ({}, {}, {}) exceeded tolerance: {}",
                    x, y, z, delta_rh
                );
                return false;
            }
        }

        true
    }
}
