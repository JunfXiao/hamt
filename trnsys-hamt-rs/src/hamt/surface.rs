use anyhow::Ok;
use ndarray::ArcArray3;
use tracing::debug;

use crate::hamt::cell::{BoundaryLocation, Cell, CellType, ADJACENT_CELLS_MAX};
use crate::hamt::{boundary::BoundaryCondition, hamt_manager::HamtSettings, physics};
use crate::trnsys::error::TrnSysError;
use anyhow::{Context, Result};
use std::{collections::HashMap, sync::Arc};

/// Snapshot of on-demand surface reporting values.
#[derive(Debug, Clone, Default)]
pub struct SurfaceReport {
    pub temp_surf_out: f64,
    pub temp_surf_in: f64,
    pub rh_surf_out: f64,
    pub rh_surf_in: f64,
    pub temp_surf_user_def: f64,
    pub rh_surf_user_def: f64,
    pub water_surf_user_def: f64,
    pub q_surf_ext: f64,
    pub q_surf_int: f64,
    pub m_surf_ext: f64,
    pub m_surf_int: f64,
    pub water_total: f64,
}

/// Holds geometry, cells, and utilities for a single surface.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Surface name.
    pub name: String,
    /// Name of the construction used by this surface.
    pub construction_name: String,
    /// Surface area [m²].
    pub area: f64,
    /// Cell lattice.
    pub cells: ArcArray3<Cell>,

    // ========= Cell Index Helpers =========
    cells_idx_map: HashMap<CellType, Vec<[usize; 3]>>,

    // ========= Boundary Conditions / Flags =========
    /// Exterior vapor transfer coefficient [kg/(m²·s·Pa)].
    ext_vtc: Option<f64>,
    /// Interior vapor transfer coefficient [kg/(m²·s·Pa)].
    int_vtc: Option<f64>,

    // ========= Optional user-defined probe =========
    /// Depth (m) at which user-defined reporting is performed.
    pub user_defined_depth: Option<f64>,
}

impl Surface {
    pub fn new(name: String, area: f64, construction_name: String, cells: ArcArray3<Cell>) -> Self {
        let mut surface = Surface {
            name,
            construction_name,
            area,
            cells,
            cells_idx_map: HashMap::new(),
            ext_vtc: None,
            int_vtc: None,
            user_defined_depth: None,
        };
        surface.init_cells_idx_map();
        surface
    }

    fn init_cells_idx_map(&mut self) {
        self.cells_idx_map.clear();
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            let cell_type = cell.cell_type.clone();
            self.cells_idx_map
                .entry(cell_type)
                .or_insert(Vec::with_capacity(self.cells.len())) // For performance
                .push([x, y, z]);
        }
    }

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
    fn boundary_cells(&self, location: BoundaryLocation) -> impl Iterator<Item = &[usize; 3]> + '_ {
        self.cells_idx_map
            .iter()
            .filter(
                move |(ctype, _)| matches!(ctype, CellType::Boundary(_, loc) if *loc == location),
            )
            .flat_map(|(_, idxs)| idxs)
    }

    /// Calculates the average value of a function over the boundary cells of a surface.
    /// The "average value" is calculated as the sum of the function values divided by the total area of the boundary cells perpendicular to the surface depth direction (x axis).
    fn avg_boundary<F>(&self, location: BoundaryLocation, f: F) -> Result<f64>
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
    fn user_defined_values(&self) -> (f64, f64, f64) {
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
    fn neighbor_toward_center(
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
    fn heat_flux(&self, location: BoundaryLocation) -> Result<f64> {
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
    fn moisture_flux(&self, location: BoundaryLocation) -> Result<f64> {
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

    /// Executes a single-timestep update assuming `self` is the writable `surface_next`.
    /// Reads from `surface_prev` (t-1) and `surface_curr` (current iteration state).
    ///
    /// Returns `true` if no further iterations are needed and the convergence criteria is met.
    pub fn calculate_timestep(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        _surface_prev: &ArcArray3<Cell>,
        surface_curr: &ArcArray3<Cell>,
    ) -> Result<bool> {
        // Collect indices first to avoid simultaneous immutable/mutable borrows of `self`.
        let exterior_idxs: Vec<[usize; 3]> = self
            .boundary_cells(BoundaryLocation::Exterior)
            .map(|v| v.to_owned())
            .collect();
        for idx in exterior_idxs {
            let cell = self.cells.get_mut(idx).with_context(|| {
                format!("Cell index {:?} of surface_next is out of bounds", idx)
            })?;

            cell.cell_temp = bc.temp_outside_air;

            let rho_vapor = physics::psy_rhov_fn_tdb_rh(bc.temp_outside_air, bc.rh_outside_air);

            cell.cell_rh = physics::psy_rh_fn_tdb_rhov(bc.temp_outside_air, rho_vapor);
        }

        let interior_idxs: Vec<[usize; 3]> = self
            .boundary_cells(BoundaryLocation::Interior)
            .map(|v| v.to_owned())
            .collect();
        for idx in interior_idxs {
            let cell = self.cells.get_mut(idx).with_context(|| {
                format!("Cell index {:?} of surface_next is out of bounds", idx)
            })?;

            cell.cell_temp = bc.temp_inside_air;

            let rho_vapor = physics::psy_rhov_fn_tdb_rh(bc.temp_inside_air, bc.rh_inside_air);

            cell.cell_rh = physics::psy_rh_fn_tdb_rhov(bc.temp_inside_air, rho_vapor);
        }

        // 2) Placeholder for full coupled heat/moisture solve (to be added)

        //    This should iterate and update `self.cells` using `surface_curr` and material props.

        // 3) Convergence check (next vs curr). Caller can use returned deltas if needed.

        Ok(self.check_convergence(surface_curr, &self.cells, settings))
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

    pub fn override_cell_state(&mut self, ref_surface: &Surface) {
        for (cell_curr, cell_ref) in self.cells.iter_mut().zip(&ref_surface.cells) {
            cell_curr.cell_temp = cell_ref.cell_temp;
            cell_curr.cell_rh = cell_ref.cell_rh;
        }
    }
}
