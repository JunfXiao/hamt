use ndarray::ArcArray3;

use crate::hamt::cell::{BoundaryLocation, Cell, CellType};
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
#[derive(Debug, Clone, Default)]
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
            ..Default::default()
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
                .or_insert(Vec::new())
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
    pub fn report(&self) -> SurfaceReport {
        let temp_surf_out = self.avg_boundary(BoundaryLocation::Exterior, |c| c.cell_temp);
        let temp_surf_in = self.avg_boundary(BoundaryLocation::Interior, |c| c.cell_temp);
        let rh_surf_out = self.avg_boundary(BoundaryLocation::Exterior, |c| c.cell_rh);
        let rh_surf_in = self.avg_boundary(BoundaryLocation::Interior, |c| c.cell_rh);

        let (temp_surf_user_def, rh_surf_user_def, water_surf_user_def) =
            self.user_defined_values();

        let q_surf_ext = self.heat_flux(BoundaryLocation::Exterior);
        let q_surf_int = self.heat_flux(BoundaryLocation::Interior);
        let m_surf_ext = self.moisture_flux(BoundaryLocation::Exterior);
        let m_surf_int = self.moisture_flux(BoundaryLocation::Interior);

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
                cell.water * cell.volume
            })
            .sum();

        SurfaceReport {
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
        }
    }

    /// Find the boundary cells for a given location.
    fn boundary_cells(&self, location: BoundaryLocation) -> impl Iterator<Item = [usize; 3]> + '_ {
        self.cells_idx_map
            .iter()
            .filter(
                move |(ctype, _)| matches!(ctype, CellType::Boundary(_, loc) if *loc == location),
            )
            .flat_map(|(_, idxs)| idxs.clone())
    }

    /// Calculates the average value of a function over the boundary cells of a surface.
    /// The "average value" is calculated as the sum of the function values divided by the total area of the boundary cells perpendicular to the surface depth direction (x axis).
    fn avg_boundary<F>(&self, location: BoundaryLocation, f: F) -> f64
    where
        F: Fn(&Cell) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_area = 0.0;
        for [x, y, z] in self.boundary_cells(location) {
            let cell = &self.cells[[x, y, z]];
            sum += f(cell);
            sum_area += cell.length.area_along(0);
        }
        if sum_area > 0.0 {
            sum / sum_area
        } else {
            0.0
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
            (cell.cell_temp, cell.cell_rh, cell.water)
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
    fn heat_flux(&self, location: BoundaryLocation) -> f64 {
        // Total Heat Flux in W
        let mut total_flux = 0.0;
        let mut total_area = 0.0;

        for idx in self.boundary_cells(location.clone()) {
            if let Some(n_idx) = self.neighbor_toward_center(idx, &location) {
                let b = &self.cells[idx];
                let n = &self.cells[n_idx];
                let delta_t = b.cell_temp - n.cell_temp;

                // Prefer boundary HTC, fallback to conduction
                let flux_w_m2 = if b.htc > 0.0 {
                    b.htc * delta_t
                } else if b.w_thermal_c > 0.0 && b.length.x() > 0.0 {
                    (b.w_thermal_c / b.length.x()) * delta_t
                } else {
                    0.0
                };
                let cell_area = b.length.area_along(0);
                total_flux += flux_w_m2 * cell_area;
                total_area += cell_area;
            }
        }

        if total_area > 0.0 {
            total_flux / total_area
        } else {
            0.0
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
    fn moisture_flux(&self, location: BoundaryLocation) -> f64 {
        // Accumulated flux in [kg/s]
        let mut accum_flux = 0.0;
        let mut total_area = 0.0;

        for idx in self.boundary_cells(location.clone()) {
            if let Some(n_idx) = self.neighbor_toward_center(idx, &location) {
                let b = &self.cells[idx];
                let n = &self.cells[n_idx];
                let delta_vp = b.vp - n.vp;

                let flux = if let Some(vtc) = match &location {
                    BoundaryLocation::Exterior => self.ext_vtc.or(Some(b.vtc)),
                    BoundaryLocation::Interior => self.int_vtc.or(Some(b.vtc)),
                } {
                    // flux = vtc * Δvp → kg/(m²·s)
                    vtc * delta_vp
                } else {
                    0.0
                };

                let cell_area = b.length.area_along(0);

                accum_flux += flux * cell_area;
                total_area += cell_area;
            }
        }

        if total_area > 0.0 {
            accum_flux / total_area
        } else {
            0.0
        }
    }
}
