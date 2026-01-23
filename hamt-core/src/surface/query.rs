use crate::{
    cell::Boundary,
    physics::ECLIPSE,
    uom::{
        Dimension, Quantity,
        dimensions::{Density, Pressure},
        units::square_meter,
    },
};

use super::*;
use anyhow::{Context, Result};

impl Surface {
    /// Compatibility helper for callers that previously expected a vector of surfaces.
    pub fn get(&self, idx: usize) -> Option<&Self> {
        if idx == 0 { Some(self) } else { None }
    }

    /// Compute all reporting values on demand.
    pub fn report(&self, bc: &BoundaryCondition) -> Result<SurfaceReport> {
        let temp_surf_out = self
            .avg_virtual_cell(BoundaryLocation::Exterior, |c| c.cell_temp)
            .with_context(|| "Failed to calculate exterior surface temperature")?;
        let temp_surf_in = self
            .avg_virtual_cell(BoundaryLocation::Interior, |c| c.cell_temp)
            .with_context(|| "Failed to calculate interior surface temperature")?;
        let rh_surf_out = self
            .avg_virtual_cell(BoundaryLocation::Exterior, |c| c.cell_rh)
            .with_context(|| "Failed to calculate exterior surface relative humidity")?;
        let rh_surf_in = self
            .avg_virtual_cell(BoundaryLocation::Interior, |c| c.cell_rh)
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
            .moisture_flux(BoundaryLocation::Exterior, bc)
            .with_context(|| "Failed to calculate exterior surface moisture flux")?;
        let m_surf_int = self
            .moisture_flux(BoundaryLocation::Interior, bc)
            .with_context(|| "Failed to calculate interior surface moisture flux")?;

        let water_total: Mass<f64> = self
            .cells_idx_map
            .iter()
            .filter_map(|(ctype, idxs)| match ctype {
                CellType::Material(_) => Some(idxs),
                _ => None,
            })
            .flatten()
            .map(|[x, y, z]| {
                let cell = &self.cells[[*x, *y, *z]];
                let water = cell.volumetric_water;
                let volume = cell.volume;
                water * volume
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
    pub fn virtual_cells(
        &self,
        location: BoundaryLocation,
    ) -> impl Iterator<Item = &[usize; 3]> + '_ {
        self.cells_idx_map
            .iter()
            .filter(move |(ctype, _)| matches!(ctype, CellType::Virtual(loc) if *loc == location))
            .flat_map(|(_, idxs)| idxs)
    }

    /// Calculates the average value of a function over the boundary cells of a surface.
    /// The "average value" is calculated as the sum of the function values divided by the total area of the boundary cells perpendicular to the surface depth direction (x axis).
    pub fn avg_virtual_cell<F, const D: Dimension>(
        &self,
        location: BoundaryLocation,
        f: F,
    ) -> Result<Quantity<f64, D>>
    where
        F: Fn(&Cell) -> Quantity<f64, D>,
    {
        let mut sum = Quantity::<f64, D>(0.0);
        let mut sum_area = 0.0;
        {
            for idx in self.virtual_cells(location) {
                let cell = self
                    .cells
                    .get(*idx)
                    .with_context(|| format!("Cell index {:?} is invalid", idx))?;
                let area = cell.size.area_along(0).value_in(square_meter);
                sum += f(cell) * area;
                sum_area += area;
            }
        }
        if sum_area > Quantity(ECLIPSE) {
            Ok(sum / sum_area)
        } else {
            Ok(Quantity(0.0))
        }
    }

    pub fn find_cell_idx_at_depth(&self, target_depth: Length<f64>) -> Option<[usize; 3]> {
        let mut best: Option<([usize; 3], Length<f64>)> = None;
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            let length = cell.origin.x() + cell.size.x() / 2.0;
            let dist = (length - target_depth).abs();
            match best {
                Some((_, best_dist)) if dist >= best_dist => {}
                _ => best = Some(([x, y, z], dist)),
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// This function returns the values of the user-defined depth.
    /// If no user-defined depth is provided, it returns the mid-depth of the surface.
    /// # Returns
    /// A tuple containing the (cell_temp, cell_rh, water) of the depth.
    pub fn user_defined_values(&self) -> (Temperature<f64>, Dimensionless<f64>, Density<f64>) {
        if let Some([x, y, z]) = self.user_defined_cell_idx {
            let cell = &self.cells[[x, y, z]];
            (cell.cell_temp, cell.cell_rh, cell.volumetric_water)
        } else {
            (Quantity(0.0), Quantity(0.0), Quantity(0.0))
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
    pub fn heat_flux(&self, location: BoundaryLocation) -> Result<EnergyFlux<f64>> {
        // Find the Virtual Cell (Surface Node)
        // We expect exactly one virtual cell column per location in this 1D model
        let v_idx = self
            .virtual_cells(location.clone())
            .next()
            .with_context(|| format!("No virtual cell found for location {:?}", location))?;
        let v = &self.cells[*v_idx];

        // Total Heat Flux in W/m² (Sum of parallel paths: Convection + Radiation + etc.)
        let mut total_flux = Quantity(0.0);

        // Iterate over all cells that are Boundary cells at this location
        for (ctype, idxs) in &self.cells_idx_map {
            if let CellType::Boundary(_, loc) = ctype {
                if *loc == location {
                    for idx in idxs {
                        let b = &self.cells[*idx];
                        
                        // Flux = h * (T_b - T_v)
                        // htc comes from the Boundary cell (e.g., ExtCon, ExtRad)
                        let htc = b.htc;
                        if htc > Quantity(ECLIPSE) {
                            let delta_t = b.cell_temp - v.cell_temp;
                            total_flux += htc * delta_t;
                        }
                    }
                }
            }
        }

        Ok(total_flux)
    }

    /// Calculate the average moisture flux across the surface boundary.
    ///
    /// Units:
    /// - `vtc`: vapor transfer coefficient [kg/(m²·s·Pa)] (boundary override or cell.vtc).
    /// - `flux`: instantaneous moisture flux per boundary face [kg/(m²·s)].
    /// - `delta_vp`: vapor pressure difference across boundary face [Pa].
    ///
    /// ## Input
    /// - `location`: boundary location (exterior or interior).
    ///
    /// ## Output
    /// - `avg_flux`: average moisture flux across the surface boundary [kg/(m²·s)].
    pub fn moisture_flux(
        &self,
        location: BoundaryLocation,
        _bc: &BoundaryCondition,
    ) -> Result<MassFlux<f64>> {
        // Find the Virtual Cell (Surface Node)
        let v_idx = self
            .virtual_cells(location.clone())
            .next()
            .with_context(|| format!("No virtual cell found for location {:?}", location))?;
        let v = &self.cells[*v_idx];

        let mut total_flux = Quantity(0.0);

        // Iterate over all cells that are Boundary cells at this location (usually just Convection)
        for (ctype, idxs) in &self.cells_idx_map {
            if let CellType::Boundary(_, loc) = ctype {
                if *loc == location {
                    for idx in idxs {
                        let b = &self.cells[*idx];
                        
                        // Flux = beta * (P_b - P_v)
                        // vtc comes from the Boundary cell
                        let vtc = b.vtc;
                        if vtc > Quantity(0.0) {
                            let delta_vp = b.cell_vp - v.cell_vp;
                            total_flux += vtc * delta_vp;
                        }
                    }
                }
            }
        }

        Ok(total_flux)
    }

    /// Checks the maximum change in temperature and RH across cells for convergence.
    pub fn check_convergence(&self, other: &Surface, settings: &HamtSettings) -> bool {
        for (((x, y, z), cell_curr), cell_next) in self.cells.indexed_iter().zip(other.cells.iter())
        {
            if !matches!(cell_curr.cell_type, CellType::Material(_)) {
                continue;
            }

            let temp_tol = settings
                .temp_convergence_tolerance
                .value_in(diman::si::units::kelvin);
            let delta_t = (cell_next.cell_temp - cell_curr.cell_temp)
                .value_in(diman::si::units::kelvin)
                .abs();

            if delta_t > temp_tol {
                debug!(
                    "Temperature change at ({}, {}, {}) exceeded tolerance: {:?}",
                    x, y, z, delta_t
                );
                return false;
            }

            let delta_rh = cell_next.cell_rh - cell_curr.cell_rh;

            if delta_rh.abs() > settings.rh_convergence_tolerance {
                debug!(
                    "Relative humidity change at ({}, {}, {}) exceeded tolerance: {:#?}",
                    x, y, z, delta_rh
                );
                return false;
            }
        }

        true
    }
}
