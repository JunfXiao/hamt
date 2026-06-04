use crate::{
    cell::Boundary,
    physics::{ECLIPSE, geometry::Pt3D},
    uom::{
        Dimension, Quantity,
        dimensions::{
            Density, HeatTransferCoeff, MassFlux, MassResistance, Pressure, ThermalConductivity,
            ThermalResistance, VaporResistance,
        },
        units::{
            dimensionless, k_per_w, kg_per_hr, kg_per_m2_s_pa, kg_per_m3, m2_per_s,
            meter, pa_s_per_kg, s_per_kg, square_meter, w_per_m2_k
        },
    },
};

use super::*;
use anyhow::{Context, Result};
use tracing::debug;

/// All three interface resistances between two cells: thermal, vapor, and liquid.
///
/// Returned by [`Surface::calculate_resistance_triple`].
///
/// # Fields
/// - `therm`: thermal resistance [K/W]
/// - `vapor`: vapor (diffusion) resistance [Pa·s/kg]
/// - `liquid`: liquid (capillary, RH-driven) resistance [s/kg] (= rh_r1 + rh_r2)
pub struct ResistanceTriple {
    pub therm: ThermalResistance<f64>,
    pub vapor: VaporResistance<f64>,
    pub liquid: MassResistance<f64>,
}

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
            temp_surf_int: temp_surf_in,
            rh_surf_out,
            rh_surf_int: rh_surf_in,
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
                // strip: area used as a raw weight; sum is a Quantity of arbitrary D
                // so we accumulate `f(cell) * area_scalar` rather than carry m² through.
                let area = cell.size.area_along(0).value_in(square_meter);
                sum += f(cell) * area;
                sum_area += area;
            }
        }
        if sum_area > ECLIPSE {
            Ok(sum / sum_area)
        } else {
            Ok(Quantity(0.0))
        }
    }

    pub fn find_cell_idx_at_depth(&self, target_depth: Length<f64>) -> Option<[usize; 3]> {
        let mut best: Option<([usize; 3], Length<f64>)> = None;
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            if let CellType::Material(_) = cell.cell_type {
                let origin_x = cell.origin.x();
                let dist = (origin_x - target_depth).abs();
                match best {
                    Some((_, best_dist)) if dist >= best_dist => {}
                    _ => best = Some(([x, y, z], dist)),
                }
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// Compute trilinear interpolation weights for a target point within the
    /// material cell grid.
    ///
    /// Returns 8 `(cell_index, weight)` pairs whose weights sum to 1.0.
    /// For axes with only one cell (e.g. ny=1, nz=1 in a 1-D simulation) the
    /// corresponding fraction is 0 and only 2 entries carry non-zero weight
    /// (degenerating to simple linear interpolation).
    ///
    /// If the target coordinate is outside the material cell range on any axis,
    /// it is clamped to the nearest cell (no extrapolation).
    pub fn compute_interpolation_weights(&self, target: &Pt3D) -> InterpolationWeights {
        let dims = self.cells.raw_dim();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);

        // Helper: extract sorted unique material-cell coordinates along one axis.
        // Returns (sorted_indices, sorted_coords).
        fn axis_coords(
            cells: &ndarray::ArcArray3<Cell>,
            axis: usize,
            n: usize,
        ) -> (Vec<usize>, Vec<f64>) {
            let mut indices = Vec::new();
            let mut coords = Vec::new();
            for k in 0..n {
                let idx: [usize; 3] = match axis {
                    0 => [k, 0, 0],
                    1 => [0, k, 0],
                    _ => [0, 0, k],
                };
                if let Some(cell) = cells.get(idx) {
                    if matches!(cell.cell_type, CellType::Material(_)) {
                        // strip: trilinear interpolation tables index by raw f64 metres.
                        let coord = match axis {
                            0 => cell.origin.x().value_in(meter),
                            1 => cell.origin.y().value_in(meter),
                            _ => cell.origin.z().value_in(meter),
                        };
                        indices.push(k);
                        coords.push(coord);
                    }
                }
            }
            (indices, coords)
        }

        // Helper: find bracket (lo_index, hi_index, fraction) for a target on one axis.
        // Returns grid indices (into the cells array) and the interpolation fraction.
        fn find_bracket(indices: &[usize], coords: &[f64], target: f64) -> (usize, usize, f64) {
            if indices.is_empty() {
                return (0, 0, 0.0);
            }
            if indices.len() == 1 || target <= coords[0] {
                return (indices[0], indices[0], 0.0);
            }
            if target >= *coords.last().unwrap() {
                let last = *indices.last().unwrap();
                return (last, last, 0.0);
            }
            // Binary-search style: find the last coord <= target
            let mut lo = 0;
            for (i, &c) in coords.iter().enumerate() {
                if c <= target {
                    lo = i;
                } else {
                    break;
                }
            }
            let hi = lo + 1;
            let span = coords[hi] - coords[lo];
            let frac = if span > 0.0 {
                (target - coords[lo]) / span
            } else {
                0.0
            };
            (indices[lo], indices[hi], frac)
        }

        let (xi, xc) = axis_coords(&self.cells, 0, nx);
        let (yi, yc) = axis_coords(&self.cells, 1, ny);
        let (zi, zc) = axis_coords(&self.cells, 2, nz);

        // strip: target coords stripped to f64 to match `xc/yc/zc` from `axis_coords` above.
        let tx_val = target.x().value_in(meter);
        let ty_val = target.y().value_in(meter);
        let tz_val = target.z().value_in(meter);

        let (x0, x1, tx) = find_bracket(&xi, &xc, tx_val);
        let (y0, y1, ty) = find_bracket(&yi, &yc, ty_val);
        let (z0, z1, tz) = find_bracket(&zi, &zc, tz_val);

        // 8 trilinear weights: w_ijk = (1-tx or tx) * (1-ty or ty) * (1-tz or tz)
        let wx = [1.0 - tx, tx];
        let wy = [1.0 - ty, ty];
        let wz = [1.0 - tz, tz];
        let xs = [x0, x1];
        let ys = [y0, y1];
        let zs = [z0, z1];

        let mut weights: InterpolationWeights = [([0, 0, 0], 0.0); 8];
        let mut k = 0;
        for (ix, &xi_val) in xs.iter().enumerate() {
            for (iy, &yi_val) in ys.iter().enumerate() {
                for (iz, &zi_val) in zs.iter().enumerate() {
                    weights[k] = ([xi_val, yi_val, zi_val], wx[ix] * wy[iy] * wz[iz]);
                    k += 1;
                }
            }
        }
        weights
    }

    /// Interpolate a quantity at an exact point using trilinear interpolation
    /// between bracketing material cells.
    ///
    /// For a 1-D grid this degenerates to linear interpolation along x.
    /// If the target is outside the material cell range, the nearest cell value
    /// is returned (no extrapolation).
    pub fn interpolate_at_point<F, const D: Dimension>(
        &self,
        target: &Pt3D,
        f: F,
    ) -> Quantity<f64, D>
    where
        F: Fn(&Cell) -> Quantity<f64, D>,
    {
        let weights = self.compute_interpolation_weights(target);
        Self::apply_interpolation_weights(&self.cells, &weights, f)
    }

    /// Evaluate a weighted sum over cells using pre-computed interpolation weights.
    pub fn apply_interpolation_weights<F, const D: Dimension>(
        cells: &ndarray::ArcArray3<Cell>,
        weights: &InterpolationWeights,
        f: F,
    ) -> Quantity<f64, D>
    where
        F: Fn(&Cell) -> Quantity<f64, D>,
    {
        let mut value = 0.0;
        for &(idx, w) in weights {
            if w > 0.0 {
                if let Some(cell) = cells.get(idx) {
                    value += w * f(cell).value_unchecked();
                }
            }
        }
        Quantity(value)
    }

    /// Convenience wrapper: interpolate at a depth along x (y=0, z=0).
    /// Equivalent to `interpolate_at_point` with a 1-D target.
    pub fn interpolate_at_depth<F, const D: Dimension>(
        &self,
        target_depth: Length<f64>,
        f: F,
    ) -> Quantity<f64, D>
    where
        F: Fn(&Cell) -> Quantity<f64, D>,
    {
        let target = Pt3D::new(target_depth, 0.0 * meter, 0.0 * meter);
        self.interpolate_at_point(&target, f)
    }

    /// This function returns the values of the user-defined depth.
    /// If no user-defined depth is provided, it returns the mid-depth of the surface.
    ///
    /// Values are interpolated using cached trilinear weights for accuracy when
    /// cell centers do not align exactly with the target depth.
    ///
    /// # Returns
    /// A tuple containing the (cell_temp, cell_rh, water) of the depth.
    pub fn user_defined_values(&self) -> (Temperature<f64>, Dimensionless<f64>, Density<f64>) {
        if let Some(ref weights) = self.user_defined_cell_idx {
            let temp = Self::apply_interpolation_weights(&self.cells, weights, |c| c.cell_temp);
            let rh = Self::apply_interpolation_weights(&self.cells, weights, |c| c.cell_rh);
            let water =
                Self::apply_interpolation_weights(&self.cells, weights, |c| c.volumetric_water);
            (temp, rh, water)
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

    /// Calculates the heat flux across the surface boundary (environment -> surface).
    ///
    /// ## Input
    /// - `location`: boundary location (exterior or interior).
    ///
    /// ## Output
    /// - `avg_flux`: average heat flux across the surface boundary [W/m²].
    pub fn heat_flux(&self, location: BoundaryLocation) -> Result<EnergyFluxDensity<f64>> {
        let mut total_flux = Quantity(0.0);
        let mut total_area = Quantity(0.0);

        let boundary_convection_idx =
            self.cells_idx_map
                .iter()
                .find_map(|(ctype, idxs)| match ctype {
                    CellType::Boundary(Boundary::Convection, loc) if *loc == location => {
                        idxs.first().copied()
                    }
                    _ => None,
                });

        let boundary_cell = boundary_convection_idx
            .and_then(|idx| self.cells.get(idx))
            .context("No boundary convection cell found for surface heat flux")?;

        for idx in self.virtual_cells(location.clone()) {
            let surf = self
                .cells
                .get(*idx)
                .with_context(|| format!("Failed to get virtual cell {:?}", idx))?;
            let delta_t = boundary_cell.cell_temp - surf.cell_temp;
            let htc = boundary_cell.htc;
            let cell_area = surf.size.area_along(0);
            if htc > ECLIPSE * w_per_m2_k {
                total_flux += htc * delta_t * cell_area;
                total_area += cell_area;
            }
        }

        Ok(if total_area > ECLIPSE * square_meter {
            total_flux / total_area
        } else {
            debug!(
                "Total area for heat flux for {} calculation is too small: {:?}. Returning zero flux.",
                location, total_area
            );
            Quantity(0.0)
        })
    }

    /// Calculate the average moisture flux across the surface boundary (environment -> surface).
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
        bc: &BoundaryCondition,
    ) -> Result<MassFluxDensity<f64>> {
        let mut accum_flux: MassFlux<f64> = 0.0 * kg_per_hr;
        let mut total_area = 0.0 * square_meter;

        let boundary_convection_idx =
            self.cells_idx_map
                .iter()
                .find_map(|(ctype, idxs)| match ctype {
                    CellType::Boundary(Boundary::Convection, loc) if *loc == location => {
                        idxs.first().copied()
                    }
                    _ => None,
                });

        let boundary_conv_cell = boundary_convection_idx
            .and_then(|idx| self.cells.get(idx))
            .with_context(|| "No boundary convection cell found for surface moisture flux")?;

        let vtc_from_bc: VaporTransportCoeff<f64> = match &location {
            BoundaryLocation::Exterior => bc.vapor_transport_coeff_ext,
            BoundaryLocation::Interior => bc.vapor_transport_coeff_int,
        };

        let vtc = if boundary_conv_cell.vtc > ECLIPSE * kg_per_m2_s_pa {
            boundary_conv_cell.vtc
        } else {
            vtc_from_bc
        };

        for idx in self.virtual_cells(location.clone()) {
            let surf = self
                .cells
                .get(*idx)
                .with_context(|| format!("Failed to get virtual cell {:?}", idx))?;

            let delta_vp: Pressure<f64> = boundary_conv_cell.cell_vp - surf.cell_vp;
            let cell_area: Area<f64> = surf.size.area_along(0);

            if vtc > ECLIPSE * kg_per_m2_s_pa {
                accum_flux += vtc * delta_vp * cell_area;
                total_area += cell_area;
            }
        }

        Ok(if total_area > 0.0 * square_meter {
            accum_flux / total_area
        } else {
            debug!(
                "Total area for moisture flux for {} calculation is too small: {:?}. Returning zero flux.",
                location, total_area
            );
            Quantity(0.0)
        })
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

    /// Linear indices into `self.cells` of every material cell, in
    /// `cells.indexed_iter()` order. Used by the block-tridiagonal
    /// assembler to walk the wall in physical order without having to
    /// re-filter on every iteration. Cached state is intentionally not
    /// stored here — the caller can memoize across an outer iteration.
    pub fn material_cell_indices(&self) -> Vec<usize> {
        let mut idx = Vec::new();
        for (lin, cell) in self.cells.iter().enumerate() {
            if matches!(cell.cell_type, CellType::Material(_)) {
                idx.push(lin);
            }
        }
        idx
    }

    /// Calculate thermal, vapor, and liquid resistance between two cells.
    ///
    /// `therm` and `vapor` are computed identically to `calculate_resistance_pair`.
    /// `liquid` is `rh_r1 + rh_r2` where each half uses the cell's `dw` and
    /// `dwd_phi` with the 1e-20 guards from the inline neighbour loop in
    /// `calculate_physics_step`.
    pub fn calculate_resistance_triple(
        cell1: &Cell,
        cell2: &Cell,
        pressure_air: Pressure<f64>,
        axis: usize,
    ) -> ResistanceTriple {
        let overlap: Area<f64> = cell1
            .size
            .area_along(axis as i32)
            .min(cell2.size.area_along(axis as i32));
        let dist1 = cell1.size.extent_along(axis).abs() * 0.5;
        let dist2 = cell2.size.extent_along(axis).abs() * 0.5;

        if overlap <= ECLIPSE * square_meter {
            tracing::warn!(
                "calc_resistance_triple: degenerate geometry no overlap={:?}, cell1={:?}, cell2={:?}",
                overlap,
                cell1.cell_type,
                cell2.cell_type
            );
            return ResistanceTriple {
                therm: 1e100 * k_per_w,
                vapor: 1e100 * pa_s_per_kg,
                liquid: 0.0 * s_per_kg,
            };
        }

        if dist1 <= ECLIPSE * meter || dist2 <= ECLIPSE * meter {
            tracing::warn!(
                "calc_resistance_triple: degenerate geometry overlap={:?}, dist1={:?}, dist2={:?}, cell1={:?}, cell2={:?}",
                overlap,
                dist1,
                dist2,
                cell1.cell_type,
                cell2.cell_type
            );
            return ResistanceTriple {
                therm: 0.0 * k_per_w,
                vapor: 0.0 * pa_s_per_kg,
                liquid: 0.0 * s_per_kg,
            };
        }

        let htc1 = cell1.htc;
        let k1 = cell1.w_thermal_c;
        let thermr1: ThermalResistance<f64> = if htc1 > HeatTransferCoeff::<f64>::zero() {
            1.0 / (htc1 * overlap)
        } else if k1 > ThermalConductivity::<f64>::zero() {
            dist1 / (k1 * overlap)
        } else {
            0.0 * k_per_w
        };
        let htc2 = cell2.htc;
        let k2 = cell2.w_thermal_c;
        let thermr2: ThermalResistance<f64> = if htc2 > HeatTransferCoeff::<f64>::zero() {
            // Convection resistance
            1.0 / (htc2 * overlap)
        } else if k2 > ThermalConductivity::<f64>::zero() {
            // Conduction resistance
            dist2 / (k2 * overlap)
        } else {
            0.0 * k_per_w
        };

        let vtc1 = cell1.vtc;
        let vaporr1: VaporResistance<f64> = if vtc1 > VaporTransportCoeff::<f64>::zero() {
            // Vapor Convection resistance
            1.0 / (overlap * vtc1)
        } else if cell1.mu > ECLIPSE * dimensionless {
            (dist1 * cell1.mu)
                / (overlap * physics::water_vapor_permeability_coeff(cell1.cell_temp, pressure_air))
        } else {
            0.0 * pa_s_per_kg
        };
        let vtc2 = cell2.vtc;
        let vaporr2: VaporResistance<f64> = if vtc2 > VaporTransportCoeff::<f64>::zero() {
            1.0 / (overlap * vtc2)
        } else if cell2.mu > ECLIPSE * dimensionless {
            (dist2 * cell2.mu)
                / (overlap * physics::water_vapor_permeability_coeff(cell2.cell_temp, pressure_air))
        } else {
            0.0 * pa_s_per_kg
        };

        let rh_r1: MassResistance<f64> =
            if cell1.dw > 1e-20 * m2_per_s && cell1.dwd_phi > 1e-20 * kg_per_m3 {
                dist1 / (overlap * cell1.dw * cell1.dwd_phi)
            } else {
                0.0f64 * s_per_kg
            };
        // dist2 is already axis-aware (extent_along(axis) * 0.5 above)
        let rh_r2: MassResistance<f64> =
            if cell2.dw > 1e-20 * m2_per_s && cell2.dwd_phi > 1e-20 * kg_per_m3 {
                dist2 / (overlap * cell2.dw * cell2.dwd_phi)
            } else {
                0.0f64 * s_per_kg
            };

        ResistanceTriple {
            therm: thermr1 + thermr2,
            vapor: vaporr1 + vaporr2,
            liquid: rh_r1 + rh_r2,
        }
    }

    /// Calculate thermal and vapor resistance between two cells.
    ///
    /// Thin wrapper around [`calculate_resistance_triple`] — returns `(therm, vapor)`.
    /// All existing callers (assembly, virtual-cell averaging, block-tridiag path)
    /// remain byte-identical.
    ///
    /// # Returns
    /// A tuple containing the thermal and vapor resistances.
    /// - Thermal resistance [K/W] (Absolute)
    /// - Vapor resistance [Pa·s/kg] (Absolute)
    pub fn calculate_resistance_pair(
        cell1: &Cell,
        cell2: &Cell,
        pressure_air: Pressure<f64>,
        axis: usize,
    ) -> (ThermalResistance<f64>, VaporResistance<f64>) {
        let t = Self::calculate_resistance_triple(cell1, cell2, pressure_air, axis);
        (t.therm, t.vapor)
    }
}

#[cfg(test)]
mod resistance_triple_tests {
    use super::*;
    use crate::uom::units::{kg_per_m3, m2_per_s, s_per_kg};

    /// Build two material cells from the standard fixture and check:
    ///   1. `calculate_resistance_triple` therm/vapor == `calculate_resistance_pair`.
    ///   2. `.liquid` equals the hand-computed rh_r1 + rh_r2.
    ///   3. `.liquid` is zero when dw or dwd_phi are below the guard threshold.
    #[test]
    fn resistance_triple_matches_pair_and_inline_liquid() {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(2);
        let bc = &mgr.boundary_condition;

        // Grab two adjacent material cells.
        let cells: Vec<&crate::cell::Cell> = mgr
            .surface_curr
            .cells
            .iter()
            .filter(|c| matches!(c.cell_type, crate::cell::CellType::Material(_)))
            .collect();
        assert!(cells.len() >= 2, "fixture must have at least 2 material cells");
        let cell1 = cells[0];
        let cell2 = cells[1];

        // ── 1. therm/vapor agree with calculate_resistance_pair ────────────────
        let triple = Surface::calculate_resistance_triple(cell1, cell2, bc.pressure_air, 0);
        let (therm_pair, vapor_pair) =
            Surface::calculate_resistance_pair(cell1, cell2, bc.pressure_air, 0);

        assert_eq!(
            triple.therm.value_unchecked(),
            therm_pair.value_unchecked(),
            "therm mismatch"
        );
        assert_eq!(
            triple.vapor.value_unchecked(),
            vapor_pair.value_unchecked(),
            "vapor mismatch"
        );

        // ── 2. liquid == hand-computed rh_r1 + rh_r2 ──────────────────────────
        let overlap = cell1.size.area_along(0).min(cell2.size.area_along(0));
        let rh_r1: crate::uom::dimensions::MassResistance<f64> =
            if cell1.dw > 1e-20 * m2_per_s && cell1.dwd_phi > 1e-20 * kg_per_m3 {
                (cell1.size.x().abs() * 0.5) / (overlap * cell1.dw * cell1.dwd_phi)
            } else {
                0.0f64 * s_per_kg
            };
        let rh_r2: crate::uom::dimensions::MassResistance<f64> =
            if cell2.dw > 1e-20 * m2_per_s && cell2.dwd_phi > 1e-20 * kg_per_m3 {
                (cell2.size.x().abs() * 0.5) / (overlap * cell2.dw * cell2.dwd_phi)
            } else {
                0.0f64 * s_per_kg
            };
        let expected_liquid = rh_r1 + rh_r2;

        assert_eq!(
            triple.liquid.value_unchecked(),
            expected_liquid.value_unchecked(),
            "liquid resistance mismatch: triple={:?}, hand={:?}",
            triple.liquid,
            expected_liquid,
        );

        // ── 3. liquid is zero when dw/dwd_phi are below the guard threshold ────
        // Clone a cell with dw = 0 (below 1e-20 m²/s guard).
        let mut cell_no_dw = cell1.clone();
        cell_no_dw.dw = 0.0 * m2_per_s;
        let triple_no_dw =
            Surface::calculate_resistance_triple(&cell_no_dw, cell2, bc.pressure_air, 0);
        // rh_r1 should be zero; only rh_r2 contributes (if cell2.dw is above threshold).
        let expected_no_dw = if cell2.dw > 1e-20 * m2_per_s && cell2.dwd_phi > 1e-20 * kg_per_m3 {
            (cell2.size.x().abs() * 0.5) / (overlap * cell2.dw * cell2.dwd_phi)
        } else {
            0.0 * s_per_kg
        };
        assert_eq!(
            triple_no_dw.liquid.value_unchecked(),
            expected_no_dw.value_unchecked(),
            "liquid should be zero for cell1 side when dw=0"
        );

        // Clone both cells with dw = 0.
        let mut cell2_no_dw = cell2.clone();
        cell2_no_dw.dw = 0.0 * m2_per_s;
        let triple_both_zero =
            Surface::calculate_resistance_triple(&cell_no_dw, &cell2_no_dw, bc.pressure_air, 0);
        assert_eq!(
            triple_both_zero.liquid.value_unchecked(),
            0.0,
            "liquid must be exactly 0 when both cells have dw=0"
        );
    }
}
