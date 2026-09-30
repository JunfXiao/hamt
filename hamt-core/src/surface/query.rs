use crate::{
    cell::Boundary,
    physics::{ECLIPSE, geometry::Pt3D},
    uom::{
        Dimension, Quantity,
        dimensions::{
            Density, HeatTransferCoeff, MassFlux, MassResistance, Power, Pressure,
            ThermalConductivity, ThermalResistance, VaporResistance,
        },
        units::{
            dimensionless, k_per_w, kg_per_hr, kg_per_m2_s_pa, kg_per_m3, m2_per_s, meter,
            pa_s_per_kg, s_per_kg, square_meter, w_per_m2_k, watts,
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

    /// Calculates the TOTAL sensible heat flux through the surface films of one face
    /// (environment -> surface), area-averaged.
    ///
    /// This is the sum of ALL films the solver applies to the face's massless surface node:
    /// the convective film and, where active, the longwave film. Interior, 14-input deck:
    /// `h_c·(T_air − T_s) + h_rad·(T_MR − T_s)`; interior, legacy 12-input deck, and
    /// exterior: exactly one active film (the others are inert placeholders), i.e. the value
    /// is bit-identical to that single film. The node stores nothing, so the total equals the
    /// conduction into the adjacent material cell.
    ///
    /// It is what Type 395 output 8 (`HeatFlux_in`, interior) and output 9 (`HeatFlux_out`,
    /// exterior) report. The surface gains `qWG1` / `qWG2` (deposited in the surface material
    /// cell) are NOT film fluxes and are not included — the heat entering the wall from a side
    /// is this flux plus that side's gain — and neither is the latent heat carried by the vapour
    /// flux through the film ([`Surface::moisture_flux`]).
    ///
    /// [`Surface::heat_flux_films`] gives the convective / longwave split and
    /// [`Surface::heat_flux_convective`] the film to the air alone (an air-node balance needs
    /// that one, not this).
    ///
    /// ## Input
    /// - `location`: boundary location (exterior or interior).
    ///
    /// ## Output
    /// - `avg_flux`: average total film heat flux across the surface boundary [W/m²].
    pub fn heat_flux(&self, location: BoundaryLocation) -> Result<EnergyFluxDensity<f64>> {
        Ok(self.heat_flux_films(location)?.total())
    }

    /// The CONVECTIVE film alone (surface ↔ air, environment -> surface) [W/m²].
    ///
    /// For an air-node energy balance in a zone model whose radiative exchange goes to another
    /// node (the radiant star node / the other surfaces): only this part belongs to the air.
    /// Type 395 outputs 8 / 9 are NOT this — they are [`Surface::heat_flux`], the total.
    pub fn heat_flux_convective(
        &self,
        location: BoundaryLocation,
    ) -> Result<EnergyFluxDensity<f64>> {
        Ok(self.heat_flux_films(location)?.convective)
    }

    /// Area-averaged heat flux through the films of one face, split into the convective and the
    /// longwave film (environment -> surface) [W/m²]. See [`FilmHeatFlux`].
    ///
    /// For every Virtual (surface) cell of `location` the boundary cells connected to it along x
    /// are walked — exactly the neighbours `solver::adi::eliminate_surface_node` applies to the
    /// node, per column — and each ACTIVE film (`htc > ECLIPSE`) contributes
    /// `h·(T_env − T_surf)·A`: `Boundary::Convection` to `convective`, `Radiation` / `Sky` /
    /// `Ground` to `radiative`. Inactive cells (exterior Radiation / Sky / Ground placeholders,
    /// the interior Radiation cell of a legacy deck) contribute exactly nothing.
    pub fn heat_flux_films(&self, location: BoundaryLocation) -> Result<FilmHeatFlux> {
        let mut convective: Power<f64> = 0.0 * watts;
        let mut radiative: Power<f64> = 0.0 * watts;
        let mut total_area: Area<f64> = 0.0 * square_meter;

        for idx in self.virtual_cells(location.clone()) {
            let surf = self
                .cells
                .get(*idx)
                .with_context(|| format!("Failed to get virtual cell {:?}", idx))?;
            let cell_area = surf.size.area_along(0);
            let mut has_active_film = false;

            for film_idx in self.axis_neighbors(*idx, 0) {
                let film = self
                    .cells
                    .get(film_idx)
                    .with_context(|| format!("Failed to get boundary cell {:?}", film_idx))?;
                // Material neighbours are conduction, not films.
                let CellType::Boundary(kind, _) = &film.cell_type else {
                    continue;
                };
                if film.htc <= ECLIPSE * w_per_m2_k {
                    continue;
                }
                let delta_t = film.cell_temp - surf.cell_temp;
                let flux = film.htc * delta_t * cell_area;
                match kind {
                    Boundary::Convection => convective += flux,
                    Boundary::Radiation | Boundary::Sky | Boundary::Ground => radiative += flux,
                    Boundary::Virtual => continue,
                }
                has_active_film = true;
            }
            if has_active_film {
                total_area += cell_area;
            }
        }

        Ok(if total_area > ECLIPSE * square_meter {
            FilmHeatFlux {
                convective: convective / total_area,
                radiative: radiative / total_area,
            }
        } else {
            debug!(
                "Total area of the active films for heat flux for {} calculation is too small: {:?}. Returning zero flux.",
                location, total_area
            );
            FilmHeatFlux::default()
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

            // Convective VAPOUR flux through the air film: vtc·(vp_air − vp_surf).
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

    /// Material cell indices as [i,j,k] triplets, in cells.indexed_iter() order.
    /// Used by the block-tridiagonal assembler to walk the wall in physical order
    /// without having to re-filter on every iteration. Cached state is intentionally
    /// not stored here — the caller can memoize across an outer iteration.
    pub fn material_cell_indices(&self) -> Vec<[usize; 3]> {
        let mut idx = Vec::new();
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            if matches!(cell.cell_type, CellType::Material(_)) {
                idx.push([x, y, z]);
            }
        }
        idx
    }

    /// Material-cell lines parallel to `axis` (0=x,1=y,2=z). Each line is the set of
    /// material cells sharing the other two coordinates, ordered by increasing coord
    /// along `axis`. Drives the per-axis ADI sweeps.
    pub fn material_lines_along(&self, axis: usize) -> Vec<Vec<[usize; 3]>> {
        use std::collections::BTreeMap;
        // key = the two fixed coords (the non-`axis` indices), value = cells on that line
        let mut lines: BTreeMap<(usize, usize), Vec<[usize; 3]>> = BTreeMap::new();
        for ((x, y, z), cell) in self.cells.indexed_iter() {
            if !matches!(cell.cell_type, CellType::Material(_)) {
                continue;
            }
            let key = match axis {
                0 => (y, z),
                1 => (x, z),
                2 => (x, y),
                _ => continue,
            };
            lines.entry(key).or_default().push([x, y, z]);
        }
        let mut out: Vec<Vec<[usize; 3]>> = lines.into_values().collect();
        for line in &mut out {
            line.sort_by_key(|idx| idx[axis]);
        }
        out
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

        // Liquid (capillary) coupling exists ONLY when BOTH cells transport liquid.
        // Ported from EnergyPlus HeatBalanceHAMTManager.cc:1482 (`if (rhr1 * rhr2 > 0)`)
        // and Fortran Type395_module.f90:2608 (`IF(rhr1*rhr2>0)`): the liquid flux is
        // included only if both half-resistances are nonzero, i.e. both cells have
        // liquid transport. At an air↔material interface (one cell is non-liquid) the
        // product is zero, so there is NO liquid coupling across that face.
        let cell1_liquid = cell1.dw > 1e-20 * m2_per_s && cell1.dwd_phi > 1e-20 * kg_per_m3;
        let cell2_liquid = cell2.dw > 1e-20 * m2_per_s && cell2.dwd_phi > 1e-20 * kg_per_m3;
        // dist2 is already axis-aware (extent_along(axis) * 0.5 above)
        let liquid: MassResistance<f64> = if cell1_liquid && cell2_liquid {
            let rh_r1 = dist1 / (overlap * cell1.dw * cell1.dwd_phi);
            let rh_r2 = dist2 / (overlap * cell2.dw * cell2.dwd_phi);
            rh_r1 + rh_r2
        } else {
            // No coupling sentinel (matches the per-cell else-branch convention);
            // callers' `r.liquid > 1e-13` guard then drops the liquid term.
            0.0 * s_per_kg
        };

        ResistanceTriple {
            therm: thermr1 + thermr2,
            vapor: vaporr1 + vaporr2,
            liquid,
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
mod material_index_tests {
    #[test]
    fn material_indices_and_lines_1d() {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let s = &mgr.surface_curr;
        let mat = s.material_cell_indices();
        assert!(mat.iter().all(|t| t[1] == 0 && t[2] == 0)); // all on x-axis
        assert!(mat.windows(2).all(|w| w[0][0] < w[1][0])); // x-ordered
        let n = mat.len();
        let lines_x = s.material_lines_along(0);
        assert_eq!(lines_x.len(), 1); // one x-line
        assert_eq!(lines_x[0].len(), n); // all material cells
        let lines_y = s.material_lines_along(1);
        assert_eq!(lines_y.len(), n); // n singleton y-lines
        assert!(lines_y.iter().all(|l| l.len() == 1));
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
        assert!(
            cells.len() >= 2,
            "fixture must have at least 2 material cells"
        );
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

        // ── 3. liquid is OPEN (zero coupling) when either cell lacks liquid transport ──
        // Product guard (EnergyPlus rhr1*rhr2>0): liquid coupling exists only when BOTH
        // cells transport liquid. With cell1.dw=0, the product is zero, so the liquid
        // resistance is the no-coupling sentinel 0.0 — NOT cell2's half-resistance.
        // Clone a cell with dw = 0 (below 1e-20 m²/s guard).
        let mut cell_no_dw = cell1.clone();
        cell_no_dw.dw = 0.0 * m2_per_s;
        let triple_no_dw =
            Surface::calculate_resistance_triple(&cell_no_dw, cell2, bc.pressure_air, 0);
        assert_eq!(
            triple_no_dw.liquid.value_unchecked(),
            0.0,
            "liquid must be 0 (no coupling) when cell1 lacks liquid transport (product guard)"
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

#[cfg(test)]
mod heat_flux_films_tests {
    //! `heat_flux_films` / `heat_flux` / `heat_flux_convective`: nothing is solved, the queries
    //! only read cell state, so the surface nodes are pinned to known temperatures and the
    //! boundary cells are filled by `apply_boundary_coefficients` from a Type 395 input vector.
    //! The end-to-end (steady-state U, accumulator, output 8) checks are in
    //! `tests/interior_film.rs`.
    use super::*;
    use crate::boundary::{BoundaryCondition, celsius_to_kelvin};
    use crate::manager::HamtManager;
    use crate::solver::test_fixtures::{build_fixture_manager, build_fixture_manager_2d};
    use crate::uom::units::w_per_m2;

    /// HCONVI / combined film [W/(m²·K)].
    const H_CONV: f64 = 8.65;

    /// 14-input deck: exterior 0 °C with 10 W/(m²·K); interior TAIR 22 °C with
    /// `h_conv_w` as HCONVI and TMR 17 °C (TSTAR = 25 °C is not used by the films).
    fn inputs_14(h_conv_w: f64) -> [f64; 14] {
        #[rustfmt::skip]
        let inputs = [
            25.0, 0.0, 50.0, 50.0, 30.0, 36.0, 0.0, 0.0, 1e-8, 1e-12, 0.0,
            22.0,           // 11 TAIR
            h_conv_w * 3.6, // 12 HCONVI [kJ/(h m2 K)]
            17.0,           // 13 TMR
        ];
        inputs
    }

    /// Legacy 12-input deck: ONE combined interior film (`h_w`) to TSTAR = 22 °C.
    fn inputs_12(h_w: f64) -> [f64; 12] {
        #[rustfmt::skip]
        let inputs = [22.0, 0.0, 50.0, 50.0, h_w * 3.6, 36.0, 0.0, 0.0, 1e-8, 1e-12, 0.0, 22.0];
        inputs
    }

    /// Push `bc` onto the boundary cells and pin the interior surface node of column `j` to
    /// `t_si_c[j]` and every exterior surface node to `t_so_c`.
    fn prepare(mgr: &mut HamtManager, bc: BoundaryCondition, t_si_c: &[f64], t_so_c: f64) {
        mgr.boundary_condition = bc;
        mgr.surface_curr
            .apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
            .unwrap();
        for ((_, j, _), cell) in mgr.surface_curr.cells.indexed_iter_mut() {
            match &cell.cell_type {
                CellType::Virtual(BoundaryLocation::Interior) => {
                    cell.cell_temp = celsius_to_kelvin(t_si_c[j]);
                }
                CellType::Virtual(BoundaryLocation::Exterior) => {
                    cell.cell_temp = celsius_to_kelvin(t_so_c);
                }
                _ => {}
            }
        }
    }

    /// A flux as a bare number [W/m²], for comparison against bare-number expectations.
    fn wm2(q: EnergyFluxDensity<f64>) -> f64 {
        q.value_in(w_per_m2) // strip: test-side scalar
    }

    fn films(mgr: &HamtManager, loc: BoundaryLocation) -> FilmHeatFlux {
        mgr.surface_curr.heat_flux_films(loc).unwrap()
    }

    fn total(mgr: &HamtManager, loc: BoundaryLocation) -> EnergyFluxDensity<f64> {
        mgr.surface_curr.heat_flux(loc).unwrap()
    }

    fn convective(mgr: &HamtManager, loc: BoundaryLocation) -> EnergyFluxDensity<f64> {
        mgr.surface_curr.heat_flux_convective(loc).unwrap()
    }

    /// htc [W/(m²·K)] of the first boundary cell of this kind / location.
    fn film_htc(mgr: &HamtManager, kind: Boundary, loc: BoundaryLocation) -> f64 {
        mgr.surface_curr
            .cells
            .iter()
            .find(|c| matches!(&c.cell_type, CellType::Boundary(b, l) if *b == kind && *l == loc))
            .map(|c| c.htc.value_in(w_per_m2_k)) // strip: test-side scalar
            .expect("boundary cell")
    }

    /// The pre-change query, `htc·ΔT·A / A` of the Convection cell only (1-D, single column):
    /// the reference every ONE-film face must still reproduce bit for bit.
    fn convective_only_reference(mgr: &HamtManager, loc: BoundaryLocation) -> f64 {
        let cells = &mgr.surface_curr.cells;
        let conv = cells
            .iter()
            .find(|c| matches!(&c.cell_type, CellType::Boundary(Boundary::Convection, l) if *l == loc))
            .expect("Convection cell");
        let node = cells
            .iter()
            .find(|c| matches!(&c.cell_type, CellType::Virtual(l) if *l == loc))
            .expect("Virtual cell");
        let area = node.size.area_along(0);
        wm2((conv.htc * (conv.cell_temp - node.cell_temp) * area) / area)
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    /// Interior of a 14-input deck: two active films, each against its OWN environment
    /// temperature (TAIR for the convective one, TMR for the longwave one).
    #[test]
    fn interior_14_input_splits_into_convective_and_longwave_film() {
        let mut mgr = build_fixture_manager(6);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_14(H_CONV));
        prepare(&mut mgr, bc, &[12.0], 5.0);
        let h_rad = film_htc(&mgr, Boundary::Radiation, BoundaryLocation::Interior);
        assert!(
            (4.0..6.0).contains(&h_rad),
            "h_rad = 4σεT_m³ at ≈ 18.5 °C is ≈ 5.06 W/(m²·K), got {h_rad}"
        );

        let f = films(&mgr, BoundaryLocation::Interior);
        let (conv, rad) = (wm2(f.convective), wm2(f.radiative));
        assert!(
            rel(conv, H_CONV * (22.0 - 12.0)) < 1e-12,
            "convective film {conv}"
        );
        assert!(
            rel(rad, h_rad * (17.0 - 12.0)) < 1e-12,
            "longwave film {rad}"
        );
        assert_eq!(wm2(f.total()), conv + rad);

        // `heat_flux` is the total, `heat_flux_convective` the film to the air alone.
        let total = total(&mgr, BoundaryLocation::Interior);
        let conv_only = convective(&mgr, BoundaryLocation::Interior);
        assert_eq!(total, f.total());
        assert_eq!(conv_only, f.convective);
        assert!(wm2(total) > wm2(conv_only));

        // report() states the total (what Type 395 output 8 is built from).
        let rep = mgr.surface_curr.report(&mgr.boundary_condition).unwrap();
        assert_eq!(rep.q_surf_int, total);
    }

    /// One-film faces reproduce the pre-change value BIT FOR BIT: legacy interior (inert
    /// Radiation cell) and the exterior of every deck (inert Radiation / Sky / Ground cells).
    #[test]
    fn one_film_faces_reproduce_the_convective_only_value_exactly() {
        // Legacy 12-input deck, interior and exterior.
        let mut mgr = build_fixture_manager(6);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_12(H_CONV));
        prepare(&mut mgr, bc, &[12.0], 5.0);
        for loc in [BoundaryLocation::Interior, BoundaryLocation::Exterior] {
            let f = films(&mgr, loc.clone());
            let reference = convective_only_reference(&mgr, loc.clone());
            assert_ne!(reference, 0.0, "{loc}: the set-up must drive a flux");
            assert_eq!(wm2(f.radiative), 0.0, "{loc}: no longwave film");
            assert_eq!(wm2(f.convective), reference, "{loc} convective");
            assert_eq!(
                wm2(total(&mgr, loc.clone())),
                reference,
                "{loc} total (legacy deck)"
            );
        }

        // 14-input deck: the exterior is still ONE film; the interior convective film alone
        // is still the old number (only the total grew by the longwave film).
        let mut mgr = build_fixture_manager(6);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_14(H_CONV));
        prepare(&mut mgr, bc, &[12.0], 5.0);
        let ext = films(&mgr, BoundaryLocation::Exterior);
        let ext_ref = convective_only_reference(&mgr, BoundaryLocation::Exterior);
        assert_ne!(ext_ref, 0.0);
        assert_eq!(
            wm2(ext.radiative),
            0.0,
            "exterior: Radiation/Sky/Ground inert"
        );
        assert_eq!(wm2(ext.total()), ext_ref, "exterior total");
        assert_eq!(
            wm2(convective(&mgr, BoundaryLocation::Interior)),
            convective_only_reference(&mgr, BoundaryLocation::Interior),
            "interior convective film alone"
        );
    }

    /// Several columns: every column's node is paired with its own films and the result is the
    /// area-weighted average. Equal columns at 10 / 12 / 14 °C average to the 12 °C flux.
    #[test]
    fn multi_column_face_is_the_area_average_of_per_column_films() {
        let mut mgr = build_fixture_manager_2d(6, 3, false);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_14(H_CONV));
        prepare(&mut mgr, bc, &[10.0, 12.0, 14.0], 5.0);
        let h_rad = film_htc(&mgr, Boundary::Radiation, BoundaryLocation::Interior);
        let f = films(&mgr, BoundaryLocation::Interior);
        assert!(
            rel(wm2(f.convective), H_CONV * (22.0 - 12.0)) < 1e-10,
            "convective {:?}",
            f.convective
        );
        assert!(
            rel(wm2(f.radiative), h_rad * (17.0 - 12.0)) < 1e-10,
            "longwave {:?}",
            f.radiative
        );
    }

    /// A face whose only active film is the longwave one is still reported (the convective-only
    /// query used to return 0 here), and a face with NO active film reports zero.
    #[test]
    fn longwave_only_and_film_free_faces() {
        // HCONVI = 0: the Convection cell is inert, the Radiation cell (14-input deck) is not.
        let mut mgr = build_fixture_manager(6);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_14(0.0));
        prepare(&mut mgr, bc, &[12.0], 5.0);
        let h_rad = film_htc(&mgr, Boundary::Radiation, BoundaryLocation::Interior);
        let f = films(&mgr, BoundaryLocation::Interior);
        assert_eq!(wm2(f.convective), 0.0);
        assert!(rel(wm2(f.radiative), h_rad * (17.0 - 12.0)) < 1e-12);
        assert_eq!(f.total(), f.radiative);

        // Legacy deck with hconv1 = 0: no active interior film at all.
        let mut mgr = build_fixture_manager(6);
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs_12(0.0));
        prepare(&mut mgr, bc, &[12.0], 5.0);
        assert_eq!(
            films(&mgr, BoundaryLocation::Interior),
            FilmHeatFlux::default()
        );
        assert_eq!(wm2(total(&mgr, BoundaryLocation::Interior)), 0.0);
    }
}
