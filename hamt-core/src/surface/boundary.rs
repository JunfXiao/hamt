use crate::uom::{
    Quantity,
    dimensions::{HeatTransferCoeff, VaporTransportCoeff},
    units::{kelvin, kg_per_m2_s_pa, pascals, square_meter, w_per_m2_k},
};

use super::*;

use anyhow::Result;

impl Surface {
    /// Push boundary htc/vtc/temp/rh/vp into boundary cells before property updates.
    ///
    /// The loop iterates ALL cells via `indexed_iter_mut` so the same logic applies
    /// to every (y,z) column in multi-column 2-D grids. For a (N,1,1) wall this is
    /// identical to the old `for i in 0..nx` loop over `cells[[i,0,0]]`.
    ///
    /// Interior Radiation `h_rad` linearisation uses a GLOBAL area-averaged interior
    /// surface material temperature (computed once over
    /// `interior_surface_material_idx_set` before the loop) as the surface proxy.
    /// For a (N,1,1) wall the set has one element so the global mean equals the
    /// single cell temperature — byte-identical to the previous single-cell lookup.
    pub(crate) fn apply_boundary_coefficients(
        &mut self,
        bc: &BoundaryCondition,
        _settings: &HamtSettings,
    ) -> Result<()> {
        // ── Global area-averaged interior surface material temperature ─────────
        // Used as the surface proxy T_s in h_rad = 4σε((T_s+T_MR)/2)³.
        // For (N,1,1): set has one element → mean == that cell's temperature
        // (byte-identical to the old `interior_mat_temp.map(|i| cells[[i,0,0]])` path).
        let interior_mat_temp_global: Option<Temperature<f64>> = {
            let set = &self.interior_surface_material_idx_set;
            if set.is_empty() {
                None
            } else {
                let mut sum_t = 0.0_f64; // strip: area-weighted mean over raw kelvin values
                let mut sum_a = 0.0_f64;
                for &idx in set {
                    if let Some(cell) = self.cells.get(idx) {
                        let a = cell.size.area_along(0).value_in(square_meter); // strip: weight
                        sum_t += cell.cell_temp.value_in(kelvin) * a;
                        sum_a += a;
                    }
                }
                if sum_a > 0.0 {
                    Some((sum_t / sum_a) * kelvin)
                } else {
                    None
                }
            }
        };

        // ── Per-cell loop over all cells ──────────────────────────────────────
        // `indexed_iter_mut` visits every [x,y,z] cell. For (N,1,1) this is
        // exactly the same sequence as `for i in 0..nx { cells[[i,0,0]] }`.
        for ((_, _, _), cell) in self.cells.indexed_iter_mut() {
            if cell.is_boundary() {
                if cell.is_exterior() {
                    cell.htc = bc.heat_transfer_coeff_ext;
                    cell.cell_temp = bc.temp_outside_air;
                    cell.cell_rh = bc.rh_outside_air;
                    // Use the dedicated moisture temperature (actual outside air T) for VP_bc,
                    // so the VP boundary condition is physically correct even when cell_temp
                    // is set to the star-node temperature for the heat convection path.
                    let psat = physics::psy_psat_fn_temp(bc.temp_outside_air_moisture);
                    let rh = cell.cell_rh;
                    let vtc = if matches!(
                        cell.cell_type,
                        CellType::Boundary(crate::cell::Boundary::Convection, _)
                    ) {
                        if bc.vapor_transport_coeff_ext > VaporTransportCoeff::<f64>::zero() {
                            bc.vapor_transport_coeff_ext
                        } else {
                            let rho_v = physics::vd_from_t_rh(cell.cell_temp, rh);
                            let denom = psat * rh;
                            if denom > Quantity(1.0e-9) {
                                bc.mass_convection_coeff_ext * rho_v / denom
                            } else {
                                Quantity(0.0)
                            }
                        }
                    } else {
                        Quantity(0.0)
                    };
                    cell.vtc = vtc;
                    cell.cell_vp = rh * psat;
                } else {
                    match &cell.cell_type {
                        CellType::Boundary(
                            crate::cell::Boundary::Convection,
                            BoundaryLocation::Interior,
                        ) => {
                            // Pure convective HTC (HCONVI). Paired with TAIR on the heat side,
                            // shared with VP_bc on the moisture side. In separated-conv-rad mode
                            // (14-input deck) temp_inside_air_moisture = TAIR (NType 1);
                            // in legacy 11-input mode it falls back to temp_inside_air = TSTAR,
                            // preserving byte-identical single-potential behaviour.
                            cell.htc = bc.heat_transfer_coeff_conv_int;
                            cell.cell_temp = bc.temp_inside_air_moisture;
                            cell.cell_rh = bc.rh_inside_air;
                            let psat = physics::psy_psat_fn_temp(bc.temp_inside_air_moisture);
                            let rh = cell.cell_rh;
                            let vtc = if bc.vapor_transport_coeff_int
                                > VaporTransportCoeff::<f64>::zero()
                            {
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
                        CellType::Boundary(
                            crate::cell::Boundary::Radiation,
                            BoundaryLocation::Interior,
                        ) => {
                            // Linearised longwave exchange against mean radiant temperature.
                            // h_rad = 4 σ ε T_m³, T_m ≈ mean of surface and TMR.
                            // Uses the GLOBAL area-averaged interior surface material temperature
                            // (computed once before this loop over interior_surface_material_idx_set).
                            // For (N,1,1) the global mean equals the single interior cell temp —
                            // byte-identical to the previous single-cell lookup.
                            // Legacy decks (<14 Type395 inputs) keep using the combined
                            // REQV-derived h_conv on the Convection cell; the Radiation cell
                            // must then be inert (h_rad = 0) so the refactor is byte-identical
                            // to the pre-refactor single-potential behaviour.
                            let h_rad = if bc.has_separate_radiant_int {
                                let t_s = interior_mat_temp_global
                                    .unwrap_or(bc.temp_inside_mean_radiant);
                                let t_m_k = (t_s + bc.temp_inside_mean_radiant) * 0.5;
                                // strip: `.powi(3)` is only defined on the inner f64; the
                                // result is re-tagged as HeatTransferCoeff via `* w_per_m2_k`.
                                let t_m_cubed = t_m_k.value_in(kelvin).powi(3);
                                4.0 * physics::SIGMA * physics::EPSILON * t_m_cubed * w_per_m2_k
                            } else {
                                HeatTransferCoeff::<f64>::zero()
                            };
                            cell.htc = h_rad;
                            cell.cell_temp = bc.temp_inside_mean_radiant;
                            cell.vtc = VaporTransportCoeff::<f64>::zero();
                            cell.cell_vp = 0.0 * pascals;
                            cell.cell_rh = bc.rh_inside_air; // unused, kept valid
                        }
                        _ => {
                            // Fallback: legacy combined behaviour for any other interior
                            // boundary cell type not explicitly handled above.
                            cell.htc = bc.heat_transfer_coeff_int;
                            cell.cell_temp = bc.temp_inside_air;
                            cell.cell_rh = bc.rh_inside_air;
                            let psat = physics::psy_psat_fn_temp(bc.temp_inside_air_moisture);
                            let rh = cell.cell_rh;
                            cell.vtc = VaporTransportCoeff::<f64>::zero();
                            cell.cell_vp = rh * psat;
                        }
                    }
                }
            } else {
                cell.vtc = VaporTransportCoeff::<f64>::zero();
                cell.htc = HeatTransferCoeff::<f64>::zero();
            }
        }
        Ok(())
    }
}
