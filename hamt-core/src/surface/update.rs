use crate::{
    parallel::{par_or_seq, par_or_seq_mut}, uom::{
        Quantity, constants::{WATER_LATENT_HEAT, WATER_SPECIFIC_HEAT}, dimensions::{
            HeatCapacity, HeatTransferCoeff, Mass, MassFlux, MassFluxPerTemperature, Power, SpecificEnergy, ThermalConductance, Time, VaporConductance, VaporTransportCoeff
        }, units::{
            dimensionless, j_per_kg, k_per_w, kelvin, kg_per_m2_s_pa, kg_per_m3, kg_per_s,
            kilograms, pa_s_per_kg, pascals, seconds, w_per_k, w_per_m2_k, watts
        }
    }
};

use super::*;

use anyhow::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;


/// Blend (next, curr) by the cell's relaxation factor, clamp RH to [0, 1.01],
/// and recompute dependent state. Shared by the BlockJacobi write-back and the
/// BlockTridiag scatter.
fn relax_write_cell(
    cell: &mut Cell,
    next_t: Temperature<f64>,
    next_rh: Dimensionless<f64>,
    curr_t: Temperature<f64>,
    curr_rh: Dimensionless<f64>,
    bc: &BoundaryCondition,
    settings: &HamtSettings,
) {
    let alpha = cell.relaxation_factor;
    let relaxed_t = if next_t.is_finite() { next_t * alpha + curr_t * (1.0 - alpha) } else { curr_t };
    let relaxed_rh = if next_rh.is_finite() { next_rh * alpha + curr_rh * (1.0 - alpha) } else { curr_rh };
    cell.cell_temp = relaxed_t;
    cell.cell_rh = relaxed_rh.clamp(Quantity(0.0), Quantity(1.01));
    cell.calc_dependent_state(bc, settings);
}

/// Conductance-weighted neighbour sums accumulated in a single pass over a
/// cell's neighbours (see [`Surface::accumulate_neighbor_sums`]). These feed
/// both the coupled 2×2 block solve and the legacy staggered scalar solve.
///
///   Heat:    `oorsum` (Σ 1/R_th), `torsum` (Σ T_nbr/R_th)
///   Vapor:   `vapor_diffusion_net_flux` (Σ (P_nbr−P)/R_v), `vpoosum` (Σ 1/R_v),
///            `vporsum` (Σ P_nbr/R_v)
///   Liquid:  `phioosum` (Σ 1/R_φ), `phiorsum` (Σ φ_nbr/R_φ)
pub struct NeighborSums {
    pub oorsum: ThermalConductance<f64>,
    pub torsum: Power<f64>,
    pub vapor_diffusion_net_flux: MassFlux<f64>,
    pub vpoosum: VaporConductance<f64>,
    pub vporsum: MassFlux<f64>,
    pub phioosum: MassFlux<f64>,
    pub phiorsum: MassFlux<f64>,
}

/// Cell-local inertial capacities for the implicit-Euler step.
pub struct CellInertia {
    /// Effective heat capacity `(ρ·c_dry + w·c_w)·V` [J/K].
    pub tcap: HeatCapacity<f64>,
    /// Moisture capacity `ξ·V = (∂w/∂φ)·V` [kg], floored to avoid singularity.
    pub wcap: Mass<f64>,
}

/// Cell-local heat sources for the energy balance.
pub struct CellSources {
    /// User-defined additional heat deposited at this cell [W].
    pub q_adds: Power<f64>,
    /// Absorbed boundary radiation deposited into the surface-adjacent cell [W].
    pub q_boundary: Power<f64>,
}

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

        // --- Combined Heat and Moisture Step ---
        // Note: a pre-step update_cell_properties call would be redundant.
        // The solver reads dependent state (vp_sat, d_psat_d_t, w_thermal_c, mu, dw,
        // dwd_phi, h_sorp_diff) from `surface_curr`, not from `self`. `surface_curr`
        // was left in a consistent state by the previous iteration's post-step call
        // (or by tick_timestep at iter=0). Boundary cells overwritten in
        // apply_boundary_coefficients above are not read from `self` during the step.
        self.calculate_physics_step(settings, bc, surface_prev, surface_curr)?;

        // --- Update moisture-dependent properties for all cells ---
        self.update_cell_properties(bc, settings)?;

        Ok(())
    }

    /// Copy the current iteration state into `surface_next` before property updates.
    pub fn sync_state_from(&mut self, surface_curr: &Surface) -> Result<()> {
        if let (Some(dst), Some(src)) = (self.cells.as_slice_mut(), surface_curr.cells.as_slice()) {
            par_or_seq_mut(dst)
                .zip(par_or_seq(src))
                .for_each(|(cell, ref_cell)| {
                    *cell = ref_cell.clone();
                    cell.sync_state_from(ref_cell);
                });
        } else {
            for (cell, ref_cell) in self.cells.iter_mut().zip(surface_curr.cells.iter()) {
                *cell = ref_cell.clone();
                cell.sync_state_from(ref_cell);
            }
        }
        Ok(())
    }

    /// Update dependent cell properties based on the current iteration's Temp and RH.
    pub fn update_cell_properties(
        &mut self,
        bc: &BoundaryCondition,
        settings: &HamtSettings,
    ) -> Result<()> {
        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).for_each(|cell| cell.calc_dependent_state(bc, settings));
        } else {
            for cell in self.cells.iter_mut() {
                cell.calc_dependent_state(bc, settings);
            }
        }
        Ok(())
    }

    /// Push boundary htc/vtc/temp/rh/vp into boundary cells before property updates.
    pub(crate) fn apply_boundary_coefficients(
        &mut self,
        bc: &BoundaryCondition,
        _settings: &HamtSettings,
    ) -> Result<()> {
        let nx = self.cells.raw_dim()[0];
        let interior_mat_idx = self.interior_surface_material_idx;
        let interior_mat_temp: Option<Temperature<f64>> =
            interior_mat_idx.map(|i| self.cells[[i, 0, 0]].cell_temp);
        for i in 0..nx {
            let cell = &mut self.cells[[i, 0, 0]];
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
                    let vtc = if matches!(cell.cell_type, CellType::Boundary(crate::cell::Boundary::Convection, _)) {
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
                        CellType::Boundary(crate::cell::Boundary::Convection, BoundaryLocation::Interior) => {
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
                        CellType::Boundary(crate::cell::Boundary::Radiation, BoundaryLocation::Interior) => {
                            // Linearised longwave exchange against mean radiant temperature.
                            // h_rad = 4 σ ε T_m³, T_m ≈ mean of surface and TMR.
                            // Use the adjacent interior material cell's current temperature as the
                            // surface proxy.
                            // Legacy decks (<14 Type395 inputs) keep using the combined
                            // REQV-derived h_conv on the Convection cell; the Radiation cell
                            // must then be inert (h_rad = 0) so the refactor is byte-identical
                            // to the pre-refactor single-potential behaviour.
                            let h_rad = if bc.has_separate_radiant_int {
                                let t_s = interior_mat_temp.unwrap_or(bc.temp_inside_mean_radiant);
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

    /// Accumulate the seven conductance-weighted neighbour sums for cell `i`
    /// in a single pass. `surf_curr` provides the lagged neighbour state read
    /// during a block-Jacobi / Picard sweep. The guards
    /// (`1e-12*k_per_w`, `1e-12*pa_s_per_kg`, the liquid `r_total` bounds)
    /// drop degenerate / zero-coupling interfaces.
    fn accumulate_neighbor_sums(
        &self,
        i: usize,
        surf_curr: &Surface,
        bc: &BoundaryCondition,
    ) -> NeighborSums {
        let nx = self.cells.raw_dim()[0];
        let cell_curr = &surf_curr.cells[[i, 0, 0]];

        let mut oorsum: ThermalConductance<f64> = 0.0 * w_per_k;
        let mut torsum: Power<f64> = 0.0 * watts;
        let mut vapor_diffusion_net_flux: MassFlux<f64> = 0.0 * kilograms / seconds;
        let mut vpoosum: VaporConductance<f64> = 0.0 * kg_per_s / pascals;
        let mut vporsum: MassFlux<f64> = 0.0 * kg_per_s;
        let mut phioosum: MassFlux<f64> = 0.0 * kg_per_s;
        let mut phiorsum: MassFlux<f64> = 0.0 * kg_per_s;

        let curr_vp_pa = cell_curr.cell_vp;

        for nbr_i in surf_curr.neighbor_indices(i, nx) {
            let nbr_curr = &surf_curr.cells[[nbr_i, 0, 0]];

            let r = Self::calculate_resistance_triple(cell_curr, nbr_curr, bc.pressure_air);

            if r.therm > 1e-12 * k_per_w {
                oorsum += 1.0 / r.therm;
                torsum += nbr_curr.cell_temp / r.therm;
            }

            if r.vapor > 1e-12 * pa_s_per_kg {
                let nbr_vp = nbr_curr.cell_vp;
                let vc = 1.0 / r.vapor;
                vapor_diffusion_net_flux += (nbr_vp - curr_vp_pa) / r.vapor;
                vpoosum += vc;
                vporsum += nbr_vp * vc;
            }

            let r_total: Quantity<f64, _> = r.liquid * dimensionless;
            if r_total < f64::MAX * dimensionless / kg_per_s
                && r_total > 1e-13 * dimensionless / kg_per_s
            {
                let cond: MassFlux<f64> = 1.0 / r_total;
                phioosum += cond;
                phiorsum += nbr_curr.cell_rh * cond;
            }
        }

        NeighborSums {
            oorsum,
            torsum,
            vapor_diffusion_net_flux,
            vpoosum,
            vporsum,
            phioosum,
            phiorsum,
        }
    }

    /// Cell-local inertial capacities `(tcap, wcap)` for the implicit step.
    fn cell_inertia(cell_curr: &Cell) -> CellInertia {
        let tcap = (cell_curr.density * cell_curr.spech
            + cell_curr.volumetric_water * WATER_SPECIFIC_HEAT)
            * cell_curr.volume;

        let wcap = if cell_curr.dwd_phi > 1e-10 * kg_per_m3 {
            cell_curr.dwd_phi * cell_curr.volume
        } else {
            debug!(
                "Cell has very low dwd_phi, using fallback moisture capacity to prevent instability"
            );
            1.0e-10 * kilograms
        };

        CellInertia { tcap, wcap }
    }

    /// Cell-local heat sources `(q_adds, q_boundary)` for cell `i`.
    fn cell_sources(
        &self,
        i: usize,
        surf_curr: &Surface,
        bc: &BoundaryCondition,
    ) -> CellSources {
        let q_weight = surf_curr
            .user_defined_cell_idx
            .as_ref()
            .map(|weights| {
                weights
                    .iter()
                    .filter(|(idx, _)| idx[0] == i)
                    .map(|(_, w)| w)
                    .sum::<f64>()
            })
            .unwrap_or(0.0);
        let q_adds = bc.q_addition_userdef * q_weight;

        // Deposit absorbed radiation into the boundary-adjacent material cell.
        let q_boundary: Power<f64> = if Some(i) == surf_curr.interior_surface_material_idx {
            bc.additional_energy_int
        } else if Some(i) == surf_curr.exterior_surface_material_idx {
            bc.additional_energy_ext
        } else {
            0.0 * watts
        };

        CellSources { q_adds, q_boundary }
    }

    /// Cell-local diagonal 2×2 block + cell-local RHS (capacities, Taylor cross,
    /// inertial + source biases). NEIGHBOUR-VALUE coupling (torsum / l_eff*vporsum
    /// into b_t; phiorsum / vporsum into b_p) is added by the caller.
    fn build_cell_block(
        sums: &NeighborSums,
        inertia: &CellInertia,
        sources: &CellSources,
        cell_curr: &Cell,
        cell_prev: &Cell,
        l_eff: SpecificEnergy<f64>,
        dt: Time<f64>,
    ) -> (crate::solver::block_2x2::Block2x2, (f64, f64)) {
        let phi_k = cell_curr.cell_rh;
        let cross: MassFluxPerTemperature<f64> = sums.vpoosum * phi_k * cell_curr.d_psat_d_t;
        let a_tt: ThermalConductance<f64> = sums.oorsum + inertia.tcap / dt + l_eff * cross;
        let a_tp: Power<f64> = l_eff * sums.vpoosum * cell_curr.vp_sat;
        let a_pt: MassFluxPerTemperature<f64> = cross;
        let a_pp: MassFlux<f64> = sums.phioosum + sums.vpoosum * cell_curr.vp_sat + inertia.wcap / dt;
        let b_t: Power<f64> = sources.q_adds + sources.q_boundary
            + inertia.tcap * cell_prev.cell_temp / dt
            + l_eff * (cross * cell_curr.cell_temp);
        let b_p: MassFlux<f64> = inertia.wcap * cell_prev.cell_rh / dt + cross * cell_curr.cell_temp;
        // strip: linear-algebra-kernel boundary (typed in, typed out at caller)
        (
            crate::solver::block_2x2::Block2x2 { a: a_tt.0, b: a_tp.0, c: a_pt.0, d: a_pp.0 },
            (b_t.0, b_p.0),
        )
    }

    /// Combined Heat and Moisture step: compute next T and RH for all cells.
    fn calculate_physics_step(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        let dt_s = settings.timestep;

        let nx = self.cells.raw_dim()[0];
        let indices: Vec<usize> = (0..nx).collect();

        let results: Vec<(Temperature<f64>, Dimensionless<f64>)> = par_or_seq(&indices)
            .map(|&i| {
                let cell_curr = &surface_curr.cells[[i, 0, 0]];

                if !cell_curr.is_boundary() {
                    let cell_prev = &surface_prev.cells[[i, 0, 0]];
                    let is_material = matches!(cell_curr.cell_type, CellType::Material(_));

                    // Conductance-weighted neighbour sums, cell-local inertia, and
                    // heat sources are extracted into testable helpers. The sums read
                    // lagged neighbour state from `surface_curr` (block-Jacobi sweep).
                    let sums = surface_curr.accumulate_neighbor_sums(i, surface_curr, bc);
                    let inertia = Self::cell_inertia(cell_curr);
                    let sources = surface_curr.cell_sources(i, surface_curr, bc);

                    let t_curr = cell_curr.cell_temp;
                    let vpsat_curr = cell_curr.vp_sat;
                    // dPsat/dT [Pa/K] pre-cached in calc_dependent_state (Magnus formula)
                    let d_psat_d_t = cell_curr.d_psat_d_t;

                    // =========================================================
                    // Solve for (T_next, RH_next)
                    // =========================================================
                    let (t_next, rh_next);

                    if settings.coupled_heat_moisture_enabled && is_material && settings.latent_heat_enabled {
                        // ── Symmetric 2×2 block solve (typed) ─────────────────────────────────
                        //
                        // Linearise  P_v,i^{k+1} ≈ Psat(T^k)·φ^{k+1} + φ^k·(dPsat/dT)·(T^{k+1}−T^k)
                        //
                        //  ┌ A_tt  A_tp ┐ ┌ T_next ┐   ┌ b_t ┐
                        //  └ A_pt  A_pp ┘ └ φ_next ┘ = └ b_p ┘
                        //
                        // `build_cell_block` assembles the cell-local diagonal block and
                        // the cell-local RHS (capacities, Taylor cross, inertial + source
                        // biases). The neighbour-VALUE coupling (torsum + l_eff·vporsum
                        // into b_t; phiorsum + vporsum into b_p) is added here at the caller.

                        // WATER_LATENT_HEAT is declared as a `constant` in unit_system!
                        // and is therefore a Unit, not a Quantity — coerce via `1.0 *`.
                        let l_eff: SpecificEnergy<f64> = 1.0 * WATER_LATENT_HEAT
                            + if settings.sorption_enthalpy_enabled {
                                cell_curr.h_sorp_diff
                            } else {
                                0.0 * j_per_kg
                            };

                        let phi_k = cell_curr.cell_rh;

                        let (diag, (b_t0, b_p0)) = Self::build_cell_block(
                            &sums, &inertia, &sources, cell_curr, cell_prev, l_eff, dt_s,
                        );

                        // Add the neighbour-VALUE RHS coupling.
                        // strip: kernel boundary
                        let b_t = b_t0 + sums.torsum.0 + (l_eff * sums.vporsum).0;
                        // strip: kernel boundary
                        let b_p = b_p0 + sums.phiorsum.0 + sums.vporsum.0;

                        // ── Linear algebra kernel ─────────────────────────────────────
                        // Cramer's rule on the 2×2 diagonal block + assembled RHS. We
                        // treat it like a BLAS call: arithmetic on SI magnitudes, re-tag
                        // the typed outputs on exit. Same singular fallback as before.
                        let det = diag.a * diag.d - diag.b * diag.c;

                        let (t_n_typed, p_n_typed): (Temperature<f64>, Dimensionless<f64>) =
                            if det.abs() > 1e-30 {
                                let t_n_v = (b_t * diag.d - diag.b * b_p) / det;
                                let p_n_v = (diag.a * b_p - diag.c * b_t) / det;
                                (t_n_v * kelvin, p_n_v * dimensionless)
                            } else {
                                // Near-singular (e.g. no vapor coupling): fall back to current
                                // state; the outer relaxation loop will converge next iteration.
                                (t_curr, phi_k)
                            };

                        t_next = if t_n_typed.is_finite() {
                            t_n_typed
                        } else {
                            t_curr
                        };
                        rh_next = if p_n_typed.is_finite() {
                            p_n_typed
                        } else {
                            cell_curr.cell_rh
                        };
                    } else {
                        // ── Legacy sequential scalar solve ────────────────────────────────────
                        // Reproduce the pre-coupling behaviour exactly when flag is false.
                        // Taylor Series Linearization for Vapor Pressure:
                        // P_v^{k+1} ≈ Psat(T^k)·φ^{k+1} + φ^k·(dPsat/dT)·(T^{k+1} − T^k)
                        let mut q_vapor: Power<f64> = 0.0 * watts;
                        if is_material && settings.latent_heat_enabled {
                            q_vapor = sums.vapor_diffusion_net_flux * WATER_LATENT_HEAT;
                            if settings.sorption_enthalpy_enabled {
                                q_vapor += sums.vapor_diffusion_net_flux * cell_curr.h_sorp_diff;
                            }
                            if q_vapor.abs() > settings.max_latent_heat {
                                // strip: signum() is only defined on the inner f64;
                                // we want a raw ±1.0 to re-tag the clamped magnitude.
                                q_vapor =
                                    q_vapor.value_in(watts).signum() * settings.max_latent_heat;
                            }
                        }

                        let denom_t: ThermalConductance<f64> = sums.oorsum + inertia.tcap / dt_s;
                        let t_next_s: Temperature<f64> = if denom_t > 1e-10 * w_per_k {
                            let inertial = inertia.tcap * cell_prev.cell_temp / dt_s;
                            (sums.torsum + q_vapor + sources.q_adds + sources.q_boundary + inertial)
                                / denom_t
                        } else {
                            t_curr
                        };
                        t_next = if t_next_s.is_finite() {
                            t_next_s
                        } else {
                            t_curr
                        };

                        // Moisture: thermal correction uses the just-computed t_next.
                        // delta_t : Temperature, d_psat_d_t : Pa/K, cell_rh : Dimensionless
                        // → p_sat_thermal_correction : Pressure (Pa)
                        // → thermal_source_term      : MassFlux (kg/s)
                        let delta_t = t_next - t_curr;
                        let p_sat_thermal_correction =
                            cell_curr.cell_rh * d_psat_d_t * delta_t;
                        let thermal_source_term = sums.vpoosum * p_sat_thermal_correction;
                        let denom_rh: MassFlux<f64> =
                            sums.phioosum + sums.vpoosum * vpsat_curr + inertia.wcap / dt_s;

                        let rh_next_s: Dimensionless<f64> = if denom_rh > 1e-20 * kg_per_s {
                            (sums.phiorsum + sums.vporsum + (inertia.wcap * cell_prev.cell_rh) / dt_s
                                - thermal_source_term)
                                / denom_rh
                        } else {
                            debug!(
                                "Cell {} has very low moisture conductance {:?}, using fallback RH to prevent instability",
                                i, denom_rh
                            );
                            cell_curr.cell_rh
                        };
                        rh_next = if rh_next_s.is_finite() {
                            rh_next_s
                        } else {
                            debug!(
                                "Cell {} has non-finite RH_next {:?}, using fallback RH to prevent instability",
                                i, rh_next_s
                            );
                            cell_curr.cell_rh
                        };
                    }

                    (t_next, rh_next)
                } else {
                    // Boundary cells: return the same T and RH that
                    // apply_boundary_coefficients() already set, so the
                    // post-step write loop does not overwrite them.
                    if cell_curr.is_exterior() {
                        (bc.temp_outside_air, bc.rh_outside_air)
                    } else {
                        match &cell_curr.cell_type {
                            CellType::Boundary(crate::cell::Boundary::Convection, BoundaryLocation::Interior) => {
                                (bc.temp_inside_air_moisture, bc.rh_inside_air)
                            }
                            CellType::Boundary(crate::cell::Boundary::Radiation, BoundaryLocation::Interior) => {
                                (bc.temp_inside_mean_radiant, bc.rh_inside_air)
                            }
                            _ => {
                                (bc.temp_inside_air, bc.rh_inside_air)
                            }
                        }
                    }
                }
            })
            .collect();

        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).enumerate().for_each(|(i, cell)| {
                let next_temp = results[i].0;
                let next_rh = results[i].1;
                let curr_temp = surface_curr.cells[[i, 0, 0]].cell_temp;
                let curr_rh = surface_curr.cells[[i, 0, 0]].cell_rh;
                relax_write_cell(cell, next_temp, next_rh, curr_temp, curr_rh, bc, settings);
            });
        } else {
            for i in 0..nx {
                let next_temp = results[i].0;
                let next_rh = results[i].1;
                let curr_temp = surface_curr.cells[[i, 0, 0]].cell_temp;
                let curr_rh = surface_curr.cells[[i, 0, 0]].cell_rh;
                relax_write_cell(
                    &mut self.cells[[i, 0, 0]],
                    next_temp, next_rh, curr_temp, curr_rh,
                    bc, settings,
                );
            }
        }
        Ok(())
    }

    /// Single outer BlockJacobi step using the direct block-tridiagonal solver.
    ///
    /// `self` is `surface_next` (writable). The method:
    ///   1. Sync `self` from `surface_curr`.
    ///   2. Apply BCs and refresh dependent state.
    ///   3. Assemble the block-tridiag system.
    ///   4. Solve via Thomas.
    ///   5. Scatter the solution into `self`'s material cells (with α
    ///      relaxation per `cell.relaxation_factor`).
    ///
    /// Boundary cells are left as-is (apply_boundary_coefficients
    /// owns them).
    pub fn calculate_iteration_block_tridiag(
        &mut self,
        settings: &HamtSettings,
        bc: &BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> Result<()> {
        self.sync_state_from(surface_curr)?;
        self.apply_boundary_coefficients(bc, settings)?;

        // Virtual cells aren't updated by `apply_boundary_coefficients`. BlockJacobi's
        // per-cell 2×2 solve covers them as a side-effect of iterating over all
        // non-boundary cells, but BlockTridiag's scatter writes only material
        // cells. Compute Virtual cells here as the conductance-weighted average
        // of their non-Virtual neighbours' state — the implicit-Euler answer
        // for a cell with negligible capacity. This produces a non-zero
        // Virtual ↔ Conv gradient that the moisture-flux query relies on.
        let nx = self.cells.raw_dim()[0];
        // Collect (virtual_li, [nbr_li, ...]) before any cell borrows to avoid
        // simultaneous &self and &self.cells conflicts in the inner loop.
        let virtual_nbr_pairs: Vec<(usize, Vec<usize>)> = (0..self.cells.len())
            .filter(|&li| matches!(self.cells[[li, 0, 0]].cell_type, crate::cell::CellType::Virtual(_)))
            .map(|li| {
                let nbrs: Vec<usize> = self.neighbor_indices(li, nx).collect();
                (li, nbrs)
            })
            .collect();

        let mut virtual_updates: Vec<(usize, f64, f64, f64)> = Vec::new();
        for (li, nbr_indices) in virtual_nbr_pairs {
            let cell = &self.cells[[li, 0, 0]];

            // strip: linear-algebra kernel boundary
            let mut sum_inv_v: f64 = 0.0;
            let mut sum_inv_v_vp: f64 = 0.0;
            let mut sum_inv_t: f64 = 0.0;
            let mut sum_inv_t_t: f64 = 0.0;

            for nbr_li in nbr_indices {
                let nbr = &self.cells[[nbr_li, 0, 0]];
                let (therm_r, vapor_r) = Surface::calculate_resistance_pair(
                    cell, nbr, bc.pressure_air,
                );
                if vapor_r > 1e-12 * pa_s_per_kg {
                    let inv_v = 1.0 / vapor_r.value_in(pa_s_per_kg);
                    sum_inv_v += inv_v;
                    sum_inv_v_vp += inv_v * nbr.cell_vp.value_in(pascals);
                }
                if therm_r > 1e-12 * k_per_w {
                    let inv_t = 1.0 / therm_r.value_in(k_per_w);
                    sum_inv_t += inv_t;
                    sum_inv_t_t += inv_t * nbr.cell_temp.value_in(kelvin);
                }
            }

            // Fall back to current values when no neighbour has non-zero coupling
            // (degenerate; should not happen for the standard 1-D wall layout).
            let new_t = if sum_inv_t > 0.0 {
                sum_inv_t_t / sum_inv_t
            } else {
                cell.cell_temp.value_in(kelvin)
            };
            let new_vp = if sum_inv_v > 0.0 {
                sum_inv_v_vp / sum_inv_v
            } else {
                cell.cell_vp.value_in(pascals)
            };
            let psat = crate::physics::psy_psat_fn_temp(new_t * kelvin)
                .value_in(pascals);
            let new_rh = if psat > 0.0 { (new_vp / psat).max(0.0).min(1.01) } else { 0.0 };
            virtual_updates.push((li, new_vp, new_t, new_rh));
        }

        for (li, vp, t, rh) in virtual_updates {
            let cell = &mut self.cells[[li, 0, 0]];
            // re-tag at kernel boundary
            cell.cell_vp = vp * pascals;
            cell.cell_temp = t * kelvin;
            cell.cell_rh = rh * dimensionless;
        }

        self.update_cell_properties(bc, settings)?;

        // Assemble from `self`, not `surface_curr`: the BC and Virtual cells on
        // `self` were just refreshed against the current iterate's surface
        // temperature, while the same cells on `surface_curr` still reflect the
        // previous iterate. Material-cell coefficients are identical between the
        // two (sync_state_from copied them and update_cell_properties is idempotent
        // on that data), so this only swaps lagged BC/Virtual coefficients for
        // current-iter ones — restoring near-Newton convergence.
        let system = crate::solver::assembly::assemble_1d(
            self, surface_prev, bc, settings,
        );
        let solution = crate::solver::block_tridiag::solve(
            &system.lower, &system.diag, &system.upper, &system.rhs,
        )
        .map_err(|e| anyhow::anyhow!("block-tridiag solve failed: {:?}", e))?;

        // Scatter solution back, applying per-cell α relaxation
        // (consistent with the BlockJacobi write-back behaviour).
        for (row, &li) in system.material_indices.iter().enumerate() {
            let (t_new_k, phi_new_raw) = solution[row];
            let curr_temp = surface_curr
                .cells
                .get([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?
                .cell_temp;
            let curr_rh = surface_curr
                .cells
                .get([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?
                .cell_rh;
            let cell = self
                .cells
                .get_mut([li, 0, 0])
                .ok_or_else(|| anyhow::anyhow!("invalid linear index {li}"))?;

            // re-tag at kernel boundary
            let t_new_typed = t_new_k * crate::uom::units::kelvin;
            let phi_new_typed = phi_new_raw * crate::uom::units::dimensionless;

            relax_write_cell(cell, t_new_typed, phi_new_typed, curr_temp, curr_rh, bc, settings);
        }

        Ok(())
    }

    /// Get neighbor indices for a 1D cell.
    pub(crate) fn neighbor_indices(&self, i: usize, nx: usize) -> impl Iterator<Item = usize> {
        let self_cell = self.cells.get([i, 0, 0]).unwrap();

        let left: Vec<usize> = if i > 0 {
            if matches!(
                self_cell.cell_type,
                CellType::Virtual(BoundaryLocation::Exterior)
            ) {
                (0..i).collect()
            } else if i > 1 {
                vec![i - 1]
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        let right = if matches!(
            self_cell.cell_type,
            CellType::Virtual(BoundaryLocation::Interior)
        ) {
            (i + 1..nx).collect()
        } else if i + 1 < nx {
            vec![i + 1]
        } else {
            vec![]
        };

        left.into_iter().chain(right)
    }
}

#[cfg(test)]
mod relax_write_cell_tests {
    use super::*;
    use crate::uom::units::{dimensionless, kelvin};

    fn make_material_cell() -> Cell {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(1);
        // Grab the first material cell from surface_curr.
        mgr.surface_curr
            .cells
            .iter()
            .find(|c| matches!(c.cell_type, crate::cell::CellType::Material(_)))
            .cloned()
            .expect("fixture must have at least one material cell")
    }

    #[test]
    fn relax_write_cell_blends_and_clamps() {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(1);
        let bc = &mgr.boundary_condition;
        let settings = &mgr.settings;

        // ── Case 1: normal blend ────────────────────────────────────────
        let mut cell = make_material_cell();
        cell.relaxation_factor = 0.5;
        let next_t   = 300.0 * kelvin;
        let curr_t   = 290.0 * kelvin;
        let next_rh  = 0.8 * dimensionless;
        let curr_rh  = 0.4 * dimensionless;
        relax_write_cell(&mut cell, next_t, next_rh, curr_t, curr_rh, bc, settings);
        let t_out  = cell.cell_temp.value_in(kelvin);
        let rh_out = cell.cell_rh.value();
        assert!((t_out - 295.0).abs() < 1e-9,  "expected 295 K, got {t_out}");
        assert!((rh_out - 0.6).abs() < 1e-9,   "expected 0.6 RH, got {rh_out}");

        // ── Case 2: next_rh = 2.0 → clamped to 1.01 ───────────────────
        let mut cell2 = make_material_cell();
        cell2.relaxation_factor = 0.5;
        let next_rh_high = 2.0 * dimensionless;   // blended = 0.5*2.0 + 0.5*0.4 = 1.2 > 1.01
        relax_write_cell(&mut cell2, next_t, next_rh_high, curr_t, curr_rh, bc, settings);
        let rh_clamped = cell2.cell_rh.value();
        assert!(
            (rh_clamped - 1.01).abs() < 1e-9,
            "expected clamp to 1.01, got {rh_clamped}"
        );

        // ── Case 3: next_t = NaN → cell_temp stays at curr_t ───────────
        let mut cell3 = make_material_cell();
        cell3.relaxation_factor = 0.5;
        let nan_t = f64::NAN * kelvin;
        relax_write_cell(&mut cell3, nan_t, next_rh, curr_t, curr_rh, bc, settings);
        let t_nan_fallback = cell3.cell_temp.value_in(kelvin);
        assert!(
            (t_nan_fallback - 290.0).abs() < 1e-9,
            "expected fallback to 290 K, got {t_nan_fallback}"
        );
    }
}

#[cfg(test)]
mod build_cell_block_tests {
    use super::*;
    use crate::uom::units::{
        dimensionless, j_per_k, j_per_kg, kelvin, kg_per_s, kilograms, pa_per_k, pascals,
        seconds, w_per_k, watts,
    };

    /// `build_cell_block` must reproduce the hand-derived typed 2×2 block and
    /// cell-local RHS, and the full Cramer solve (after adding the neighbour-value
    /// RHS terms) must match a direct 2×2 solve of the same system.
    #[test]
    fn build_cell_block_matches_reference() {
        // ── Known typed inputs ─────────────────────────────────────────
        let sums = NeighborSums {
            oorsum: 3.0 * w_per_k,
            torsum: 900.0 * watts,
            vapor_diffusion_net_flux: 0.0 * kg_per_s,
            vpoosum: 2.0e-7 * (kg_per_s / pascals),
            vporsum: 1.5e-4 * kg_per_s,
            phioosum: 5.0e-6 * kg_per_s,
            phiorsum: 2.0e-6 * kg_per_s,
        };
        let inertia = CellInertia {
            tcap: 4000.0 * j_per_k,
            wcap: 0.02 * kilograms,
        };
        let sources = CellSources {
            q_adds: 10.0 * watts,
            q_boundary: 5.0 * watts,
        };

        let cell_curr = Cell {
            cell_rh: 0.6 * dimensionless,
            cell_temp: 295.0 * kelvin,
            vp_sat: 2600.0 * pascals,
            d_psat_d_t: 160.0 * pa_per_k,
            ..Cell::default()
        };

        let cell_prev = Cell {
            cell_rh: 0.55 * dimensionless,
            cell_temp: 293.0 * kelvin,
            ..Cell::default()
        };

        let l_eff = 2.5e6 * j_per_kg;
        let dt = 3600.0 * seconds;

        // ── Hand-derived expected block + cell-local RHS (typed formulas) ──
        let phi_k = cell_curr.cell_rh;
        let cross: MassFluxPerTemperature<f64> = sums.vpoosum * phi_k * cell_curr.d_psat_d_t;
        let exp_a_tt: ThermalConductance<f64> = sums.oorsum + inertia.tcap / dt + l_eff * cross;
        let exp_a_tp: Power<f64> = l_eff * sums.vpoosum * cell_curr.vp_sat;
        let exp_a_pt: MassFluxPerTemperature<f64> = cross;
        let exp_a_pp: MassFlux<f64> =
            sums.phioosum + sums.vpoosum * cell_curr.vp_sat + inertia.wcap / dt;
        let exp_b_t: Power<f64> = sources.q_adds + sources.q_boundary
            + inertia.tcap * cell_prev.cell_temp / dt
            + l_eff * (cross * cell_curr.cell_temp);
        let exp_b_p: MassFlux<f64> =
            inertia.wcap * cell_prev.cell_rh / dt + cross * cell_curr.cell_temp;

        let (diag, (b_t0, b_p0)) = Surface::build_cell_block(
            &sums, &inertia, &sources, &cell_curr, &cell_prev, l_eff, dt,
        );

        let rel = |got: f64, exp: f64| (got - exp).abs() / exp.abs().max(1.0);
        assert!(rel(diag.a, exp_a_tt.0) < 1e-6, "a_tt: {} vs {}", diag.a, exp_a_tt.0);
        assert!(rel(diag.b, exp_a_tp.0) < 1e-6, "a_tp: {} vs {}", diag.b, exp_a_tp.0);
        assert!(rel(diag.c, exp_a_pt.0) < 1e-6, "a_pt: {} vs {}", diag.c, exp_a_pt.0);
        assert!(rel(diag.d, exp_a_pp.0) < 1e-6, "a_pp: {} vs {}", diag.d, exp_a_pp.0);
        assert!(rel(b_t0, exp_b_t.0) < 1e-6, "b_t0: {} vs {}", b_t0, exp_b_t.0);
        assert!(rel(b_p0, exp_b_p.0) < 1e-6, "b_p0: {} vs {}", b_p0, exp_b_p.0);

        // ── End-to-end: add neighbour terms + Cramer-solve, compare with a
        //    direct 2×2 solve of the same fully-assembled system. ───────────
        let b_t = b_t0 + sums.torsum.0 + (l_eff * sums.vporsum).0;
        let b_p = b_p0 + sums.phiorsum.0 + sums.vporsum.0;

        let det = diag.a * diag.d - diag.b * diag.c;
        assert!(det.abs() > 1e-30, "test matrix must be non-singular");
        let t_solver = (b_t * diag.d - diag.b * b_p) / det;
        let p_solver = (diag.a * b_p - diag.c * b_t) / det;

        // Independent direct solve via matrix inversion (different code path
        // than the inline Cramer numerators): x = M^-1 · (b_t, b_p).
        let inv = diag.try_inv(1e-30).expect("non-singular");
        let (t_ref, p_ref) = inv.mul_vec((b_t, b_p));

        assert!((t_solver - t_ref).abs() < 1e-9, "T solve mismatch: {t_solver} vs {t_ref}");
        assert!((p_solver - p_ref).abs() < 1e-9, "phi solve mismatch: {p_solver} vs {p_ref}");
    }
}

#[cfg(test)]
mod block_tridiag_tests {
    use super::*;

    fn fixture(n_div: usize) -> crate::hamt_manager::HamtManager {
        crate::solver::test_fixtures::build_fixture_manager(n_div)
    }

    /// Property: a single call to `calculate_iteration_block_tridiag`
    /// from a state ALREADY at equilibrium with the BCs is a fixed
    /// point — no material cell moves more than 1e-5 K or 1e-5 RH.
    /// Catches sign errors in the assembly→solve→write-back chain.
    #[test]
    fn one_iteration_at_equilibrium_is_no_op() {
        let mut mgr = fixture(6);
        // Drive cells to BCs (same shape as solver::assembly::tests).
        let bc_temp = mgr.boundary_condition.temp_inside_air;
        let bc_rh = mgr.boundary_condition.rh_inside_air;
        mgr.boundary_condition.temp_outside_air = bc_temp;
        mgr.boundary_condition.temp_outside_air_moisture = bc_temp;
        mgr.boundary_condition.rh_outside_air = bc_rh;
        for cell in mgr.surface_curr.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                cell.cell_temp = bc_temp;
                cell.cell_rh = bc_rh;
            }
        }
        for cell in mgr.surface_prev.cells.iter_mut() {
            if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                cell.cell_temp = bc_temp;
                cell.cell_rh = bc_rh;
            }
        }
        // Apply BCs and refresh dependent state in surface_curr/prev so
        // boundary cells (incl. virtual) reflect the symmetric BC.
        mgr.surface_curr.apply_boundary_coefficients(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_prev.apply_boundary_coefficients(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_curr.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();
        mgr.surface_prev.update_cell_properties(
            &mgr.boundary_condition, &mgr.settings,
        ).unwrap();

        // surface_next is scratch; clone from surface_curr.
        let mut surface_next = mgr.surface_curr.clone();
        surface_next.calculate_iteration_block_tridiag(
            &mgr.settings, &mgr.boundary_condition,
            &mgr.surface_prev, &mgr.surface_curr,
        ).unwrap();

        for (cell_n, cell_c) in surface_next.cells.iter().zip(mgr.surface_curr.cells.iter()) {
            if !matches!(cell_n.cell_type, crate::cell::CellType::Material(_)) { continue; }
            let dt_k = (cell_n.cell_temp - cell_c.cell_temp)
                .value_in(crate::uom::units::kelvin).abs();
            let drh = (cell_n.cell_rh - cell_c.cell_rh).value().abs();
            assert!(dt_k < 1e-5 && drh < 1e-5,
                    "ΔT={dt_k}, Δφ={drh} at supposed equilibrium");
        }
    }

    /// Property: a single call away from equilibrium produces a
    /// physically-plausible direction. With cooler BC than wall, every
    /// material cell's T_next ≤ T_curr (within iteration noise).
    /// Verifies the solve goes the right direction; sign-flipped
    /// off-diagonals would make T_next *increase* instead.
    #[test]
    fn one_iteration_moves_in_correct_direction() {
        use crate::boundary::celsius_to_kelvin;
        let mut mgr = fixture(6);
        // Wall at 25 °C / 50 %; air at 15 °C / 30 % (cooler & drier).
        let t_wall = celsius_to_kelvin(25.0);
        let phi_wall = 0.5 * crate::uom::units::dimensionless;
        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                    cell.cell_temp = t_wall;
                    cell.cell_rh = phi_wall;
                }
            }
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings).unwrap();
        }
        mgr.boundary_condition.temp_inside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air = celsius_to_kelvin(15.0);
        mgr.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(15.0);
        mgr.boundary_condition.rh_inside_air = 0.3 * crate::uom::units::dimensionless;
        mgr.boundary_condition.rh_outside_air = 0.3 * crate::uom::units::dimensionless;

        let mut surface_next = mgr.surface_curr.clone();
        surface_next.calculate_iteration_block_tridiag(
            &mgr.settings, &mgr.boundary_condition,
            &mgr.surface_prev, &mgr.surface_curr,
        ).unwrap();

        for cell in surface_next.cells.iter() {
            if !matches!(cell.cell_type, crate::cell::CellType::Material(_)) { continue; }
            let t_new = cell.cell_temp.value_in(crate::uom::units::kelvin);
            assert!(t_new <= t_wall.value_in(crate::uom::units::kelvin) + 1e-3,
                    "T_next {t_new} K > T_curr 298.15 K — wrong direction");
        }
    }

    /// Property: one orchestrator iteration must read BC and Virtual cell
    /// state from `self` (just refreshed via `apply_boundary_coefficients`
    /// and the virtual-cell update against the current-iterate material
    /// state) — NOT from `surface_curr` (where those cells still reflect
    /// the previous outer iteration). At convergence the two are equal,
    /// but mid-iteration the lag matters: if the assembly reads
    /// `surface_curr`, stale BC/Virtual coefficients leak into the linear
    /// system and convergence drops from near-Newton to slow BlockJacobi.
    ///
    /// Test: build two runs that differ ONLY in the Virtual(Interior)
    /// cell state of `surface_curr` (the Virtual cell sits between
    /// material and the Convection/Radiation BC cells; its `cell_temp`
    /// enters the resistance walk via the assembler's neighbour stencil
    /// over the interior-most material cell). With the fix, the
    /// orchestrator's virtual-cell update step rewrites the Virtual cell
    /// on `self` from current-iterate neighbours, so what was on
    /// `surface_curr` is irrelevant — the two runs must agree to
    /// floating-point noise. Without the fix, the assembler reads the
    /// poked Virtual cell from `surface_curr` and the runs disagree.
    #[test]
    fn iteration_uses_current_iter_state_not_stale_surface_curr() {
        use crate::boundary::celsius_to_kelvin;
        let mut mgr = fixture(6);
        mgr.boundary_condition.has_separate_radiant_int = true;
        // 5 K gap between wall and TMR / air, so a stale BC/Virtual state
        // would visibly disagree with a fresh one.
        let t_wall = celsius_to_kelvin(25.0);
        mgr.boundary_condition.temp_inside_mean_radiant = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_inside_air = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_outside_air = celsius_to_kelvin(20.0);
        mgr.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(20.0);

        for s in [&mut mgr.surface_curr, &mut mgr.surface_prev] {
            for cell in s.cells.iter_mut() {
                if matches!(cell.cell_type, crate::cell::CellType::Material(_)) {
                    cell.cell_temp = t_wall;
                }
            }
            s.apply_boundary_coefficients(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
            s.update_cell_properties(&mgr.boundary_condition, &mgr.settings)
                .unwrap();
        }

        // Baseline: surface_curr is consistent with its material-cell state.
        let mut next_fresh = mgr.surface_curr.clone();
        next_fresh
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &mgr.surface_curr,
            )
            .unwrap();

        // Stale variant: poke surface_curr's BC + Virtual interior cells
        // to deliberately-wrong values. The orchestrator must overwrite
        // them on `self` before assembly, so the result must be
        // unchanged.
        let mut surface_curr_stale = mgr.surface_curr.clone();
        let mut poked_rad = false;
        let mut poked_conv = false;
        let mut poked_virt = false;
        let bogus_t = celsius_to_kelvin(80.0);
        for cell in surface_curr_stale.cells.iter_mut() {
            match &cell.cell_type {
                crate::cell::CellType::Boundary(
                    crate::cell::Boundary::Radiation,
                    crate::cell::BoundaryLocation::Interior,
                ) => {
                    cell.htc = cell.htc * 10.0;
                    cell.cell_temp = bogus_t;
                    poked_rad = true;
                }
                crate::cell::CellType::Boundary(
                    crate::cell::Boundary::Convection,
                    crate::cell::BoundaryLocation::Interior,
                ) => {
                    cell.htc = cell.htc * 10.0;
                    cell.cell_temp = bogus_t;
                    poked_conv = true;
                }
                crate::cell::CellType::Virtual(
                    crate::cell::BoundaryLocation::Interior,
                ) => {
                    cell.cell_temp = bogus_t;
                    poked_virt = true;
                }
                _ => {}
            }
        }
        assert!(
            poked_rad && poked_conv && poked_virt,
            "fixture missing one of: interior Rad / Conv / Virtual cell"
        );

        let mut next_stale = mgr.surface_curr.clone();
        next_stale
            .calculate_iteration_block_tridiag(
                &mgr.settings,
                &mgr.boundary_condition,
                &mgr.surface_prev,
                &surface_curr_stale,
            )
            .unwrap();

        for (cf, cs) in next_fresh.cells.iter().zip(next_stale.cells.iter()) {
            if !matches!(cf.cell_type, crate::cell::CellType::Material(_)) { continue; }
            let dt = (cf.cell_temp - cs.cell_temp)
                .value_in(crate::uom::units::kelvin)
                .abs();
            let drh = (cf.cell_rh - cs.cell_rh).value().abs();
            assert!(
                dt < 1e-9 && drh < 1e-9,
                "stale surface_curr leaked into orchestrator result: ΔT={dt} K, Δφ={drh}"
            );
        }
    }
}
