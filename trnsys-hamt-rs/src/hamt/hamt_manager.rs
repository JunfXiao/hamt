//! The main orchestrator for the HAMT simulation.
//! TODO: Refactor to Arc-backed registries and indexed cells/surfaces.

use tracing::warn;

use super::boundary::BoundaryCondition;
use super::surface::Surface;
use crate::hamt::cell::{Cell, ADJACENT_CELLS_MAX};
use crate::hamt::material::{Construction, Material};
use crate::hamt::physics;
use crate::trnsys::error::TrnSysError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Manages the state and execution of the HAMT simulation for all surfaces.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// A flat vector containing all cells for all simulated surfaces.
    pub cells: Vec<Cell>,

    /// Materials registry; cells hold Arc references directly.
    pub materials: Vec<Arc<Material>>,
    /// Constructions registry; surfaces hold Arc references directly.
    pub constructions: Vec<Arc<Construction>>,

    pub surface: Surface,

    /// Boundary conditions for the simulation.
    pub boundary_condition: BoundaryCondition,

    /// Simulation-wide settings
    pub settings: HamtSettings,
}

/// Global settings for the HAMT simulation.
#[derive(Debug, Clone)]
pub struct HamtSettings {
    /// Timestep [s].
    pub timestep: f64,
    /// Maximum nonlinear iterations (count).
    pub max_iterations: i32,
    /// Temperature convergence tolerance [K or °C].
    pub temp_convergence_tolerance: f64,
    /// Relative humidity convergence tolerance [-].
    pub rh_convergence_tolerance: f64,
    /// Flag to enable latent heat coupling (dimensionless).
    pub latent_heat_enabled: bool,
}

impl HamtManager {
    /// Creates a new `HamtManager` and initializes all surfaces.
    pub fn new(
        settings: HamtSettings,
        materials: HashMap<String, Material>,
        layer_names: Vec<String>,
        construction_name: String,
        surface_name: String,
        surface_area: f64,
    ) -> Result<Self, TrnSysError> {
        let mut cells: Vec<Cell> = Vec::new();
        let mut layers: Vec<Arc<Material>> = Vec::with_capacity(layer_names.len());

        let mut material_map = HashMap::new();
        for mat_name in layer_names.into_iter() {
            let mat_arc = material_map.get(&mat_name).cloned().ok_or_else(|| {
                TrnSysError::HamtError(format!(
                    "Material '{}' not found for surface '{}'",
                    mat_name, surface_name
                ))
            })?;
            layers.push(mat_arc);
        }

        let con_arc: Construction = Construction {
            name: construction_name,
            layers: layers,
        };
        let base = cells.len();
        let mut prev_index: Option<usize> = None;
        for (layer_idx, mat_arc) in con_arc.layers.iter().enumerate() {
            let thickness = mat_arc.thickness.max(1.0e-6);
            let mut cell = Cell {
                material: Some(mat_arc.clone()),
                length: [thickness, 0.0],
                volume: *area * thickness,
                overlap: [*area, *area],
                dist: [thickness / 2.0, thickness / 2.0],
                adjs: [0, 0],
                adjsl: [0, 0],
            };

            let idx = cells.len();
            if let Some(prev) = prev_index {
                let prev_half = cells[prev].length[0] / 2.0;
                let mid_dist = prev_half + thickness / 2.0;
                cells[prev].adjs[1] = idx as i32;
                cells[prev].adjsl[1] = 0;
                cells[idx].adjs[0] = prev as i32;
                cells[idx].adjsl[0] = 1;
                cells[prev].dist[1] = mid_dist;
                cells[idx].dist[0] = mid_dist;
                cells[idx].overlap[0] = *area;
            }
            cells.push(cell);
            prev_index = Some(idx);
        }

        // Skip surfaces with no valid layers
        let Some(last_cell) = prev_index else {
            continue;
        };
        let first_cell = base;

        let surf = Surface {
            name: name.clone(),
            construction_name: construction_name.clone(),
            construction: Some(con_arc.clone()),
            area: *area,
            first_cell,
            last_cell,
            ext_cell: first_cell,
            int_cell: last_cell,
            ext_con_cell: first_cell,
            int_con_cell: last_cell,
            ext_rad_cell: first_cell,
            ext_sky_cell: first_cell,
            ext_grn_cell: first_cell,
            user_def_cell: first_cell,
            ..Surface::default()
        };
        surface_states.push(surf);

        Ok(Self {
            cells,
            materials: materials_vec,
            constructions: constructions_vec,
            surface: surface_states,
            settings,
            boundary_condition,
        })
    }

    /// Executes the iterative solver for one time step.
    /// This is a Rust translation of the `CalcHeatBalHAMT` subroutine.
    pub fn calculate_timestep(&mut self) -> Result<(), TrnSysError> {
        for s_idx in 0..self.surface.len() {
            let (
                first_cell,
                last_cell,
                ext_cell,
                int_cell,
                user_def_cell,
                ext_con_cell,
                int_con_cell,
                surface_name,
            ) = {
                let surface = &self.surface[s_idx];
                (
                    surface.first_cell,
                    surface.last_cell,
                    surface.ext_cell,
                    surface.int_cell,
                    surface.user_def_cell,
                    surface.ext_con_cell,
                    surface.int_con_cell,
                    surface.name.clone(),
                )
            };

            // --- Apply Boundary Conditions ---
            // This section needs to be populated from TRNSYS inputs for each time step.
            let temp_outside_air = self.boundary_condition.temp_outside_air;
            let rh_outside_air = self.boundary_condition.rh_outside_air;
            let temp_inside_air = self.boundary_condition.temp_inside_air;
            let rh_inside_air = self.boundary_condition.rh_inside_air;
            let _pressure = self.boundary_condition.pressure_air;

            let rho_vapor_out = physics::psy_rhov_fn_tdb_rh(temp_outside_air, rh_outside_air);
            let rho_vapor_in = physics::psy_rhov_fn_tdb_rh(temp_inside_air, rh_inside_air);

            self.cells[ext_con_cell].temp = temp_outside_air;
            self.cells[ext_con_cell].rh =
                physics::psy_rh_fn_tdb_rhov(temp_outside_air, rho_vapor_out);
            self.cells[int_con_cell].temp = temp_inside_air;
            self.cells[int_con_cell].rh =
                physics::psy_rh_fn_tdb_rhov(temp_inside_air, rho_vapor_in);
            // ... apply other boundary conditions (radiation, heat transfer coeffs, etc.)

            // --- Iteration Loop ---
            for itter in 0..=self.settings.max_iterations {
                // Update moisture-dependent properties for all material cells
                for cid in first_cell..=last_cell {
                    if let Some(mat_arc) = self.cells[cid].material.clone() {
                        // Update base thermo properties
                        self.cells[cid].density = mat_arc.density;
                        self.cells[cid].spech = mat_arc.spec_heat;

                        // Sorption isotherm: water content vs RH
                        let (theta, dtheta_drh) =
                            physics::interp(&mat_arc.sorption_isotherm, self.cells[cid].rh_p1);
                        self.cells[cid].water = theta.max(0.0);
                        self.cells[cid].dwd_phi = dtheta_drh;

                        // Thermal conductivity vs water content (fallback to dry conductivity)
                        let (lambda, _) =
                            physics::interp(&mat_arc.thermal_conductivity, self.cells[cid].water);
                        self.cells[cid].w_thermal_c = if lambda > 0.0 {
                            lambda
                        } else {
                            mat_arc.conductivity
                        };

                        // Vapor diffusion factor vs RH
                        let (mu, _) =
                            physics::interp(&mat_arc.vapor_diffusion, self.cells[cid].rh_p1);
                        if mu > 0.0 {
                            self.cells[cid].mu = mu;
                        }

                        // Liquid transport coefficient vs water content
                        let (dw, _) = physics::interp(&mat_arc.suction, self.cells[cid].water);
                        if dw > 0.0 {
                            self.cells[cid].dw = dw;
                        }
                    }
                    self.cells[cid].vp =
                        physics::rh_to_vp(self.cells[cid].rh, self.cells[cid].temp);
                    self.cells[cid].vp_p1 =
                        physics::rh_to_vp(self.cells[cid].rh_p1, self.cells[cid].temp_p1);
                    self.cells[cid].vp_sat = physics::psy_psat_fn_temp(self.cells[cid].temp_p1);
                }

                // Solve Heat & Moisture Balance for each material cell
                for cid in ext_cell..=int_cell {
                    let mut torsum = 0.0;
                    let mut oorsum = 0.0;

                    // Heat Balance Terms
                    for i in 0..ADJACENT_CELLS_MAX {
                        let adj_cid = self.cells[cid].adjs[i];
                        if adj_cid == -1 {
                            break;
                        }
                        let adj_cid = adj_cid as usize;

                        // Skip degenerate faces
                        if self.cells[cid].overlap[i] <= 0.0 {
                            continue;
                        }

                        // Simplified resistance calculation
                        let therm_r1 = if self.cells[cid].htc > 0.0 {
                            1.0 / (self.cells[cid].overlap[i] * self.cells[cid].htc)
                        } else if self.cells[cid].w_thermal_c > 0.0 {
                            self.cells[cid].dist[i]
                                / (self.cells[cid].overlap[i] * self.cells[cid].w_thermal_c)
                        } else {
                            continue;
                        };
                        let therm_r2 = if self.cells[adj_cid].htc > 0.0 {
                            1.0 / (self.cells[cid].overlap[i] * self.cells[adj_cid].htc)
                        } else if self.cells[adj_cid].w_thermal_c > 0.0 {
                            self.cells[adj_cid].dist[self.cells[cid].adjsl[i] as usize]
                                / (self.cells[cid].overlap[i] * self.cells[adj_cid].w_thermal_c)
                        } else {
                            continue;
                        };

                        if therm_r1 + therm_r2 > 0.0 {
                            oorsum += 1.0 / (therm_r1 + therm_r2);
                            torsum += self.cells[adj_cid].temp_p1 / (therm_r1 + therm_r2);
                        }
                    }

                    let tcap = (self.cells[cid].density * self.cells[cid].spech
                        + self.cells[cid].water * physics::W_SPECH)
                        * self.cells[cid].volume;
                    let qvp = 0.0; // Latent heat calculation placeholder
                    let denom = oorsum + (tcap / self.settings.timestep);
                    if denom > 0.0 {
                        self.cells[cid].temp_p1 = (torsum
                            + qvp
                            + self.cells[cid].q_adds
                            + (tcap * self.cells[cid].temp / self.settings.timestep))
                            / denom;
                    }

                    // Moisture Balance (placeholder)
                    self.cells[cid].rh_p1 = self.cells[cid].rh_p2; // simplified
                }

                // --- Convergence Check ---
                let mut max_temp_change: f64 = 0.0;
                let mut max_rh_change: f64 = 0.0;
                for cid in ext_cell..=int_cell {
                    max_temp_change = max_temp_change
                        .max((self.cells[cid].temp_p1 - self.cells[cid].temp_p2).abs());
                    max_rh_change =
                        max_rh_change.max((self.cells[cid].rh_p1 - self.cells[cid].rh_p2).abs());
                }

                if itter > 0
                    && max_temp_change < self.settings.temp_convergence_tolerance
                    && max_rh_change < self.settings.rh_convergence_tolerance
                {
                    break; // Converged
                }

                if itter == self.settings.max_iterations {
                    // Handle non-convergence
                    return Err(TrnSysError::HamtError(format!(
                        "Surface {} failed to converge.",
                        surface_name
                    )));
                }

                // Prepare for next iteration
                for cid in first_cell..=last_cell {
                    self.cells[cid].temp_p2 = self.cells[cid].temp_p1;
                    self.cells[cid].rh_p2 = self.cells[cid].rh_p1;
                }
            }

            let water_total: f64 = (first_cell..=last_cell)
                .map(|cid| self.cells[cid].water * self.cells[cid].volume)
                .sum();
            let surf_state = &mut self.surface[s_idx];
            surf_state.temp_surf_out = self.cells[ext_cell].temp_p1;
            surf_state.temp_surf_in = self.cells[int_cell].temp_p1;
            surf_state.rh_surf_out = self.cells[ext_cell].rh_p1;
            surf_state.rh_surf_in = self.cells[int_cell].rh_p1;
            surf_state.temp_surf_user_def = self.cells[user_def_cell].temp_p1;
            surf_state.rh_surf_user_def = self.cells[user_def_cell].rh_p1;
            surf_state.water_surf_user_def = self.cells[user_def_cell].water;
            surf_state.q_surf_ext = 0.0;
            surf_state.q_surf_int = 0.0;
            surf_state.m_surf_ext = 0.0;
            surf_state.m_surf_int = 0.0;
            surf_state.water_total = water_total;
        }
        Ok(())
    }

    /// Placeholder for the end-of-timestep update.
    pub fn update_history(&mut self) {
        for cell in self.cells.iter_mut() {
            cell.temp = cell.temp_p1;
            cell.rh = cell.rh_p1;
        }
    }
}

impl Default for HamtSettings {
    fn default() -> Self {
        Self {
            timestep: 3600.0,
            max_iterations: 150,
            temp_convergence_tolerance: 0.0001,
            rh_convergence_tolerance: 0.0001,
            latent_heat_enabled: true,
        }
    }
}
