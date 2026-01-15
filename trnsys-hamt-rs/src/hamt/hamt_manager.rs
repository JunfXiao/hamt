//! The main orchestrator for the HAMT simulation.
//! TODO: Refactor to Arc-backed registries and indexed cells/surfaces.

use tracing::warn;

use super::boundary::BoundaryCondition;
use crate::hamt::cell::{SubCell, ADJACENT_CELLS_MAX};
use crate::hamt::material::{Construction, Material};
use crate::hamt::physics;
use crate::trnsys::error::TrnSysError;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages the state and execution of the HAMT simulation for all surfaces.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// A flat vector containing all cells for all simulated surfaces.
    pub cells: Vec<SubCell>,

    /// Materials registry; cells hold Arc references directly.
    pub materials: Vec<Arc<Material>>,
    /// Constructions registry; surfaces hold Arc references directly.
    pub constructions: Vec<Arc<Construction>>,

    /// State and index information for each individual surface being simulated.
    pub surface_states: Vec<SurfaceState>,

    /// Boundary conditions for the simulation.
    pub boundary_condition: BoundaryCondition,

    /// Simulation-wide settings
    pub settings: HamtSettings,
}

/// Holds the indices and reporting variables for a single surface.
#[derive(Debug, Clone, Default)]
pub struct SurfaceState {
    pub name: String,
    pub construction_name: String,
    /// Construction reference, if known.
    pub construction: Option<Arc<Construction>>,
    pub area: f64,

    // Indices mapping into the main `cells` vector
    pub first_cell: usize,
    pub last_cell: usize,
    pub ext_cell: usize,     // Exterior virtual cell
    pub int_cell: usize,     // Interior virtual cell
    pub ext_con_cell: usize, // Exterior convection cell
    pub int_con_cell: usize, // Interior convection cell
    pub ext_rad_cell: usize,
    pub ext_sky_cell: usize,
    pub ext_grn_cell: usize,
    pub user_def_cell: usize, // Cell at user-defined depth

    // Boundary conditions / Flags
    pub ext_vtc_flag: bool,
    pub int_vtc_flag: bool,
    pub ext_vtc: f64,
    pub int_vtc: f64,

    // Reporting Variables (Outputs)
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

/// Global settings for the HAMT simulation.
#[derive(Debug, Clone)]
pub struct HamtSettings {
    pub timestep: f64,
    pub max_iterations: i32,
    pub temp_convergence_tolerance: f64,
    pub rh_convergence_tolerance: f64,
    pub latent_heat_enabled: bool,
}

impl HamtManager {
    /// Creates a new `HamtManager` and initializes all surfaces.
    pub fn new(
        settings: HamtSettings,
        materials: HashMap<String, Material>,
        constructions: HashMap<String, Construction>,
        surfaces_to_init: &[(String, String, f64)], // Tuples of (surface_name, construction_name, area)
        boundary_condition: BoundaryCondition,
    ) -> Result<Self, TrnSysError> {
        let mut materials_vec = Vec::new();
        let mut material_map = HashMap::new();
        for (name, mat) in materials.into_iter() {
            let arc = Arc::new(mat);
            materials_vec.push(arc.clone());
            if let Some(_) = material_map.insert(name, arc) {
                warn!(
                    "Duplicated Material {} appeared more than once, it is replaced.",
                    name
                );
            }
        }

        let mut constructions_vec = Vec::new();
        let mut construction_map = HashMap::new();
        for (name, con) in constructions.into_iter() {
            let arc = Arc::new(con);
            constructions_vec.push(arc.clone());
            construction_map.insert(name, arc);
        }

        let mut cells = Vec::new();
        let mut surface_states = Vec::new();

        for (name, construction_name, area) in surfaces_to_init.iter() {
            let con_arc = construction_map
                .get(construction_name)
                .cloned()
                .ok_or_else(|| {
                    TrnSysError::HamtError(format!(
                        "Construction '{}' not found for surface '{}'",
                        construction_name, name
                    ))
                })?;
            // Minimal single-cell allocation per surface to keep indexing valid.
            let base = cells.len();
            let mut cell = SubCell::default();
            cell.s_id = surface_states.len() as i32;
            cell.material = con_arc
                .layer_names
                .get(0)
                .and_then(|mat_name| material_map.get(mat_name).cloned());
            cells.push(cell);

            let surf = SurfaceState {
                name: name.clone(),
                construction_name: construction_name.clone(),
                construction: Some(con_arc.clone()),
                area: *area,
                first_cell: base,
                last_cell: base,
                ext_cell: base,
                int_cell: base,
                ext_con_cell: base,
                int_con_cell: base,
                ext_rad_cell: base,
                ext_sky_cell: base,
                ext_grn_cell: base,
                user_def_cell: base,
                ..SurfaceState::default()
            };
            // flags/defaults already set by Default
            surface_states.push(surf);
        }

        Ok(Self {
            cells,
            materials: materials_vec,
            constructions: constructions_vec,
            surface_states,
            settings,
            boundary_condition,
        })
    }

    /// Executes the iterative solver for one time step.
    /// This is a Rust translation of the `CalcHeatBalHAMT` subroutine.
    pub fn calculate_timestep(&mut self) -> Result<(), TrnSysError> {
        for s_idx in 0..self.surface_states.len() {
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
                let surface = &self.surface_states[s_idx];
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
                    if let Some(mat) = self.cells[cid].material.as_ref() {
                        // Placeholder for property update using material curves.
                        // self.cells[cid].w_thermal_c = physics::interp(&mat.thermal_conductivity, self.cells[cid].water).0;
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
            let surf_state = &mut self.surface_states[s_idx];
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
