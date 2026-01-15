//! The main orchestrator for the HAMT simulation.

use tracing::warn;

use super::boundary::BoundaryCondition;
use super::surface::Surface;
use crate::hamt::cell::{Cell, ADJACENT_CELLS_MAX};
use crate::hamt::material::{Construction, Material};
use crate::hamt::physics;
use crate::trnsys::error::TrnSysError;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

/// Manages the state and execution of the HAMT simulation for a surface.
#[derive(Debug, Clone)]
pub struct HamtManager {
    /// A flat vector containing all cells for the simulated surface.
    pub cells: Vec<Cell>,

    /// Materials registry for this manager.
    pub materials: Vec<Arc<Material>>,
    /// Construction for the surface.
    pub construction: Arc<Construction>,

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
    /// Flag to indicate if rain is present on the exterior surface.
    pub rain_enabled: bool,
}

const ADJ_DIST: f64 = 0.00005; // Allowable distance between two cells
const DIV_SIZE: f64 = 0.05; // Simplified default division size [m]
const DIV_MIN: i32 = 1; // Simplified minimum divisions

impl HamtManager {
    /// Creates a new `HamtManager` and initializes the cell structure for a single surface.
    /// This is a Rust translation of the `InitHeatBalHAMT` subroutine logic.
    pub fn new(
        settings: HamtSettings,
        construction: Arc<Construction>,
        surface_name: String,
        surface_area: f64,
    ) -> Result<Self, TrnSysError> {
        // --- 1. Assemble Construction ---

        // --- 2. Create and Initialize Cells ---
        let mut cells: Vec<Cell> = Vec::new();
        let mut runor = -0.02; // Running origin for cell placement

        let first_cell = cells.len();

        // --- Exterior Boundary Cells (5) ---
        let ext_con_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            cells.len() - 1
        };
        let ext_rad_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            cells.len() - 1
        };
        let ext_sky_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            cells.len() - 1
        };
        let ext_grn_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            cells.len() - 1
        };
        runor += 0.01;

        // --- Exterior Virtual Surface Cell ---
        let ext_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            runor += cells.last().unwrap().length[0];
            cells.len() - 1
        };

        // --- Material Cells ---
        for mat_arc in construction.layers.iter() {
            // Simplified division calculation
            let mut divs = ((mat_arc.thickness / DIV_SIZE).ceil() as i32).max(DIV_MIN);

            // Ensure cell length is not too small
            let sin_neg_pi_ovr_2 = (PI * -0.5).sin();
            loop {
                let test_len = mat_arc.thickness
                    * (((PI * (-1.0 / divs as f64) - PI / 2.0).sin() / 2.0)
                        - (sin_neg_pi_ovr_2 / 2.0));
                if test_len > ADJ_DIST || divs <= 1 {
                    break;
                }
                divs -= 1;
            }

            for did in 1..=divs {
                let did_f64 = did as f64;
                let divs_f64 = divs as f64;
                // Make cells smaller near the surface using sine distribution
                let length = mat_arc.thickness
                    * (((PI * (-did_f64 / divs_f64) - PI / 2.0).sin() / 2.0)
                        - ((PI * (-(did_f64 - 1.0) / divs_f64) - PI / 2.0).sin() / 2.0));

                let origin = runor + length / 2.0;
                runor += length;
                let volume = length * surface_area;

                let mut cell =
                    Cell::new_material(mat_arc.clone(), length, origin, volume, surface_area);

                // Initial conditions from material
                cell.temp = mat_arc.initial_temp;
                cell.temp_p1 = mat_arc.initial_temp;
                cell.temp_p2 = mat_arc.initial_temp;
                cell.rh = mat_arc.initial_rh;
                cell.rh_p1 = mat_arc.initial_rh;
                cell.rh_p2 = mat_arc.initial_rh;

                cells.push(cell);
            }
        }

        // --- Interior Virtual Surface Cell ---
        let int_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            runor += cells.last().unwrap().length[0];
            cells.len() - 1
        };

        // --- Interior Boundary Cell ---
        let int_con_cell = {
            cells.push(Cell::new_boundary(runor, 0.01));
            cells.len() - 1
        };

        let last_cell = cells.len() - 1;

        // Finalize origins for virtual surface cells
        cells[ext_cell].origin[0] += cells[ext_cell].length[0] / 2.0;
        cells[int_cell].origin[0] -= cells[int_cell].length[0] / 2.0;
        cells[ext_cell].volume = 0.0;
        cells[int_cell].volume = 0.0;

        // --- 3. Find and Link Adjacent Cells ---
        for cid1 in 0..cells.len() {
            for cid2 in 0..cells.len() {
                if cid1 == cid2 {
                    continue;
                }

                let (cell1_origin, cell1_len) = (cells[cid1].origin[0], cells[cid1].length[0]);
                let (cell2_origin, cell2_len) = (cells[cid2].origin[0], cells[cid2].length[0]);

                let high1 = cell1_origin + cell1_len / 2.0;
                let low2 = cell2_origin - cell2_len / 2.0;

                if (low2 - high1).abs() < ADJ_DIST {
                    let adj1 = cells[cid1]
                        .adjs
                        .iter()
                        .position(|&a| a == -1)
                        .unwrap_or(ADJACENT_CELLS_MAX);
                    let adj2 = cells[cid2]
                        .adjs
                        .iter()
                        .position(|&a| a == -1)
                        .unwrap_or(ADJACENT_CELLS_MAX);

                    if adj1 < ADJACENT_CELLS_MAX && adj2 < ADJACENT_CELLS_MAX {
                        cells[cid1].adjs[adj1] = cid2 as i32;
                        cells[cid2].adjs[adj2] = cid1 as i32;
                        cells[cid1].adjsl[adj1] = adj2 as i32;
                        cells[cid2].adjsl[adj2] = adj1 as i32;

                        cells[cid1].overlap[adj1] = surface_area;
                        cells[cid2].overlap[adj2] = surface_area;

                        cells[cid1].dist[adj1] = cell1_len / 2.0;
                        cells[cid2].dist[adj2] = cell2_len / 2.0;
                    }
                }
            }
        }

        let surface = Surface {
            name: surface_name,
            construction_name: construction.name.clone(),
            area: surface_area,
            first_cell,
            last_cell,
            ext_cell,
            int_cell,
            ext_con_cell,
            int_con_cell,
            ext_rad_cell,
            ext_sky_cell,
            ext_grn_cell,
            user_def_cell: ext_cell, // Default to exterior surface
            ..Surface::default()
        };

        Ok(Self {
            cells,
            materials: materials_vec,
            construction,
            surface,
            settings,
            boundary_condition: BoundaryCondition::default(),
        })
    }

    /// Executes the iterative solver for one time step.
    /// This is a Rust translation of the `CalcHeatBalHAMT` subroutine.
    pub fn calculate_timestep(&mut self) -> Result<(), TrnSysError> {
        let surface = &self.surface;
        let (first_cell, last_cell, ext_cell, int_cell) = (
            surface.first_cell,
            surface.last_cell,
            surface.ext_cell,
            surface.int_cell,
        );

        // --- Apply Boundary Conditions ---
        let bc = &self.boundary_condition;
        self.cells[surface.ext_con_cell].temp = bc.temp_outside_air;
        let rho_vapor_out = physics::psy_rhov_fn_tdb_rh(bc.temp_outside_air, bc.rh_outside_air);
        self.cells[surface.ext_con_cell].rh =
            physics::psy_rh_fn_tdb_rhov(bc.temp_outside_air, rho_vapor_out).min(physics::RH_MAX);

        self.cells[surface.int_con_cell].temp = bc.temp_inside_air;
        let rho_vapor_in = physics::psy_rhov_fn_tdb_rh(bc.temp_inside_air, bc.rh_inside_air);
        self.cells[surface.int_con_cell].rh =
            physics::psy_rh_fn_tdb_rhov(bc.temp_inside_air, rho_vapor_in).min(physics::RH_MAX);

        // --- Iteration Loop ---
        for itter in 0..=self.settings.max_iterations {
            // 1. Update moisture-dependent properties for all cells
            for cid in first_cell..=last_cell {
                self.cells[cid].vp = physics::rh_to_vp(self.cells[cid].rh, self.cells[cid].temp);
                self.cells[cid].vp_p1 =
                    physics::rh_to_vp(self.cells[cid].rh_p1, self.cells[cid].temp_p1);
                self.cells[cid].vp_sat = physics::psy_psat_fn_temp(self.cells[cid].temp_p1);

                if let Some(mat_arc) = self.cells[cid].material.clone() {
                    self.cells[cid].density = mat_arc.density;
                    self.cells[cid].spech = mat_arc.spec_heat;

                    let (theta, dtheta_drh) =
                        physics::interp(&mat_arc.sorption_isotherm, self.cells[cid].rh_p1);
                    self.cells[cid].water = theta.max(0.0);
                    self.cells[cid].dwd_phi = dtheta_drh;

                    let (lambda, _) =
                        physics::interp(&mat_arc.thermal_conductivity, self.cells[cid].water);
                    self.cells[cid].w_thermal_c = if lambda > 0.0 {
                        lambda
                    } else {
                        mat_arc.conductivity
                    };

                    let (mu, _) = physics::interp(&mat_arc.vapor_diffusion, self.cells[cid].rh_p1);
                    if mu > 0.0 {
                        self.cells[cid].mu = mu;
                    }

                    let (dw, _) = if self.settings.rain_enabled {
                        physics::interp(&mat_arc.suction, self.cells[cid].water)
                    } else {
                        physics::interp(&mat_arc.redistribution, self.cells[cid].water)
                    };
                    if dw > 0.0 {
                        self.cells[cid].dw = dw;
                    }
                }
            }

            // 2. Solve Heat Balance for each material cell
            for cid in ext_cell..=int_cell {
                let mut torsum = 0.0;
                let mut oorsum = 0.0;
                let mut vpdiff = 0.0;

                for i in 0..ADJACENT_CELLS_MAX {
                    let adj_cid = self.cells[cid].adjs[i];
                    if adj_cid == -1 {
                        break;
                    }
                    let adj_cid = adj_cid as usize;
                    let adj_sl = self.cells[cid].adjsl[i] as usize;

                    if self.cells[cid].overlap[i] <= 0.0 {
                        continue;
                    }

                    let (therm_r1, vapor_r1) = self.calculate_resistances(cid, i, bc.pressure_air);
                    let (therm_r2, vapor_r2) =
                        self.calculate_resistances(adj_cid, adj_sl, bc.pressure_air);

                    if therm_r1 + therm_r2 > 0.0 {
                        oorsum += 1.0 / (therm_r1 + therm_r2);
                        torsum += self.cells[adj_cid].temp_p1 / (therm_r1 + therm_r2);
                    }
                    if vapor_r1 + vapor_r2 > 0.0 {
                        vpdiff +=
                            (self.cells[adj_cid].vp - self.cells[cid].vp) / (vapor_r1 + vapor_r2);
                    }
                }

                let tcap = (self.cells[cid].density * self.cells[cid].spech
                    + self.cells[cid].water * physics::W_SPECH)
                    * self.cells[cid].volume;

                let mut qvp = 0.0;
                if self.settings.latent_heat_enabled && self.cells[cid].material.is_some() {
                    qvp = vpdiff * physics::W_HV;
                    if qvp.abs() > physics::QVP_LIMIT {
                        warn!("Large latent heat detected for surface {}", surface.name);
                        qvp = 0.0;
                    }
                }

                let denom = oorsum + (tcap / self.settings.timestep);
                if denom > 0.0 {
                    self.cells[cid].temp_p1 = (torsum
                        + qvp
                        + self.cells[cid].q_adds
                        + (tcap * self.cells[cid].temp / self.settings.timestep))
                        / denom;
                }
            }

            // 3. Solve Moisture Balance for each material cell
            for cid in ext_cell..=int_cell {
                let mut phiorsum = 0.0;
                let mut phioosum = 0.0;
                let mut vporsum = 0.0;
                let mut vpoosum = 0.0;

                for i in 0..ADJACENT_CELLS_MAX {
                    let adj_cid = self.cells[cid].adjs[i];
                    if adj_cid == -1 {
                        break;
                    }
                    let adj_cid = adj_cid as usize;
                    let adj_sl = self.cells[cid].adjsl[i] as usize;
                    let overlap = self.cells[cid].overlap[i];

                    if overlap <= 0.0 {
                        continue;
                    }

                    let (_, vapor_r1) = self.calculate_resistances(cid, i, bc.pressure_air);
                    let (_, vapor_r2) =
                        self.calculate_resistances(adj_cid, adj_sl, bc.pressure_air);

                    if vapor_r1 + vapor_r2 > 0.0 {
                        vpoosum += 1.0 / (vapor_r1 + vapor_r2);
                        vporsum += self.cells[adj_cid].vp_p1 / (vapor_r1 + vapor_r2);
                    }

                    let rhr1 = if self.cells[cid].dw > 0.0 && self.cells[cid].dwd_phi > 0.0 {
                        self.cells[cid].dist[i]
                            / (overlap * self.cells[cid].dw * self.cells[cid].dwd_phi)
                    } else {
                        0.0
                    };

                    let rhr2 = if self.cells[adj_cid].dw > 0.0 && self.cells[adj_cid].dwd_phi > 0.0
                    {
                        self.cells[adj_cid].dist[adj_sl]
                            / (overlap * self.cells[adj_cid].dw * self.cells[adj_cid].dwd_phi)
                    } else {
                        0.0
                    };

                    if rhr1 * rhr2 > 0.0 {
                        phioosum += 1.0 / (rhr1 + rhr2);
                        phiorsum += self.cells[adj_cid].rh_p1 / (rhr1 + rhr2);
                    }
                }

                let wcap = if self.cells[cid].dwd_phi > 0.0 {
                    self.cells[cid].dwd_phi * self.cells[cid].volume
                } else {
                    0.0
                };

                let denom =
                    phioosum + vpoosum * self.cells[cid].vp_sat + wcap / self.settings.timestep;
                if denom != 0.0 {
                    self.cells[cid].rh_p1 =
                        (phiorsum + vporsum + (wcap * self.cells[cid].rh) / self.settings.timestep)
                            / denom;
                } else {
                    return Err(TrnSysError::HamtError(format!(
                        "Moisture balance denominator is zero for cell {} on surface {}",
                        cid, surface.name
                    )));
                }

                if self.cells[cid].rh_p1 > physics::RH_MAX {
                    self.cells[cid].rh_p1 = physics::RH_MAX;
                }
            }

            // --- Convergence Check ---
            let (max_temp_change, max_rh_change) = self.check_convergence(ext_cell, int_cell);

            if itter > 0
                && max_temp_change < self.settings.temp_convergence_tolerance
                && max_rh_change < self.settings.rh_convergence_tolerance
            {
                break; // Converged
            }

            if itter == self.settings.max_iterations {
                return Err(TrnSysError::HamtError(format!(
                    "Surface {} failed to converge. Temp change: {:.4}, RH change: {:.4}",
                    surface.name, max_temp_change, max_rh_change
                )));
            }

            // Prepare for next iteration
            for cid in first_cell..=last_cell {
                self.cells[cid].temp_p2 = self.cells[cid].temp_p1;
                self.cells[cid].rh_p2 = self.cells[cid].rh_p1;
            }
        }

        // --- Finalize Timestep ---
        let surf_state = &mut self.surface;
        surf_state.temp_surf_out = self.cells[ext_cell].temp_p1;
        surf_state.temp_surf_in = self.cells[int_cell].temp_p1;
        surf_state.rh_surf_out = self.cells[ext_cell].rh_p1;
        surf_state.rh_surf_in = self.cells[int_cell].rh_p1;
        surf_state.water_total = (first_cell..=last_cell)
            .map(|cid| self.cells[cid].water * self.cells[cid].volume)
            .sum();

        Ok(())
    }

    /// Calculates thermal and vapor resistances for a cell's face.
    fn calculate_resistances(&self, cid: usize, face_idx: usize, pressure: f64) -> (f64, f64) {
        let cell = &self.cells[cid];
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
            let wvdc = physics::wvdc(cell.temp_p1, pressure);
            (dist * cell.mu) / (overlap * wvdc)
        } else {
            0.0
        };

        (therm_r, vapor_r)
    }

    /// Checks the maximum change in temperature and RH across cells for convergence.
    fn check_convergence(&self, start_cid: usize, end_cid: usize) -> (f64, f64) {
        let mut max_temp_change = 0.0;
        let mut max_rh_change = 0.0;
        for cid in start_cid..=end_cid {
            max_temp_change =
                max_temp_change.max((self.cells[cid].temp_p1 - self.cells[cid].temp_p2).abs());
            max_rh_change =
                max_rh_change.max((self.cells[cid].rh_p1 - self.cells[cid].rh_p2).abs());
        }
        (max_temp_change, max_rh_change)
    }

    /// Updates the `temp` and `rh` fields from the converged `temp_p1` and `rh_p1` values.
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
            rain_enabled: false,
        }
    }
}
