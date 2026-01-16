use crate::hamt::boundary::BoundaryCondition;
use crate::hamt::hamt_manager::{HamtManager, HamtSettings};
use crate::hamt::io::load_hamt_data;
use crate::hamt::material::{Construction, Material};
use crate::trnsys::error::{InputError, TrnSysError};
use crate::trnsys::iteration_mode::IterationMode;
use crate::trnsys::param::TrnSysValue;
use crate::trnsys::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::warn;

pub(crate) struct TrnSysType {}

impl TrnSysType {
    /// Create a new instance.
    pub fn new() -> Self {
        TrnSysType {}
    }

    /// Very first call of the simulation: declare I/O sizes and iteration mode.
    pub fn first_call_of_simulation(&self, state: &mut TrnSysState) -> Result<()> {
        state.num_params = 8;
        state.num_inputs = 11;
        state.num_outputs = 11;
        state.num_derivatives = 0;
        state.iteration_mode = IterationMode::DynamicMode;
        Ok(())
    }

    /// Validate parameters.
    pub fn validate_parameters(&self, state: &mut TrnSysState) -> Result<()> {
        if state.params.len() != 8 {
            return Err(InputError::BadParameter {
                index: 1,
                message: "8 parameters are required.".to_string(),
            }
            .into());
        }
        let surface_area: f64 = (&state.params[1]).into();
        if surface_area <= 0.0 {
            return Err(InputError::BadParameter {
                index: 2,
                message: "Surface area must be positive".to_string(),
            }
            .into());
        }

        let logical_unit: i32 =
            (&state.params[2])
                .try_into()
                .map_err(|_| InputError::BadParameter {
                    index: 4,
                    message: "Parameter 4 must be an integer logical unit.".to_string(),
                })?;
        let file = get_lu_filename(logical_unit);
        let file_path = Path::new(&file);
        if !file_path.exists() {
            return Err(InputError::BadParameter {
                index: 4,
                message: format!(
                    "File for logical unit {} does not exist: {}",
                    logical_unit,
                    file_path.display()
                ),
            }
            .into());
        }
        // create library
        let mat_library = load_hamt_data(file_path)?;
        let construction = mat_library
            .get_construction_read_lock()?
            .values()
            .next()
            .cloned()
            .ok_or(TrnSysError::from(InputError::MissingItem {
                item: "Construction".to_string(),
                name: "No Construction Found.".to_string(),
            }))?;

        let user_defined_depth: f64 = (&state.params[4]).into();

        let enable_report_writing: bool = f64::from((&state.params[5])) > 0.0;

        let output_interval_hr: f64 = (&state.params[6]).into();

        let output_file_id: i32 =
            (&state.params[7])
                .try_into()
                .map_err(|_| InputError::BadParameter {
                    index: 8,
                    message: "Output File ID must be an integer".to_string(),
                })?;

        let settings = HamtSettings::default();

        let manager = HamtManager::new(
            settings,
            construction,
            format!("S256_{}", get_current_unit()),
            surface_area,
        )?;
        state.hamt_manager = Some(manager);
        Ok(())
    }

    /// Start of simulation (first timestep): initialize HAMT mesh/state.
    pub fn simulation_starts(&self, state: &mut TrnSysState) -> Result<(), TrnSysError> {
        Ok(())
    }

    /// Last call of simulation.
    pub fn simulation_ends(&self, state: &mut TrnSysState) -> Result<()> {
        state.hamt_manager = None;
        Ok(())
    }

    /// Called at each TRNSYS call within a timestep.
    pub fn iterate(&self, state: &mut TrnSysState) -> Result<Vec<TrnSysValue>> {
        if let Some(manager) = &mut state.hamt_manager {
            manager.calculate_timestep()?;

            // Map outputs
            if let Some(surface_state) = manager.surface_curr.get(0) {
                let report = surface_state.report();
                let outputs = vec![
                    report.temp_surf_out,
                    report.temp_surf_in,
                    report.rh_surf_out,
                    report.rh_surf_in,
                    report.temp_surf_user_def,
                    report.rh_surf_user_def,
                    report.water_surf_user_def,
                    report.q_surf_ext,
                    report.q_surf_int,
                    report.m_surf_ext,
                    report.m_surf_int,
                ];
                return Ok(outputs
                    .into_iter()
                    .map(|v| TrnSysValue { value: v })
                    .collect());
            }
        }

        warn!("Iterate called before HAMT manager was initialized.");
        self.get_default_output_values(state)
    }

    /// End of timestep.
    pub fn end_of_timestep(&self, state: &mut TrnSysState) -> Result<()> {
        if let Some(manager) = &mut state.hamt_manager {
            manager.update_history();
        }
        Ok(())
    }

    /// Default outputs used during start time before real iteration.
    pub fn get_default_output_values(&self, state: &mut TrnSysState) -> Result<Vec<TrnSysValue>> {
        let defaults = vec![0.0; state.num_outputs as usize];
        Ok(defaults
            .into_iter()
            .map(|v| TrnSysValue { value: v })
            .collect())
    }
}
