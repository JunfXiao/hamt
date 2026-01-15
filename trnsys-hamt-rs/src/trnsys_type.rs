use crate::hamt::boundary::BoundaryCondition;
use crate::hamt::hamt_manager::{HamtManager, HamtSettings};
use crate::hamt::material::{Construction, Material};
use crate::trnsys::error::{InputError, TrnSysError};
use crate::trnsys::iteration_mode::IterationMode;
use crate::trnsys::param::TrnSysValue;
use crate::trnsys::*;
use std::collections::HashMap;
use tracing::warn;

pub(crate) struct TrnSysType {}

impl TrnSysType {
    /// Create a new instance.
    pub fn new() -> Self {
        TrnSysType {}
    }

    /// Very first call of the simulation: declare I/O sizes and iteration mode.
    pub fn first_call_of_simulation(&self, state: &mut TrnSysState) -> Result<(), TrnSysError> {
        state.num_params = 8;
        state.num_inputs = 11;
        state.num_outputs = 11;
        state.num_derivatives = 0;
        state.iteration_mode = IterationMode::DynamicMode;
        Ok(())
    }

    /// Validate parameters.
    pub fn validate_parameters(&self, state: &mut TrnSysState) -> Result<(), InputError> {
        if state.params.len() < 2 {
            return Err(InputError::BadParameter {
                index: 1,
                message: "At least 2 parameters are required.".to_string(),
            });
        }
        let surface_area: f64 = (&state.params[1]).into();
        if surface_area <= 0.0 {
            return Err(InputError::BadParameter {
                index: 2,
                message: "Surface area must be positive".to_string(),
            });
        }
        Ok(())
    }

    /// Start of simulation (first timestep): initialize HAMT mesh/state.
    pub fn simulation_starts(&self, state: &mut TrnSysState) -> Result<(), TrnSysError> {
        let settings = HamtSettings::default();

        // Dummy data - replace with actual data from `io.rs` later
        let mut materials = HashMap::new();
        let mat1 = Material::new("Brick".to_string());
        materials.insert("Brick".to_string(), mat1);

        let mut constructions = HashMap::new();
        todo!("Read Construction");

        let surface_area: f64 = (&state.params[1]).into();
        let surfaces_to_init = vec![(
            "TestSurface".to_string(),
            "SimpleWall".to_string(),
            surface_area,
        )];

        let manager = HamtManager::new(
            settings,
            materials,
            constructions,
            &surfaces_to_init,
            BoundaryCondition::default(),
        )?;
        state.hamt_manager = Some(manager);

        Ok(())
    }

    /// Last call of simulation.
    pub fn simulation_ends(&self, state: &mut TrnSysState) -> Result<(), TrnSysError> {
        state.hamt_manager = None;
        Ok(())
    }

    /// Called at each TRNSYS call within a timestep.
    pub fn iterate(&self, state: &mut TrnSysState) -> Result<Vec<TrnSysValue>, TrnSysError> {
        if let Some(manager) = &mut state.hamt_manager {
            manager.calculate_timestep()?;

            // Map outputs
            if let Some(surface_state) = manager.surface.get(0) {
                let outputs = vec![
                    surface_state.temp_surf_out,
                    surface_state.temp_surf_in,
                    surface_state.rh_surf_out,
                    surface_state.rh_surf_in,
                    surface_state.temp_surf_user_def,
                    surface_state.rh_surf_user_def,
                    surface_state.water_surf_user_def,
                    surface_state.q_surf_ext,
                    surface_state.q_surf_int,
                    surface_state.m_surf_ext,
                    surface_state.m_surf_int,
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
    pub fn end_of_timestep(&self, state: &mut TrnSysState) -> Result<(), TrnSysError> {
        if let Some(manager) = &mut state.hamt_manager {
            manager.update_history();
        }
        Ok(())
    }

    /// Default outputs used during start time before real iteration.
    pub fn get_default_output_values(
        &self,
        state: &mut TrnSysState,
    ) -> Result<Vec<TrnSysValue>, TrnSysError> {
        let defaults = vec![0.0; state.num_outputs as usize];
        Ok(defaults
            .into_iter()
            .map(|v| TrnSysValue { value: v })
            .collect())
    }
}
