use crate::trnsys::error::{Error, InputError};
use crate::trnsys::iteration_mode::IterationMode;
use crate::trnsys::param::TrnSysValue;
use crate::trnsys::*;
use anyhow::{Context, Ok, Result};
use csv::Writer;
use hamt_core::boundary::{celsius_to_kelvin, kelvin_to_celsius};
use hamt_core::hamt_manager::{HamtManager, HamtSettings};
use hamt_core::io::load_hamt_by_path;
use hamt_core::uom::units::{
    dimensionless, hours, kg_per_hr, kg_per_m3, kj_per_hr, kj_per_hr_m2_k, m_per_s, meter, square_meter
};
use std::iter;
use std::path::Path;
use tracing::{debug, info, warn};
pub(crate) struct TrnSysType {}

impl TrnSysType {
    /// Create a new instance.
    pub fn new() -> Self {
        TrnSysType {}
    }

    /// Very first call of the simulation: declare I/O sizes and iteration mode.
    pub fn first_call_of_simulation(&self, state: &mut TrnSysState) -> Result<()> {
        state.num_inputs = 11;
        state.num_outputs = 11;
        state.num_derivatives = 0;
        state.iteration_mode = IterationMode::DynamicMode;
        Ok(())
    }

    /// Validate parameters.
    pub fn validate_parameters(&self, state: &mut TrnSysState) -> Result<()> {
        if state.params.len() < 8 || state.params.len() > 9 {
            return Err(InputError::BadParameter {
                index: 1,
                message: "8 or 9 parameters are required.".to_string(),
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
        let mat_library = load_hamt_by_path(file_path)?;
        let construction = mat_library
            .get_construction_read_lock()?
            .values()
            .next()
            .cloned()
            .ok_or(Error::from(InputError::MissingItem {
                item: "Construction".to_string(),
                name: "No Construction Found.".to_string(),
            }))?;

        let user_defined_depth: f64 = (&state.params[4]).into();

        state.write_report = f64::from(&state.params[5]) > 0.0;

        state.write_report_interval = (&state.params[6]).value * hours;

        let mut settings = HamtSettings {
            timestep: get_simulation_time_step() * hours,
            user_defined_depth: Some(user_defined_depth * meter),
            ..Default::default()
        };
        
        // Note: Param 7 is handled in the main function to set up output files, 
        // so we don't validate it here.

        if let Some(n_div) = state.params.get(8) {
            let n_div_value: i32 = n_div.try_into().map_err(|_| InputError::BadParameter {
                index: 9,
                message: "Number of divisions must be an integer".to_string(),
            })?;
            if n_div_value <= 0 {
                return Err(InputError::BadParameter {
                    index: 9,
                    message: "Number of divisions must be positive".to_string(),
                }
                .into());
            }
            settings.min_div = n_div_value;
        }

        let manager = HamtManager::new(
            Some(settings),
            None,
            construction,
            format!("S256_{}", get_current_unit()),
            surface_area * square_meter,
        )?;
        debug!("Initialized");
        state.hamt_manager = Some(manager);
        debug!(
            "Report after initialization:\n{:#?}",
            self.update_output(state, true)
        );

        // Output Files

        let output_file_id: i32 =
            (&state.params[7])
                .try_into()
                .map_err(|_| InputError::BadParameter {
                    index: 8,
                    message: "Output File ID must be an integer".to_string(),
                })?;

        let t_output_path = format!("Type395_0{}_t.out", output_file_id);
        state.t_csv_writer = Some(Writer::from_path(t_output_path)?);

        let rh_output_path = format!("Type395_0{}_rh.out", output_file_id);
        state.rh_csv_writer = Some(Writer::from_path(rh_output_path)?);

        let general_output_path = format!("Type395_0{}.out", output_file_id);
        state.output_csv_writer = Some(Writer::from_path(general_output_path)?);

        let water_output_path = format!("Type395_0{}_water.out", output_file_id);
        state.water_csv_writer = Some(Writer::from_path(water_output_path)?);

        // Write t title columns
        let cell_table_titles = vec!["time [h]".to_string()].into_iter().chain(
            (0..state
                .hamt_manager
                .as_ref()
                .unwrap()
                .surface_curr
                .cells
                .len())
                .map(|i| i.to_string()),
        );

        // cell depth information
        let cells_ref = state
            .hamt_manager
            .as_ref()
            .unwrap()
            .surface_curr
            .cells
            .clone();
        let cell_depth_titles = vec!["depth [m]".to_string()].into_iter().chain(
            (0..state
                .hamt_manager
                .as_ref()
                .unwrap()
                .surface_curr
                .cells
                .len())
                .map(|i| format!("{:.3e}", cells_ref[[i, 0, 0]].origin.x().value_in(meter))),
        );
        let cell_thickness_titles = vec!["thickness [m]".to_string()].into_iter().chain(
            (0..state
                .hamt_manager
                .as_ref()
                .unwrap()
                .surface_curr
                .cells
                .len())
                .map(|i| format!("{:.3e}", cells_ref[[i, 0, 0]].size.x().value_in(meter))),
        );
        let cell_material_titles = vec!["material".to_string()].into_iter().chain(
            (0..state
                .hamt_manager
                .as_ref()
                .unwrap()
                .surface_curr
                .cells
                .len())
                .map(|i| format!("{}", cells_ref[[i, 0, 0]].cell_type)),
        );

        if let Some(writer) = &mut state.t_csv_writer {
            writer.write_record(cell_table_titles.clone())?;
            writer.write_record(cell_depth_titles.clone())?;
            writer.write_record(cell_thickness_titles.clone())?;
            writer.write_record(cell_material_titles.clone())?;
        }

        if let Some(writer) = &mut state.rh_csv_writer {
            writer.write_record(cell_table_titles.clone())?;
            writer.write_record(cell_depth_titles.clone())?;
            writer.write_record(cell_thickness_titles.clone())?;
            writer.write_record(cell_material_titles.clone())?;
        }

        if let Some(writer) = &mut state.water_csv_writer {
            writer.write_record(cell_table_titles)?;
            writer.write_record(cell_depth_titles)?;
            writer.write_record(cell_thickness_titles)?;
            writer.write_record(cell_material_titles)?;
        }

        // Write output title columns
        if let Some(writer) = &mut state.output_csv_writer {
            writer.write_record([
                "Time [h]",
                "T_in [C]",
                "T_out [C]",
                "RH_in [-]",
                "RH_out [-]",
                "T_userdef [C]",
                "RH_userdef [-]",
                "Water_userdef [kg/m3]",
                "HeatFlux_in [KJ/hr]",
                "HeatFlux_out [KJ/hr]",
                "Water_in [kg/h]",
                "Water_out [kg/h]",
                "HeatFlux_in [W]",
                "HeatFlux_out [W]",
            ])?;
        }

        Ok(())
    }

    /// Start of simulation (first timestep): initialize HAMT mesh/state.
    pub fn simulation_starts(&self, state: &mut TrnSysState) -> Result<()> {
        info!("Simulation Begins");
        self.update_output(state, true)?;
        Ok(())
    }

    /// Last call of simulation.
    pub fn simulation_ends(&self, state: &mut TrnSysState) -> Result<()> {
        state.hamt_manager = None;
        Ok(())
    }

    /// Report simulation data.
    ///
    /// Values:
    /// T_in
    /// - T_out
    /// - RH_in
    /// - RH_out
    /// - T_userdef
    /// - RH_userdef
    /// - MassFlux_userdef
    /// - HeatFlux_in
    /// - HeatFlux_out
    /// - MassFlux_in
    /// - MassFlux_out

    pub fn update_output(&self, state: &mut TrnSysState, write_file: bool) -> Result<()> {
        let report_row = if let Some(manager) = &mut state.hamt_manager {
            // Map outputs
            let report = manager.surface_curr.report(&manager.boundary_condition)?;
            let surface_area = manager.surface_curr.area;
            let outputs = [
                kelvin_to_celsius(report.temp_surf_int),
                kelvin_to_celsius(report.temp_surf_out),
                report.rh_surf_int.value(),
                report.rh_surf_out.value(),
                kelvin_to_celsius(report.temp_surf_user_def),
                report.rh_surf_user_def.value(),
                report.water_surf_user_def.value_in(kg_per_m3),
                (report.q_surf_int * surface_area).value_in(kj_per_hr),
                (report.q_surf_ext * surface_area).value_in(kj_per_hr),
                (report.m_surf_int * surface_area).value_in(kg_per_hr),
                (report.m_surf_ext * surface_area).value_in(kg_per_hr),
            ];
            outputs
        } else {
            [
                20.0, // Inside Surface Temperature
                20.0, // Outside Surface Temperature
                0.1,  // Inside Relative Humidity
                0.1,  // Outside Relative Humidity
                20.0, // User-defined Surface Temperature
                0.0,  // Inside Water Vapor Density
                0.0,  // Outside Water Vapor Density
                0.0,  // Inside Heat Flux
                0.0,  // Outside Heat Flux
                0.0,  // Inside Mass Flux
                0.0,  // Outside Mass Flux
            ]
        };

        if write_file && let Some(writer) = &mut state.output_csv_writer {
            let time = get_simulation_time();
            let report_rows = iter::once(time)
                .chain(report_row.clone().into_iter())
                // chain with additional heat flux outputs in W
                .chain(
                    [
                        report_row[7] * 1000.0 / 3600.0, // Convert kJ/hr to W
                        report_row[8] * 1000.0 / 3600.0, // Convert kJ/hr to W
                    ]
                    .into_iter(),
                )
                .map(|v| v.to_string());
            writer.write_record(report_rows).with_context(|| {
                format!(
                    "Error writing report row at time {}\n{:?}",
                    time, report_row
                )
            })?;
            writer.flush()?;
            self.output_cell_states(state)?
        }

        for (i, v) in report_row.into_iter().enumerate() {
            state.outputs[i] = TrnSysValue { value: v };
        }

        // default value
        Ok(())
    }

    pub fn output_cell_states(&self, state: &mut TrnSysState) -> Result<()> {
        // Output cell RH
        if let Some(writer) = &mut state.rh_csv_writer {
            let time = get_simulation_time();
            let cell_rh_rows = iter::once(time.to_string()).chain(
                state
                    .hamt_manager
                    .as_ref()
                    .unwrap()
                    .surface_curr
                    .cells
                    .iter()
                    .map(|c| format!("{:.3e}", c.cell_rh.value())),
            );
            writer
                .write_record(cell_rh_rows)
                .context("Error writing cell RH row")?;
            writer.flush()?;
        }

        // Output cell temperature
        if let Some(writer) = &mut state.t_csv_writer {
            let time = get_simulation_time();
            let cell_temp_rows = iter::once(time.to_string()).chain(
                state
                    .hamt_manager
                    .as_ref()
                    .unwrap()
                    .surface_curr
                    .cells
                    .iter()
                    .map(|c| format!("{:.3e}", kelvin_to_celsius(c.cell_temp))),
            );
            writer
                .write_record(cell_temp_rows)
                .context("Error writing cell temperature row")?;
            writer.flush()?;
        }

        // Output cell water content
        if let Some(writer) = &mut state.water_csv_writer {
            let time = get_simulation_time();
            let cell_water_rows = iter::once(time.to_string()).chain(
                state
                    .hamt_manager
                    .as_ref()
                    .unwrap()
                    .surface_curr
                    .cells
                    .iter()
                    .map(|c| format!("{:.3e}", c.volumetric_water.value_in(kg_per_m3))),
            );
            writer
                .write_record(cell_water_rows)
                .context("Error writing cell water content row")?;
            writer.flush()?;
        }

        Ok(())
    }

    /// Called at each TRNSYS call within a timestep.
    pub fn iterate(&self, state: &mut TrnSysState) -> Result<()> {
        let trnsys_iter = get_timestep_iteration();
        let trnsys_time = get_simulation_time();
        debug!(
            time = trnsys_time,
            trnsys_iter,
            "iterate: TRNSYS calling Type395"
        );

        // Read Input
        let t_int = state.inputs[0].value;
        let t_out = state.inputs[1].value;
        let rh_int = state.inputs[2].value;
        let rh_out = state.inputs[3].value;
        let hconv_int = state.inputs[4].value;
        let hconv_out = state.inputs[5].value;
        let q_int = state.inputs[6].value;
        let q_out = state.inputs[7].value;
        let hmass_conv_int = state.inputs[8].value;
        let hmass_conv_out = state.inputs[9].value;
        let q_additional_userdef = state.inputs[10].value;

        if let Some(manager) = &mut state.hamt_manager {
            manager.boundary_condition.temp_inside_air = celsius_to_kelvin(t_int);
            manager.boundary_condition.temp_outside_air = celsius_to_kelvin(t_out);
            manager.boundary_condition.rh_outside_air = rh_out / 100.0 * dimensionless;
            manager.boundary_condition.rh_inside_air = rh_int / 100.0 * dimensionless;
            manager.boundary_condition.heat_transfer_coeff_ext = hconv_out * kj_per_hr_m2_k;
            manager.boundary_condition.heat_transfer_coeff_int = hconv_int * kj_per_hr_m2_k;
            manager.boundary_condition.additional_energy_ext = q_out * kj_per_hr;
            manager.boundary_condition.additional_energy_int = q_int * kj_per_hr;
            manager.boundary_condition.mass_convection_coeff_ext = hmass_conv_out * m_per_s;
            manager.boundary_condition.mass_convection_coeff_int = hmass_conv_int * m_per_s;
            manager.boundary_condition.q_addition_userdef = q_additional_userdef * kj_per_hr;

            debug!(
                "Time = {}, bc: \n{:#?}",
                get_simulation_time(),
                manager.boundary_condition
            );

            let solve_start = std::time::Instant::now();
            let result = manager
                .solve_timestep_adaptive(&mut state.adaptive_state, None)
                .with_context(|| {
                    format!(
                        "Solving time step {}, iteration {}",
                        trnsys_time,
                        trnsys_iter
                    )
                });
            let elapsed = solve_start.elapsed();
            if elapsed.as_millis() > 500 {
                warn!(
                    time = trnsys_time,
                    trnsys_iter,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "iterate: solve took >500ms"
                );
            }
            result?;

            // Map outputs
            return self.update_output(state, false);
        }

        warn!("Iterate called before HAMT manager was initialized.");

        Ok(())
    }

    /// End of timestep.
    pub fn end_of_timestep(&self, state: &mut TrnSysState) -> Result<()> {
        let current_time = get_simulation_time() * hours;
        let should_report = state.write_report
            && (current_time - state.last_report_time) >= state.write_report_interval;

        self.update_output(state, should_report)?;

        if should_report {
            state.last_report_time = current_time;
        }
        state
            .hamt_manager
            .as_mut()
            .with_context(|| "End of timestep")?
            .tick_timestep()
            .with_context(|| format!("At the end of time step {}", get_simulation_time()))
    }

    /// Default outputs used during start time before real iteration.
    pub fn set_default_output_values(&self, state: &mut TrnSysState) -> Result<()> {
        self.update_output(state, false)
    }
}
