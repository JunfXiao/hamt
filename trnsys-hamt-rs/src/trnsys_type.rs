use crate::trnsys::error::{Error, InputError};
use crate::trnsys::iteration_mode::IterationMode;
use crate::trnsys::param::TrnSysValue;
use crate::trnsys::*;
use anyhow::{Context, Ok, Result};
use csv::Writer;
use hamt_core::boundary::{celsius_to_kelvin, kelvin_to_celsius};
use hamt_core::hamt_manager::{HamtManager, HamtSettings, SolverMethod};
use hamt_core::io::load_hamt_by_path;
use hamt_core::uom::units::{
    dimensionless, hours, kg_per_hr, kg_per_m3, kg_per_m2_s_pa, kj_per_hr, kj_per_hr_m2_k, meter, pascals, seconds, square_meter, w_per_m2_k,
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
        if state.num_inputs >= 11 && state.num_inputs <= 14 {
            // valid
        } else {
            return Err(InputError::BadParameter {
                index: 1,
                message: format!("Number of inputs must be 11 to 14. Got: {}", state.num_inputs),
            }
            .into());

        }

        state.num_outputs = 11;
        state.num_derivatives = 0;
        state.iteration_mode = IterationMode::DynamicMode;
        Ok(())
    }

    /// Validate parameters.
    pub fn validate_parameters(&self, state: &mut TrnSysState) -> Result<()> {
        if state.params.len() < 8 || state.params.len() > 10 {
            return Err(InputError::BadParameter {
                index: 1,
                message: "8, 9, or 10 parameters are required.".to_string(),
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

        // to avoid floating point precision issue, truncate to 4 decimal places (0.0001 hours = 0.36 seconds)
        state.write_report_interval = ((&state.params[6]).value * 1e4).round() / 1e4 * hours;
        


        info!("Write report: {}, interval: {} hours", state.write_report, state.write_report_interval.value_in(hours));

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

        // Optional param 10: inner solver selector. 0 = Picard, 1 = BlockTridiagonal.
        if let Some(p) = state.params.get(9) {
            let v: i32 = p.try_into().map_err(|_| InputError::BadParameter {
                index: 10,
                message: format!("Solver method must be 0 (Picard) or 1 (BlockTridiagonal). Got: {:?}", p),
            })?;
            settings.solver_method = match v {
                0 => SolverMethod::BlockJacobi,
                1 => SolverMethod::BlockTridiagonal,
                _ => {
                    return Err(InputError::BadParameter {
                        index: 10,
                  message: format!("Solver method must be 0 (Picard) or 1 (BlockTridiagonal). Got: {:?}", p),
                    }
                    .into());
                }
            };
            info!("Selected solver_method = {:?}", settings.solver_method);
        }

        settings.min_relaxation_factor = 0.01; // allow more aggressive refinement for better convergence in TRNSYS

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

        let vp_output_path = format!("Type395_0{}_vp.out", output_file_id);
        state.vp_csv_writer = Some(Writer::from_path(vp_output_path)?);

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
            writer.write_record(cell_table_titles.clone())?;
            writer.write_record(cell_depth_titles.clone())?;
            writer.write_record(cell_thickness_titles.clone())?;
            writer.write_record(cell_material_titles.clone())?;
        }

        if let Some(writer) = &mut state.vp_csv_writer {
            writer.write_record(cell_table_titles)?;
            writer.write_record(cell_depth_titles)?;
            writer.write_record(cell_thickness_titles)?;
            writer.write_record(cell_material_titles)?;
        }

        // Boundary condition debug output
        let boundary_output_path = format!("Type395_0{}_boundary.out", output_file_id);
        state.boundary_csv_writer = Some(Writer::from_path(boundary_output_path)?);
        if let Some(writer) = &mut state.boundary_csv_writer {
            let headers = vec![
                "Time [h]",
                "Iter",
                "NInputs",
                // Inputs 0-10 (always present)
                "In0_T_int_TSTAR [C]",
                "In1_T_out [C]",
                "In2_RH_int [%]",
                "In3_RH_out [%]",
                "In4_hconv_int [kJ/hr.m2.K]",
                "In5_hconv_out [kJ/hr.m2.K]",
                "In6_q_int [kJ/hr]",
                "In7_q_out [kJ/hr]",
                "In8_beta_int [kg/m2.s.Pa]",
                "In9_beta_out [kg/m2.s.Pa]",
                "In10_q_userdef [kJ/hr]",
                // Inputs 11-13 (optional, 14-input mode)
                "In11_T_int_moist [C]",
                "In12_HCONVI [kJ/hr.m2.K]",
                "In13_TMR [C]",
                // Derived BC fields
                "BC_has_sep_rad",
                "BC_h_conv_int [W/m2.K]",
                "BC_h_rad_int [W/m2.K]",
                "BC_vtc_int [kg/m2.s.Pa]",
                "BC_add_energy_int [W]",
                // Flux diagnostics (interior side)
                "T_conv_BC [C]",
                "T_rad_BC [C]",
                "T_virtual_int [C]",
                "T_surf_mat_int [C]",
                "Q_conv_int [W]",
                "Q_rad_int [W]",
                "Q_conv+rad_int [W]",
                "VP_conv_BC [Pa]",
                "VP_virtual_int [Pa]",
                "MoistFlux_int [g/h]",
            ];
            writer.write_record(&headers)?;
            writer.flush()?;
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
            let report: hamt_core::surface::SurfaceReport = manager.surface_curr.report(&manager.boundary_condition)?;
            let surface_area = manager.surface_curr.area;

            // Use time-averaged fluxes from adaptive accumulator when available.
            // This corrects the endpoint-vs-average error that occurs with substepping.
            let acc = &state.adaptive_state.flux_accumulator;
            let (q_int_kj_hr, q_ext_kj_hr, m_int_kg_hr, m_ext_kg_hr) = if acc.total_time > 0.0 * seconds {
                (
                    (acc.avg_q_int() * surface_area).value_in(kj_per_hr),
                    (acc.avg_q_ext() * surface_area).value_in(kj_per_hr),
                    (acc.avg_m_int() * surface_area).value_in(kg_per_hr),
                    (acc.avg_m_ext() * surface_area).value_in(kg_per_hr),
                )
            } else {
                // Fallback to endpoint fluxes (no substepping or first call)
                (
                    (report.q_surf_int * surface_area).value_in(kj_per_hr),
                    (report.q_surf_ext * surface_area).value_in(kj_per_hr),
                    (report.m_surf_int * surface_area).value_in(kg_per_hr),
                    (report.m_surf_ext * surface_area).value_in(kg_per_hr),
                )
            };

            let outputs = [
                kelvin_to_celsius(report.temp_surf_int),
                kelvin_to_celsius(report.temp_surf_out),
                report.rh_surf_int.value(),
                report.rh_surf_out.value(),
                kelvin_to_celsius(report.temp_surf_user_def),
                report.rh_surf_user_def.value(),
                report.water_surf_user_def.value_in(kg_per_m3),
                q_int_kj_hr,
                q_ext_kj_hr,
                m_int_kg_hr,
                m_ext_kg_hr,
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
            self.output_cell_states(state)?;
            self.output_boundary_state(state)?;
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

        // Output cell vapor pressure [Pa]
        if let Some(writer) = &mut state.vp_csv_writer {
            let time = get_simulation_time();
            let cell_vp_rows = iter::once(time.to_string()).chain(
                state
                    .hamt_manager
                    .as_ref()
                    .unwrap()
                    .surface_curr
                    .cells
                    .iter()
                    .map(|c| format!("{:.3e}", c.cell_vp.value_in(pascals))),
            );
            writer
                .write_record(cell_vp_rows)
                .context("Error writing cell VP row")?;
            writer.flush()?;
        }

        Ok(())
    }

    /// Output boundary condition state for debugging.
    /// Writes one row per reporting interval with all raw inputs and derived BC fields.
    pub fn output_boundary_state(&self, state: &mut TrnSysState) -> Result<()> {
        if let Some(writer) = &mut state.boundary_csv_writer {
            let time = get_simulation_time();
            let iter = get_timestep_iteration();
            let n_inputs = state.inputs.len();

            // Raw inputs (always present: 0-10)
            let in0  = state.inputs[0].value;   // T_int TSTAR [C]
            let in1  = state.inputs[1].value;   // T_out [C]
            let in2  = state.inputs[2].value;   // RH_int [%]
            let in3  = state.inputs[3].value;   // RH_out [%]
            let in4  = state.inputs[4].value;   // hconv_int [kJ/hr.m2.K]
            let in5  = state.inputs[5].value;   // hconv_out [kJ/hr.m2.K]
            let in6  = state.inputs[6].value;   // q_int [kJ/hr]
            let in7  = state.inputs[7].value;   // q_out [kJ/hr]
            let in8  = state.inputs[8].value;   // beta_int [kg/m2.s.Pa]
            let in9  = state.inputs[9].value;   // beta_out [kg/m2.s.Pa]
            let in10 = state.inputs[10].value;  // q_userdef [kJ/hr]

            // Optional inputs (14-input mode)
            let in11 = state.inputs.get(11).map_or(f64::NAN, |v| v.value); // T_int_moist [C]
            let in12 = state.inputs.get(12).map_or(f64::NAN, |v| v.value); // HCONVI [kJ/hr.m2.K]
            let in13 = state.inputs.get(13).map_or(f64::NAN, |v| v.value); // TMR [C]

            // Derived BC fields and flux diagnostics from the manager
            let (
                has_sep_rad, h_conv_int_w, h_rad_int_w, vtc_int, add_energy_w,
                t_conv_bc, t_rad_bc, t_virtual_int, t_surf_mat_int,
                q_conv_w, q_rad_w, q_total_w,
                vp_conv_bc, vp_virtual_int, moist_flux_g_h,
            ) = if let Some(manager) = &state.hamt_manager {
                    let bc = &manager.boundary_condition;
                    let cells = &manager.surface_curr.cells;
                    let area = manager.surface_curr.area.value_in(square_meter);
                    let has_sep = if bc.has_separate_radiant_int { 1.0 } else { 0.0 };
                    let h_conv = bc.heat_transfer_coeff_conv_int.value_in(w_per_m2_k);
                    let vtc = bc.vapor_transport_coeff_int.value_in(kg_per_m2_s_pa);
                    let add_e = bc.additional_energy_int.value_in(kj_per_hr) * 1000.0 / 3600.0;

                    // Find interior BC cells and virtual cell
                    let conv_bc = cells.iter().find(|c| matches!(&c.cell_type,
                        hamt_core::cell::CellType::Boundary(
                            hamt_core::cell::Boundary::Convection,
                            hamt_core::cell::BoundaryLocation::Interior)));
                    let rad_bc = cells.iter().find(|c| matches!(&c.cell_type,
                        hamt_core::cell::CellType::Boundary(
                            hamt_core::cell::Boundary::Radiation,
                            hamt_core::cell::BoundaryLocation::Interior)));
                    let virt = cells.iter().find(|c| matches!(&c.cell_type,
                        hamt_core::cell::CellType::Virtual(
                            hamt_core::cell::BoundaryLocation::Interior)));
                    // Interior surface material cell (first material cell from interior side)
                    let surf_mat_idx = manager.surface_curr.interior_surface_material_idx;
                    let surf_mat = surf_mat_idx.and_then(|i| cells.get([i, 0, 0]));

                    let h_rad_val = rad_bc.map_or(0.0, |c| c.htc.value_in(w_per_m2_k));

                    let t_conv = conv_bc.map_or(f64::NAN, |c| kelvin_to_celsius(c.cell_temp));
                    let t_rad = rad_bc.map_or(f64::NAN, |c| kelvin_to_celsius(c.cell_temp));
                    let t_virt = virt.map_or(f64::NAN, |c| kelvin_to_celsius(c.cell_temp));
                    let t_smat = surf_mat.map_or(f64::NAN, |c| kelvin_to_celsius(c.cell_temp));

                    // Q_conv = h_conv * (T_convBC - T_virtual) * area [W]
                    let q_conv = if let (Some(cb), Some(v)) = (conv_bc, virt) {
                        let dt = (cb.cell_temp - v.cell_temp).value_in(diman::si::units::kelvin);
                        cb.htc.value_in(w_per_m2_k) * dt * area
                    } else { 0.0 };

                    // Q_rad = h_rad * (T_radBC - T_virtual) * area [W]
                    let q_rad = if let (Some(rb), Some(v)) = (rad_bc, virt) {
                        let dt = (rb.cell_temp - v.cell_temp).value_in(diman::si::units::kelvin);
                        rb.htc.value_in(w_per_m2_k) * dt * area
                    } else { 0.0 };

                    // VP diagnostics
                    let vp_conv = conv_bc.map_or(0.0, |c| c.cell_vp.value_in(pascals));
                    let vp_virt = virt.map_or(0.0, |c| c.cell_vp.value_in(pascals));

                    // Moisture flux = vtc * (VP_conv_BC - VP_virtual) * area [kg/s] -> g/h
                    let mf = if let (Some(cb), Some(v)) = (conv_bc, virt) {
                        let dvp = (cb.cell_vp - v.cell_vp).value_in(pascals);
                        let cell_vtc = if cb.vtc.value_in(kg_per_m2_s_pa) > 1e-20 {
                            cb.vtc.value_in(kg_per_m2_s_pa)
                        } else { vtc };
                        cell_vtc * dvp * area * 3600.0 * 1000.0 // kg/s -> g/h
                    } else { 0.0 };

                    (has_sep, h_conv, h_rad_val, vtc, add_e,
                     t_conv, t_rad, t_virt, t_smat,
                     q_conv, q_rad, q_conv + q_rad,
                     vp_conv, vp_virt, mf)
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0,
                     f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)
                };

            let row: Vec<String> = vec![
                format!("{:.4}", time),
                format!("{}", iter),
                format!("{}", n_inputs),
                format!("{:.4}", in0),
                format!("{:.4}", in1),
                format!("{:.4}", in2),
                format!("{:.4}", in3),
                format!("{:.6}", in4),
                format!("{:.6}", in5),
                format!("{:.4}", in6),
                format!("{:.4}", in7),
                format!("{:.3e}", in8),
                format!("{:.3e}", in9),
                format!("{:.4}", in10),
                format!("{:.4}", in11),
                format!("{:.6}", in12),
                format!("{:.4}", in13),
                format!("{:.0}", has_sep_rad),
                format!("{:.4}", h_conv_int_w),
                format!("{:.4}", h_rad_int_w),
                format!("{:.3e}", vtc_int),
                format!("{:.4}", add_energy_w),
                // Flux diagnostics
                format!("{:.4}", t_conv_bc),
                format!("{:.4}", t_rad_bc),
                format!("{:.4}", t_virtual_int),
                format!("{:.4}", t_surf_mat_int),
                format!("{:.2}", q_conv_w),
                format!("{:.2}", q_rad_w),
                format!("{:.2}", q_total_w),
                format!("{:.2}", vp_conv_bc),
                format!("{:.2}", vp_virtual_int),
                format!("{:.4}", moist_flux_g_h),
            ];
            writer.write_record(&row)?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Called at each TRNSYS call within a timestep.
    ///
    /// Expected inputs (up to 14 total; backward compatible — decks with 11/12/13 inputs
    /// fall back to legacy behaviour for any missing trailing inputs):
    /// 0. T_int        – inside star-node temperature TSTAR [°C] (used for heat convection path)
    /// 1. T_out        – outside air temperature [°C]
    /// 2. RH_int       – inside relative humidity [%]
    /// 3. RH_out       – outside relative humidity [%]
    /// 4. hconv_int    – inside convective heat transfer coefficient [kJ/(hr·m²·K)]
    /// 5. hconv_out    – outside convective heat transfer coefficient [kJ/(hr·m²·K)]
    /// 6. q_int        – additional energy source on inside surface [kJ/hr]
    /// 7. q_out        – additional energy source on outside surface [kJ/hr]
    /// 8. beta_int     – inside vapor transport coefficient β [kg/(m²·s·Pa)]
    ///                   Deck formula: β ≈ 7e-9 × h_conv[W/m²K] (Lewis analogy at ~20 °C)
    ///                   i.e. `0.000000007 * hconv_kJ_hr_m2K / 3.6`
    /// 9. beta_out     – outside vapor transport coefficient β [kg/(m²·s·Pa)]  (same formula)
    /// 10. q_userdef   – additional heat gain at user-defined depth [kJ/hr]
    /// 11. T_int_moist – actual inside air temperature TAIR [°C], used only for moisture VP_bc.
    ///                   Decouples the moisture boundary VP from the star-node temperature.
    ///                   Falls back to T_int (input 0) when the deck provides only 11 inputs.
    /// 12. HCONVI      – pure convective interior HTC (TRNBuild NType 108) [kJ/(hr·m²·K)].
    ///                   Excludes the radiation component embedded in the REQV-derived hconv_int.
    ///                   Falls back to input 4 (combined hconv_int) for decks with <13 inputs.
    /// 13. TMR         – interior mean radiant temperature (TRNBuild NType 120) [°C], used by
    ///                   the separate interior Radiation boundary cell for linearised longwave
    ///                   exchange. Falls back to T_int (input 0, TSTAR) for decks with <14 inputs.
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
        let beta_int = state.inputs[8].value;  // vapor transport coeff β [kg/(m²·s·Pa)]
        let beta_out = state.inputs[9].value;  // vapor transport coeff β [kg/(m²·s·Pa)]
        let q_additional_userdef = state.inputs[10].value;
        // Input 11 (optional): actual room air temperature TAIR for moisture VP_bc.
        // Falls back to T_int (TSTAR) when the deck still provides only 11 inputs.
        let t_int_moisture = state.inputs.get(11).map_or(t_int, |input| input.value);
        // Input 12 (optional): pure convective HTC [kJ/(hr·m²·K)] — NType 108 HCONVI.
        // Falls back to input 4 (combined hconv_int) when the deck still provides <13 inputs.
        let hconv_conv_int = state.inputs.get(12).map_or(hconv_int, |input| input.value);
        // Input 13 (optional): mean radiant temperature [°C] — NType 120 TMR.
        // Falls back to input 0 (TSTAR) when the deck still provides <14 inputs.
        let tmr_int = state.inputs.get(13).map_or(t_int, |input| input.value);


        if let Some(manager) = &mut state.hamt_manager {
            manager.boundary_condition.temp_inside_air = celsius_to_kelvin(t_int);
            manager.boundary_condition.temp_outside_air = celsius_to_kelvin(t_out);
            manager.boundary_condition.rh_outside_air = rh_out / 100.0 * dimensionless;
            manager.boundary_condition.rh_inside_air = rh_int / 100.0 * dimensionless;
            manager.boundary_condition.heat_transfer_coeff_ext = hconv_out * kj_per_hr_m2_k;
            manager.boundary_condition.heat_transfer_coeff_int = hconv_int * kj_per_hr_m2_k;
            manager.boundary_condition.additional_energy_ext = q_out * kj_per_hr;
            manager.boundary_condition.additional_energy_int = q_int * kj_per_hr;
            // Inputs 8/9 are vapor transport coefficients β [kg/(m²·s·Pa)], set directly.
            // The deck formula is: β ≈ 7e-9 × h_conv[W/m²K] (Lewis analogy).
            // Do NOT use mass_convection_coeff_ext/int here — those trigger an extra ÷(R_v·T)
            // division in HamtManager::solve_timestep that would make β ~135 000× too small.
            manager.boundary_condition.vapor_transport_coeff_ext = beta_out * kg_per_m2_s_pa;
            manager.boundary_condition.vapor_transport_coeff_int = beta_int * kg_per_m2_s_pa;
            // Set the new decoupled interior thermal BCs (with legacy fallbacks).
            manager.boundary_condition.heat_transfer_coeff_conv_int = hconv_conv_int * kj_per_hr_m2_k;
            manager.boundary_condition.temp_inside_mean_radiant = celsius_to_kelvin(tmr_int);
            // Enable the separated radiant path only when the deck supplies both inputs 12
            // (HCONVI) and 13 (TMR). Legacy decks keep the inert Radiation cell so they
            // remain byte-identical to the pre-refactor combined-potential behaviour.
            manager.boundary_condition.has_separate_radiant_int = state.inputs.len() >= 14;
            // Lewis-analogy fallback: when the deck passes β_int = 0, derive it from the pure
            // convective HTC (HCONVI). 7e-9 [kg/(m²·s·Pa)] per [W/(m²·K)] is the Lewis-number
            // coefficient at ~20 °C used throughout the d18 deck.
            if manager.boundary_condition.vapor_transport_coeff_int.value_in(kg_per_m2_s_pa) == 0.0 {
                let h_w = manager
                    .boundary_condition
                    .heat_transfer_coeff_conv_int
                    .value_in(w_per_m2_k);
                manager.boundary_condition.vapor_transport_coeff_int =
                    7.0e-9 * h_w * kg_per_m2_s_pa;
            }
            manager.boundary_condition.q_addition_userdef = q_additional_userdef * kj_per_hr;
            manager.boundary_condition.temp_inside_air_moisture = celsius_to_kelvin(t_int_moisture);
            // Outside moisture temperature equals outside air temperature (no star-node effect outside).
            manager.boundary_condition.temp_outside_air_moisture = celsius_to_kelvin(t_out);

            debug!(
                "Time = {}, bc: \n{:#?}",
                get_simulation_time(),
                manager.boundary_condition
            );

            // let solve_start = std::time::Instant::now();
            
            // Set the current time in the adaptive state for use in substepping. This ensures that any time-dependent
            state.adaptive_state.current_time = get_simulation_time() * hours;
            manager
                .solve_timestep_adaptive(&mut state.adaptive_state, None)
                .with_context(|| {
                    format!(
                        "Solving time step {}, iteration {}",
                        trnsys_time,
                        trnsys_iter
                    )
                })?;


            // let elapsed = solve_start.elapsed();
            // if elapsed.as_millis() > 500 {
            //     warn!(
            //         time = trnsys_time,
            //         trnsys_iter,
            //         elapsed_ms = elapsed.as_millis() as u64,
            //         "iterate: solve took >500ms"
            //     );
            // }
            // result?;

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
