use std::fs::File;
use std::path::{Path, PathBuf};

use csv::Writer;

use hamt_core::hamt_manager::HamtManager;
use hamt_core::uom::dimensions::Time;
use hamt_core::uom::units::hours;
use crate::storage::StoreProvider;
use crate::trnsys::iteration_mode::IterationMode;
use crate::trnsys::param::TrnSysValue;
use crate::trnsys::{
    get_current_unit, get_input_value, get_label, get_output_value, get_parameter_value,
};

const OUTPUT_SIZE: usize = 11;

pub(crate) struct TrnSysState {
    pub(crate) trnsys_standard_version: i32,
    pub(crate) num_params: i32,
    pub(crate) params: Vec<TrnSysValue>,
    pub(crate) num_labels: i32,
    pub(crate) labels: Vec<String>,
    pub(crate) num_inputs: i32,
    pub(crate) inputs: Vec<TrnSysValue>,
    pub(crate) num_derivatives: i32,
    pub(crate) num_outputs: i32,
    pub(crate) default_output_values: Vec<TrnSysValue>,
    pub(crate) outputs: [TrnSysValue; OUTPUT_SIZE],
    pub(crate) iteration_mode: IterationMode,

    pub hamt_manager: Option<HamtManager>,
    pub csv_writer: Option<Writer<File>>,

    pub write_report: bool,
    pub write_report_interval: Time<f64>,
    pub last_report_time: Time<f64>,
}

impl TrnSysState {
    /// set up parameters for the TRNSYS type
    pub fn new() -> Self {
        TrnSysState {
            trnsys_standard_version: 18,
            iteration_mode: IterationMode::default(),

            num_params: 0,
            params: vec![],

            num_labels: 0,
            labels: vec![],

            num_inputs: 0,
            inputs: vec![],

            num_derivatives: 0,

            num_outputs: 0,
            outputs: [TrnSysValue::default(); OUTPUT_SIZE],
            default_output_values: vec![],

            hamt_manager: None,
            csv_writer: None,

            write_report: false,
            write_report_interval: 1.0 * hours,
            last_report_time: 0.0 * hours,
        }
    }

    pub fn read_parameter_values(&mut self) {
        self.params = (1..self.num_params + 1)
            .map(|i| TrnSysValue {
                value: get_parameter_value(i),
            })
            .collect();

        self.labels = (1..self.num_labels + 1)
            .map(|i| get_label(get_current_unit(), i))
            .collect();
    }

    pub fn read_input_values(&mut self) {
        self.inputs = (1..self.num_inputs + 1)
            .map(|i| TrnSysValue {
                value: get_input_value(i),
            })
            .collect();

        for i in 0..OUTPUT_SIZE {
            self.outputs[i] = TrnSysValue {
                value: get_output_value(i as i32),
            }
        }
    }
}
