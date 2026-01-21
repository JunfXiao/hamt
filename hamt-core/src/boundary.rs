use serde::Serialize;

use crate::{
    physics::{ATMOSPHERE_PRESSURE, KELVIN_CONV},
    uom::{
        PressureQty, Quantity,
        dimensions::{
            Dimensionless, HeatTransferCoeff, MassConvectionCoeff, Power, Temperature,
            VaporTransportCoeff,
        },
        units::{self, m_per_s, w_per_m2_k, watts},
    },
};

/// Typed boundary conditions for the simulation.
/// Temperatures are stored as absolute (Kelvin). Relative humidity is
/// dimensionless (0–1). Pressure is absolute (Pascal).
#[derive(Debug, Clone, Serialize)]
pub struct BoundaryCondition {
    /// Outside air temperature [K].
    pub temp_outside_air: Temperature<f64>,
    /// Inside air temperature [K].
    pub temp_inside_air: Temperature<f64>,
    /// Relative humidity of the outside air [-].
    pub rh_outside_air: Dimensionless<f64>,
    /// Relative humidity of the inside air [-].
    pub rh_inside_air: Dimensionless<f64>,
    /// Pressure of the air [Pa].
    pub pressure_air: PressureQty<f64>,

    /// External heat transfer coefficient [W/m²K].
    pub heat_transfer_coeff_ext: HeatTransferCoeff<f64>,

    /// Internal heat transfer coefficient [W/m²K].
    pub heat_transfer_coeff_int: HeatTransferCoeff<f64>,

    /// External mass convection coefficient [m/s].
    pub mass_convection_coeff_ext: MassConvectionCoeff<f64>,

    /// Internal mass convection coefficient [kg/m²s].
    pub mass_convection_coeff_int: MassConvectionCoeff<f64>,

    /// External vapor transport coefficient [kg/(m²·s·Pa)].
    pub vapor_transport_coeff_ext: VaporTransportCoeff<f64>,

    /// Internal vapor transport coefficient [kg/(m²·s·Pa)].
    pub vapor_transport_coeff_int: VaporTransportCoeff<f64>,

    /// External additional energy [W].
    pub additional_energy_ext: Power<f64>,

    /// Internal additional energy [W].
    pub additional_energy_int: Power<f64>,

    /// Additional heat source in user-defined cell depth [W].
    pub q_addition_userdef: Power<f64>,

    // Placeholder for future boundary conditions
    pub is_raining: bool,
}

impl BoundaryCondition {
    /// Create a new typed boundary condition from raw inputs.
    ///
    /// `temp_*_c` are in °C, `rh_*` are fractions (0–1), `pressure_pa` in Pa.
    pub fn new(
        temp_outside_c: f64,
        temp_inside_c: f64,
        rh_outside: f64,
        rh_inside: f64,
        pressure_pa: f64,
        heat_transfer_coeff_ext: f64,
        heat_transfer_coeff_int: f64,
        mass_conv_coeff_ext: f64,
        mass_conv_coeff_int: f64,
        q_additional_out: f64,
        q_additional_in: f64,
        q_additional_userdef: f64,
    ) -> Self {
        Self {
            temp_outside_air: celsius_to_kelvin(temp_outside_c),
            temp_inside_air: celsius_to_kelvin(temp_inside_c),
            rh_outside_air: Quantity(rh_outside),
            rh_inside_air: Quantity(rh_inside),
            pressure_air: pressure_pa * units::pascals,
            heat_transfer_coeff_ext: heat_transfer_coeff_ext * w_per_m2_k,
            heat_transfer_coeff_int: heat_transfer_coeff_int * w_per_m2_k,
            mass_convection_coeff_ext: mass_conv_coeff_ext * m_per_s,
            mass_convection_coeff_int: mass_conv_coeff_int * m_per_s,
            vapor_transport_coeff_ext: Quantity(0.0),
            vapor_transport_coeff_int: Quantity(0.0),
            additional_energy_ext: q_additional_out * watts,
            additional_energy_int: q_additional_in * watts,
            q_addition_userdef: q_additional_userdef * watts,
            is_raining: false,
        }
    }
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self {
            temp_outside_air: celsius_to_kelvin(20.0),
            temp_inside_air: celsius_to_kelvin(20.0),
            rh_outside_air: Quantity(0.5),
            rh_inside_air: Quantity(0.5),
            pressure_air: ATMOSPHERE_PRESSURE,
            heat_transfer_coeff_ext: 1.5 * w_per_m2_k,
            heat_transfer_coeff_int: 1.5 * w_per_m2_k,
            mass_convection_coeff_ext: 0.0 * m_per_s,
            mass_convection_coeff_int: 0.0 * m_per_s,
            vapor_transport_coeff_ext: Quantity(0.0),
            vapor_transport_coeff_int: Quantity(0.0),
            additional_energy_ext: 0.0 * watts,
            additional_energy_int: 0.0 * watts,
            q_addition_userdef: 0.0 * watts,
            is_raining: false,
        }
    }
}

/// Convert °C to K.
pub fn celsius_to_kelvin(temp_c: f64) -> Temperature<f64> {
    units::kelvin * (temp_c + KELVIN_CONV)
}

/// Convert K to °C.
pub fn kelvin_to_celsius(temp_k: Temperature<f64>) -> f64 {
    temp_k.value_in(units::kelvin) - KELVIN_CONV
}
