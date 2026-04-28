use serde::Serialize;

use crate::{
    physics::{ KELVIN_CONV},
    uom::{
        Quantity,
        dimensions::{
            Dimensionless, HeatTransferCoeff, MassConvectionCoeff, Power, Pressure, Temperature, VaporTransportCoeff
        },
        units::{self, m_per_s, w_per_m2_k, watts},
        constants::{ATMOSPHERE_PRESSURE}
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
    pub pressure_air: Pressure<f64>,

    /// External heat transfer coefficient [W/m²K].
    pub heat_transfer_coeff_ext: HeatTransferCoeff<f64>,

    /// Internal heat transfer coefficient [W/m²K].
    pub heat_transfer_coeff_int: HeatTransferCoeff<f64>,

    /// External mass convection coefficient [m/s].
    pub mass_convection_coeff_ext: MassConvectionCoeff<f64>,

    /// Internal mass convection coefficient [m/s].
    pub mass_convection_coeff_int: MassConvectionCoeff<f64>,

    /// External vapor transport coefficient [kg/(m²·s·Pa)].
    pub vapor_transport_coeff_ext: VaporTransportCoeff<f64>,

    /// Internal vapor transport coefficient [kg/(m²·s·Pa)].
    pub vapor_transport_coeff_int: VaporTransportCoeff<f64>,

    /// Room air temperature used only for interior moisture VP: VP_bc = RH × Psat(T).
    /// Distinct from `temp_inside_air`, which is the star-node temperature (TSTAR) used
    /// for the heat convection path paired with the REQV-derived h_conv.
    /// Set this to the actual zone air temperature (TAIR from TRNBuild [56,1]).
    pub temp_inside_air_moisture: Temperature<f64>,

    /// Outside air temperature used only for exterior moisture VP.
    /// Defaults to `temp_outside_air` (equal in the current d18 deck).
    pub temp_outside_air_moisture: Temperature<f64>,

    /// Mean radiant temperature seen by the interior surface (TMR, TRNBuild NType 120). [K]
    pub temp_inside_mean_radiant: Temperature<f64>,

    /// Pure convective interior HTC (HCONVI, NType 108), excludes the radiation
    /// component embedded in REQV-derived h_conv. [W/(m²·K)]
    pub heat_transfer_coeff_conv_int: HeatTransferCoeff<f64>,

    /// True when the deck supplies the separated radiant coupling (TMR + HCONVI,
    /// Type395 inputs 12 & 13). When false, legacy decks still use the combined
    /// REQV-derived `heat_transfer_coeff_int` + TSTAR, and the interior Radiation
    /// boundary cell must be inert (h_rad = 0) so legacy runs stay byte-identical.
    pub has_separate_radiant_int: bool,

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
            temp_inside_air_moisture: celsius_to_kelvin(temp_inside_c),
            temp_outside_air_moisture: celsius_to_kelvin(temp_outside_c),
            temp_inside_mean_radiant: celsius_to_kelvin(temp_inside_c),
            heat_transfer_coeff_conv_int: heat_transfer_coeff_int * w_per_m2_k,
            additional_energy_ext: q_additional_out * watts,
            additional_energy_int: q_additional_in * watts,
            q_addition_userdef: q_additional_userdef * watts,
            is_raining: false,
            has_separate_radiant_int: false,
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
            pressure_air: ATMOSPHERE_PRESSURE * 1.0,
            heat_transfer_coeff_ext: 1.5 * w_per_m2_k,
            heat_transfer_coeff_int: 1.5 * w_per_m2_k,
            mass_convection_coeff_ext: 0.0 * m_per_s,
            mass_convection_coeff_int: 0.0 * m_per_s,
            vapor_transport_coeff_ext: Quantity(0.0),
            vapor_transport_coeff_int: Quantity(0.0),
            temp_inside_air_moisture: celsius_to_kelvin(20.0),
            temp_outside_air_moisture: celsius_to_kelvin(20.0),
            temp_inside_mean_radiant: celsius_to_kelvin(20.0),
            heat_transfer_coeff_conv_int: 1.5 * w_per_m2_k,
            additional_energy_ext: 0.0 * watts,
            additional_energy_int: 0.0 * watts,
            q_addition_userdef: 0.0 * watts,
            is_raining: false,
            has_separate_radiant_int: false,
        }
    }
}

impl BoundaryCondition {
    /// Linearly interpolate between two boundary conditions.
    ///
    /// `t = 0.0` returns `self`, `t = 1.0` returns `other`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let s = 1.0 - t;
        Self {
            temp_outside_air: self.temp_outside_air * s + other.temp_outside_air * t,
            temp_inside_air: self.temp_inside_air * s + other.temp_inside_air * t,
            rh_outside_air: self.rh_outside_air * s + other.rh_outside_air * t,
            rh_inside_air: self.rh_inside_air * s + other.rh_inside_air * t,
            pressure_air: self.pressure_air * s + other.pressure_air * t,
            heat_transfer_coeff_ext: self.heat_transfer_coeff_ext * s
                + other.heat_transfer_coeff_ext * t,
            heat_transfer_coeff_int: self.heat_transfer_coeff_int * s
                + other.heat_transfer_coeff_int * t,
            mass_convection_coeff_ext: self.mass_convection_coeff_ext * s
                + other.mass_convection_coeff_ext * t,
            mass_convection_coeff_int: self.mass_convection_coeff_int * s
                + other.mass_convection_coeff_int * t,
            vapor_transport_coeff_ext: self.vapor_transport_coeff_ext * s
                + other.vapor_transport_coeff_ext * t,
            vapor_transport_coeff_int: self.vapor_transport_coeff_int * s
                + other.vapor_transport_coeff_int * t,
            temp_inside_air_moisture: self.temp_inside_air_moisture * s
                + other.temp_inside_air_moisture * t,
            temp_outside_air_moisture: self.temp_outside_air_moisture * s
                + other.temp_outside_air_moisture * t,
            temp_inside_mean_radiant: self.temp_inside_mean_radiant * s
                + other.temp_inside_mean_radiant * t,
            heat_transfer_coeff_conv_int: self.heat_transfer_coeff_conv_int * s
                + other.heat_transfer_coeff_conv_int * t,
            additional_energy_ext: self.additional_energy_ext * s
                + other.additional_energy_ext * t,
            additional_energy_int: self.additional_energy_int * s
                + other.additional_energy_int * t,
            q_addition_userdef: self.q_addition_userdef * s + other.q_addition_userdef * t,
            // For booleans, use the end-of-timestep value (e.g. is_raining is an event flag).
            is_raining: other.is_raining,
            has_separate_radiant_int: other.has_separate_radiant_int,
        }
    }
}

/// Convert °C to K.
pub fn celsius_to_kelvin(temp_c: f64) -> Temperature<f64> {
    units::kelvin * (temp_c + KELVIN_CONV)
}

/// Convert K to °C.
pub fn kelvin_to_celsius(temp_k: Temperature<f64>) -> f64 {
    // strip: returns a raw Celsius scalar for external API boundaries
    // (TRNSYS, IDF, °C-based empirical formulas); stays untyped by design.
    temp_k.value_in(units::kelvin) - KELVIN_CONV
}
