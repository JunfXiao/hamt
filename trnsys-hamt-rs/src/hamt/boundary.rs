use crate::hamt::uom::{
    PressureQty, Quantity, TemperatureQty,
    dimensions::{Dimensionless, Temperature},
    units,
};

/// Typed boundary conditions for the simulation.
/// Temperatures are stored as absolute (Kelvin). Relative humidity is
/// dimensionless (0–1). Pressure is absolute (Pascal).
#[derive(Debug, Clone)]
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

    // Placeholder for future boundary conditions
    pub is_raining: bool,
}

impl BoundaryCondition {
    /// Create a new typed boundary condition from raw inputs.
    ///
    /// `temp_*_c` are in °C, `rh_*` are fractions (0–1), `pressure_pa` in Pa.
    pub fn from_raw(
        temp_outside_c: f64,
        temp_inside_c: f64,
        rh_outside: f64,
        rh_inside: f64,
        pressure_pa: f64,
    ) -> Self {
        Self {
            temp_outside_air: celsius_to_kelvin(temp_outside_c),
            temp_inside_air: celsius_to_kelvin(temp_inside_c),
            rh_outside_air: Quantity(rh_outside),
            rh_inside_air: Quantity(rh_inside),
            pressure_air: pressure_pa * units::pascals,
            is_raining: false,
        }
    }

    /// Returns outside temperature in °C for legacy IO.
    pub fn temp_outside_c(&self) -> f64 {
        kelvin_to_celsius(self.temp_outside_air)
    }

    /// Returns inside temperature in °C for legacy IO.
    pub fn temp_inside_c(&self) -> f64 {
        kelvin_to_celsius(self.temp_inside_air)
    }

    /// Returns pressure in Pa for legacy IO.
    pub fn pressure_pa(&self) -> f64 {
        self.pressure_air.value_in(units::pascals)
    }
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self::from_raw(20.0, 20.0, 0.5, 0.5, 101_325.0)
    }
}

/// Convert °C to K.
fn celsius_to_kelvin(temp_c: f64) -> TemperatureQty<f64> {
    units::kelvin * (temp_c + 273.15)
}

/// Convert K to °C.
fn kelvin_to_celsius(temp_k: TemperatureQty<f64>) -> f64 {
    temp_k.value_in(units::kelvin) - 273.15
}
