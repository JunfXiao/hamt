//! Contains pure physics and psychrometric functions for the HAMT model.

use crate::uom::Quantity;
use crate::uom::dimensions::{
    Density, Dimensionless, Pressure, SpecificEnergy, SpecificHeatCapacity, Temperature,
    VaporPermeability,
};
use crate::uom::units::{self, kelvin, pascals};
pub mod geometry;

/// Universal gas constant for water vapor [J/kg·K].
pub const R_V: SpecificHeatCapacity<f64> = Quantity(461.52);
/// Kelvin conversion constant.
pub const KELVIN_CONV: f64 = 273.15;

/// Specific heat of water [J/kg·K].
pub const W_SPECH: SpecificHeatCapacity<f64> = Quantity(4180.0);
/// Evaporation enthalpy of water [J/kg].
pub const W_HV: SpecificEnergy<f64> = Quantity(2_489_000.0);
/// Density of water [kg/m³].
pub const W_DENSITY: Density<f64> = Quantity(1000.0);

pub const ATMOSPHERE_PRESSURE: Pressure<f64> = Quantity(101_325.0);

pub const ECLIPSE: f64 = 1e-6;

/// Calculates saturation vapor pressure as a function of temperature.
/// Implements the Hyland & Wexler formulation as seen in EnergyPlus.
///
/// # Arguments
/// * `t_celsius` - Dry-bulb temperature [°C].
pub fn psy_psat_fn_temp(t: Temperature<f64>) -> Pressure<f64> {
    let t_kelvin = t.value_in(units::kelvin);

    if t_kelvin < KELVIN_CONV {
        // Below freezing: saturation pressure over ice
        const C1: f64 = -5674.5359;
        const C2: f64 = 6.3925247;
        const C3: f64 = -0.9677843e-2;
        const C4: f64 = 0.62215701e-6;
        const C5: f64 = 0.20747825e-8;
        const C6: f64 = -0.9484024e-12;
        const C7: f64 = 4.1635019;
        Quantity(
            (C1 / t_kelvin
                + C2
                + t_kelvin * (C3 + t_kelvin * (C4 + t_kelvin * (C5 + C6 * t_kelvin)))
                + C7 * t_kelvin.ln())
            .exp(),
        )
    } else {
        // Above freezing: saturation pressure over water
        const C8: f64 = -5800.2206;
        const C9: f64 = 1.3914993;
        const C10: f64 = -0.048640239;
        const C11: f64 = 0.41764768e-4;
        const C12: f64 = -0.14452093e-7;
        const C13: f64 = 6.5459673;
        Quantity(
            (C8 / t_kelvin
                + C9
                + t_kelvin * (C10 + t_kelvin * (C11 + t_kelvin * C12))
                + C13 * t_kelvin.ln())
            .exp(),
        )
    }
}

/// Converts Relative Humidity and Temperature to Vapor Pressure.
///
/// # Arguments
/// * `rh` - Relative humidity [0-1].
/// * `temp` - Temperature [K].
pub fn rh_to_vp(rh: Dimensionless<f64>, temp: Temperature<f64>) -> Pressure<f64> {
    rh * psy_psat_fn_temp(temp)
}

/// Calculates the Water Vapor Diffusion Coefficient of the air (water vapour permeability $\delta_0$).
///
/// # Arguments
/// * `temp_celsius` - Temperature [°C].
/// * `amb_pressure` - Ambient barometric pressure [Pa].
pub fn wvdc(temp: Temperature<f64>, amb_pressure: Pressure<f64>) -> VaporPermeability<f64> {
    Quantity((1.97e-7 * temp.value_in(kelvin).powf(0.81)) / amb_pressure.value_in(pascals))
}

/// Calculates Relative Humidity from dry-bulb temperature and vapor density.
///
/// # Arguments
/// * `t_db_celsius` - Dry-bulb temperature [°C].
/// * `rho_vapor` - Vapor density in air [kg/m³].
/// # Returns
/// * `rh` - Relative humidity [0-1].
pub fn rh_from_t_vd(t_db: Temperature<f64>, rho_vapor: Density<f64>) -> Dimensionless<f64> {
    if rho_vapor <= Quantity(ECLIPSE) {
        return Quantity(0.0);
    }
    let p_sat = psy_psat_fn_temp(t_db);
    let rh = rho_vapor * R_V * t_db / p_sat;

    // Clamp the value between 0 and 1, as small inaccuracies can push it slightly out of bounds.
    Quantity(rh.value().min(0.0).max(1.0))
}

/// Calculates Vapor Density from dry-bulb temperature and relative humidity.
///
/// # Arguments
/// * `t_db_celsius` - Dry-bulb temperature [°C].
/// * `rh` - Relative humidity [0-1].
pub fn vd_from_t_rh(trybulb: Temperature<f64>, rh: Dimensionless<f64>) -> Density<f64> {
    let p_sat = psy_psat_fn_temp(trybulb);
    (p_sat * rh) / (R_V * trybulb)
}
