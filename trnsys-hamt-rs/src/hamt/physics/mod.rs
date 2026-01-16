//! Contains pure physics and psychrometric functions for the HAMT model.

use crate::hamt::material::Curve;
pub mod geometry;

/// Universal gas constant for water vapor [J/kg·K].
pub const R_V: f64 = 461.52;
/// Kelvin conversion constant.
pub const KELVIN_CONV: f64 = 273.15;
/// Specific heat of water [J/kg·K].
pub const W_SPECH: f64 = 4180.0;
/// Evaporation enthalpy of water [J/kg].
pub const W_HV: f64 = 2_489_000.0;
/// Density of water [kg/m³].
pub const W_DENSITY: f64 = 1000.0;

/// Linearly interpolates a value from a curve.
/// Also optionally returns the gradient at the interpolation point.
pub fn interp(curve: &Curve, in_value: f64) -> (f64, f64) {
    if curve.x.is_empty() {
        return (0.0, 0.0);
    }

    if in_value.is_nan() {
        return (curve.y[0], 0.0);
    }

    let mut upper_index: Option<usize> = None;
    for (i, &xv) in curve.x.iter().enumerate() {
        match xv.partial_cmp(&in_value) {
            Some(std::cmp::Ordering::Equal) => return (curve.y[i], 0.0),
            Some(std::cmp::Ordering::Greater) => {
                upper_index = Some(i);
                break;
            }
            _ => {}
        }
    }

    match upper_index {
        Some(0) => (curve.y[0], 0.0),
        Some(i) => {
            let x1 = curve.x[i - 1];
            let x2 = curve.x[i];
            let y1 = curve.y[i - 1];
            let y2 = curve.y[i];

            let grad = if (x2 - x1).abs() > 1e-9 {
                (y2 - y1) / (x2 - x1)
            } else {
                0.0
            };
            let val = y1 + grad * (in_value - x1);
            (val, grad)
        }
        None => (*curve.y.last().unwrap(), 0.0),
    }
}

/// Calculates saturation vapor pressure as a function of temperature.
/// Implements the Hyland & Wexler formulation as seen in EnergyPlus.
///
/// # Arguments
/// * `t_celsius` - Dry-bulb temperature [°C].
pub fn psy_psat_fn_temp(t_celsius: f64) -> f64 {
    let t_kelvin = t_celsius + KELVIN_CONV;

    if t_kelvin < KELVIN_CONV {
        // Below freezing: saturation pressure over ice
        const C1: f64 = -5674.5359;
        const C2: f64 = 6.3925247;
        const C3: f64 = -0.9677843e-2;
        const C4: f64 = 0.62215701e-6;
        const C5: f64 = 0.20747825e-8;
        const C6: f64 = -0.9484024e-12;
        const C7: f64 = 4.1635019;
        (C1 / t_kelvin
            + C2
            + t_kelvin * (C3 + t_kelvin * (C4 + t_kelvin * (C5 + C6 * t_kelvin)))
            + C7 * t_kelvin.ln())
        .exp()
    } else {
        // Above freezing: saturation pressure over water
        const C8: f64 = -5800.2206;
        const C9: f64 = 1.3914993;
        const C10: f64 = -0.048640239;
        const C11: f64 = 0.41764768e-4;
        const C12: f64 = -0.14452093e-7;
        const C13: f64 = 6.5459673;
        (C8 / t_kelvin
            + C9
            + t_kelvin * (C10 + t_kelvin * (C11 + t_kelvin * C12))
            + C13 * t_kelvin.ln())
        .exp()
    }
}

/// Converts Relative Humidity and Temperature to Vapor Pressure.
///
/// # Arguments
/// * `rh` - Relative humidity [0-1].
/// * `temp_celsius` - Temperature [°C].
pub fn rh_to_vp(rh: f64, temp_celsius: f64) -> f64 {
    rh * psy_psat_fn_temp(temp_celsius)
}

/// Calculates the Water Vapor Diffusion Coefficient in air.
///
/// # Arguments
/// * `temp_celsius` - Temperature [°C].
/// * `amb_pressure` - Ambient barometric pressure [Pa].
pub fn wvdc(temp_celsius: f64, amb_pressure: f64) -> f64 {
    (2e-7 * (temp_celsius + KELVIN_CONV).powf(0.81)) / amb_pressure
}

/// Calculates Relative Humidity from dry-bulb temperature and vapor density.
///
/// # Arguments
/// * `t_db_celsius` - Dry-bulb temperature [°C].
/// * `rho_vapor` - Vapor density in air [kg/m³].
pub fn psy_rh_fn_tdb_rhov(t_db_celsius: f64, rho_vapor: f64) -> f64 {
    if rho_vapor <= 0.0 {
        return 0.0;
    }
    let p_sat = psy_psat_fn_temp(t_db_celsius);
    let rh = rho_vapor * R_V * (t_db_celsius + KELVIN_CONV) / p_sat;

    // Clamp the value between 0 and 1, as small inaccuracies can push it slightly out of bounds.
    rh.max(0.0).min(1.0)
}

/// Calculates Vapor Density from dry-bulb temperature and relative humidity.
///
/// # Arguments
/// * `t_db_celsius` - Dry-bulb temperature [°C].
/// * `rh` - Relative humidity [0-1].
pub fn psy_rhov_fn_tdb_rh(t_db_celsius: f64, rh: f64) -> f64 {
    let p_sat = psy_psat_fn_temp(t_db_celsius);
    (p_sat * rh) / (R_V * (t_db_celsius + KELVIN_CONV))
}
