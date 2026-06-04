use serde::Serialize;

use crate::{
    physics::{ KELVIN_CONV},
    uom::{
        Quantity,
        dimensions::{
            Dimensionless, HeatTransferCoeff, MassConvectionCoeff, Power, Pressure, Temperature, VaporTransportCoeff
        },
        units::{self, dimensionless, kg_per_m2_s_pa, kj_per_hr, kj_per_hr_m2_k, m_per_s, w_per_m2_k, watts},
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
    /// Construct a `BoundaryCondition` from the raw TRNSYS Type 395 input array.
    ///
    /// The `inputs` slice must contain between 11 and 14 `f64` values in the order
    /// defined by the Type 395 interface (see `trnsys_type.rs` docs).  Values for
    /// absent trailing inputs are derived via the same fallback rules that previously
    /// lived in `trnsys_type.rs::iterate`:
    ///
    /// | idx | field                       | unit in   | fallback (missing input) |
    /// |-----|-----------------------------|-----------|--------------------------|
    /// | 0   | temp_inside_air (TSTAR)     | °C        | —                        |
    /// | 1   | temp_outside_air            | °C        | —                        |
    /// | 2   | rh_inside_air               | % (0–100) | —                        |
    /// | 3   | rh_outside_air              | % (0–100) | —                        |
    /// | 4   | heat_transfer_coeff_int     | kJ/hr·m²·K| —                        |
    /// | 5   | heat_transfer_coeff_ext     | kJ/hr·m²·K| —                        |
    /// | 6   | additional_energy_int       | kJ/hr     | —                        |
    /// | 7   | additional_energy_ext       | kJ/hr     | —                        |
    /// | 8   | vapor_transport_coeff_int β | kg/m²·s·Pa| Lewis fallback if == 0   |
    /// | 9   | vapor_transport_coeff_ext β | kg/m²·s·Pa| —                        |
    /// | 10  | q_addition_userdef          | kJ/hr     | —                        |
    /// | 11  | temp_inside_air_moisture    | °C        | falls back to inputs[0]  |
    /// | 12  | heat_transfer_coeff_conv_int| kJ/hr·m²·K| falls back to inputs[4]  |
    /// | 13  | temp_inside_mean_radiant    | °C        | falls back to inputs[0]  |
    ///
    /// `has_separate_radiant_int` is set to `inputs.len() >= 14`.
    ///
    /// Lewis-analogy fallback: when `inputs[8] == 0.0` the interior vapor transport
    /// coefficient is derived as `7e-9 × heat_transfer_coeff_conv_int[W/m²K]`.
    pub fn from_trnsys_inputs(inputs: &[f64]) -> Self {
        // Always-present inputs (indices 0–10).
        let t_int   = inputs[0];  // TSTAR [°C]
        let t_out   = inputs[1];  // outside air T [°C]
        let rh_int  = inputs[2];  // inside RH [%]
        let rh_out  = inputs[3];  // outside RH [%]
        let hconv_int  = inputs[4]; // combined interior HTC [kJ/hr·m²·K]
        let hconv_out  = inputs[5]; // exterior HTC [kJ/hr·m²·K]
        let q_int   = inputs[6];  // interior additional energy [kJ/hr]
        let q_out   = inputs[7];  // exterior additional energy [kJ/hr]
        let beta_int = inputs[8]; // interior β [kg/m²·s·Pa]
        let beta_out = inputs[9]; // exterior β [kg/m²·s·Pa]
        let q_userdef = inputs[10]; // user-defined depth heat [kJ/hr]

        // Optional input 11: actual interior air T for moisture VP.
        // Falls back to TSTAR (input 0) when <12 inputs.
        let t_int_moisture = inputs.get(11).copied().unwrap_or(t_int);

        // Optional input 12: pure convective interior HTC (HCONVI).
        // Falls back to combined hconv_int (input 4) when <13 inputs.
        let hconv_conv_int = inputs.get(12).copied().unwrap_or(hconv_int);

        // Optional input 13: mean radiant temperature (TMR).
        // Falls back to TSTAR (input 0) when <14 inputs.
        let tmr_int = inputs.get(13).copied().unwrap_or(t_int);

        // Separated radiant path enabled only when both inputs 12 and 13 are present.
        let has_separate_radiant_int = inputs.len() >= 14;

        // Apply Lewis-analogy fallback: when beta_int == 0 derive from hconv_conv_int.
        // Uses 7e-9 [kg/(m²·s·Pa)] per [W/(m²·K)], the Lewis coefficient at ~20 °C.
        let vtc_int = if beta_int == 0.0 {
            // strip: empirical Lewis constant; hconv_conv_int is kJ/hr·m²·K -> /3.6 = W/m²·K
            let h_w = (hconv_conv_int * kj_per_hr_m2_k).value_in(w_per_m2_k);
            7.0e-9 * h_w * kg_per_m2_s_pa
        } else {
            beta_int * kg_per_m2_s_pa
        };

        Self {
            temp_inside_air: celsius_to_kelvin(t_int),
            temp_outside_air: celsius_to_kelvin(t_out),
            rh_inside_air: rh_int / 100.0 * dimensionless,
            rh_outside_air: rh_out / 100.0 * dimensionless,
            heat_transfer_coeff_int: hconv_int * kj_per_hr_m2_k,
            heat_transfer_coeff_ext: hconv_out * kj_per_hr_m2_k,
            additional_energy_int: q_int * kj_per_hr,
            additional_energy_ext: q_out * kj_per_hr,
            vapor_transport_coeff_int: vtc_int,
            vapor_transport_coeff_ext: beta_out * kg_per_m2_s_pa,
            q_addition_userdef: q_userdef * kj_per_hr,
            temp_inside_air_moisture: celsius_to_kelvin(t_int_moisture),
            temp_outside_air_moisture: celsius_to_kelvin(t_out),
            heat_transfer_coeff_conv_int: hconv_conv_int * kj_per_hr_m2_k,
            temp_inside_mean_radiant: celsius_to_kelvin(tmr_int),
            has_separate_radiant_int,
            // Fields not driven by inputs keep sensible defaults.
            pressure_air: ATMOSPHERE_PRESSURE * 1.0,
            mass_convection_coeff_ext: Quantity(0.0),
            mass_convection_coeff_int: Quantity(0.0),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uom::units::{kelvin, kg_per_m2_s_pa, w_per_m2_k, watts};

    // Conversion factor: 1 kJ/hr·m²·K = 1/3.6 W/m²·K
    const KJ_HR_M2K_TO_W_M2K: f64 = 1000.0 / 3600.0;
    // Conversion factor: 1 kJ/hr = 1000/3600 W
    const KJ_HR_TO_W: f64 = 1000.0 / 3600.0;
    const KELVIN_OFFSET: f64 = 273.15;

    /// Build a canonical 14-element input vector with distinct values for every field.
    fn make_14_inputs() -> [f64; 14] {
        [
            // idx  field                     value (unit)
            21.0,  // 0  TSTAR (inside air T)  [°C]
            -5.0,  // 1  T_out                 [°C]
            55.0,  // 2  RH_int                [%]
            80.0,  // 3  RH_out                [%]
            7.2,   // 4  hconv_int (combined)  [kJ/hr·m²·K]
            10.8,  // 5  hconv_out             [kJ/hr·m²·K]
            3.6,   // 6  q_int                 [kJ/hr]
            7.2,   // 7  q_out                 [kJ/hr]
            1.4e-8,// 8  beta_int              [kg/m²·s·Pa]
            2.8e-8,// 9  beta_out              [kg/m²·s·Pa]
            1.8,   // 10 q_userdef             [kJ/hr]
            19.0,  // 11 T_int_moisture (TAIR) [°C]
            5.4,   // 12 HCONVI (conv only)    [kJ/hr·m²·K]
            22.0,  // 13 TMR                   [°C]
        ]
    }

    /// 14-input deck: every field is directly set from its input, no fallbacks needed.
    #[test]
    fn test_from_trnsys_inputs_14_all_fields() {
        let inputs = make_14_inputs();
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs);

        // Temperatures → Kelvin
        let eps_k = 1e-9;
        assert!((bc.temp_inside_air.value_in(kelvin) - (21.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_air mismatch");
        assert!((bc.temp_outside_air.value_in(kelvin) - (-5.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_outside_air mismatch");
        assert!((bc.temp_inside_air_moisture.value_in(kelvin) - (19.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_air_moisture mismatch: expected input[11]=19.0 °C -> K");
        assert!((bc.temp_outside_air_moisture.value_in(kelvin) - (-5.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_outside_air_moisture = t_out");
        assert!((bc.temp_inside_mean_radiant.value_in(kelvin) - (22.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_mean_radiant mismatch: expected input[13]=22.0 °C -> K");

        // RH → fraction
        let eps_rh = 1e-12;
        assert!((bc.rh_inside_air.value() - 0.55).abs() < eps_rh,
            "rh_inside_air: expected 55/100 = 0.55, got {}", bc.rh_inside_air.value());
        assert!((bc.rh_outside_air.value() - 0.80).abs() < eps_rh,
            "rh_outside_air: expected 80/100 = 0.80, got {}", bc.rh_outside_air.value());

        // HTC → W/m²·K
        let eps_h = 1e-10;
        let expected_hconv_int = 7.2 * KJ_HR_M2K_TO_W_M2K;
        let expected_hconv_ext = 10.8 * KJ_HR_M2K_TO_W_M2K;
        let expected_hconv_conv_int = 5.4 * KJ_HR_M2K_TO_W_M2K;
        assert!((bc.heat_transfer_coeff_int.value_in(w_per_m2_k) - expected_hconv_int).abs() < eps_h,
            "heat_transfer_coeff_int");
        assert!((bc.heat_transfer_coeff_ext.value_in(w_per_m2_k) - expected_hconv_ext).abs() < eps_h,
            "heat_transfer_coeff_ext");
        assert!((bc.heat_transfer_coeff_conv_int.value_in(w_per_m2_k) - expected_hconv_conv_int).abs() < eps_h,
            "heat_transfer_coeff_conv_int: expected input[12]=5.4 kJ/hr·m²·K");

        // Beta → kg/m²·s·Pa  (non-zero so Lewis fallback must NOT fire)
        let eps_beta = 1e-20;
        assert!((bc.vapor_transport_coeff_int.value_in(kg_per_m2_s_pa) - 1.4e-8).abs() < eps_beta,
            "vapor_transport_coeff_int");
        assert!((bc.vapor_transport_coeff_ext.value_in(kg_per_m2_s_pa) - 2.8e-8).abs() < eps_beta,
            "vapor_transport_coeff_ext");

        // Energies → W  (1 kJ/hr = 1000/3600 W)
        let eps_w = 1e-10;
        assert!((bc.additional_energy_int.value_in(watts) - 3.6 * KJ_HR_TO_W).abs() < eps_w,
            "additional_energy_int");
        assert!((bc.additional_energy_ext.value_in(watts) - 7.2 * KJ_HR_TO_W).abs() < eps_w,
            "additional_energy_ext");
        assert!((bc.q_addition_userdef.value_in(watts) - 1.8 * KJ_HR_TO_W).abs() < eps_w,
            "q_addition_userdef");

        // Radiant flag
        assert!(bc.has_separate_radiant_int, "has_separate_radiant_int must be true for 14 inputs");

        // Defaults preserved
        assert!(!bc.is_raining, "is_raining default is false");
    }

    /// 11-input deck: inputs 11/12/13 absent, fallback rules apply.
    /// - has_separate_radiant_int = false
    /// - t_int_moisture = t_int (TSTAR)
    /// - hconv_conv_int = hconv_int (combined)
    /// - tmr_int = t_int (TSTAR)
    /// - Lewis fallback fires because beta_int == 0
    #[test]
    fn test_from_trnsys_inputs_11_fallbacks_and_lewis() {
        // Use same first 11 inputs, beta_int = 0 to trigger Lewis fallback.
        let inputs: [f64; 11] = [
            21.0,   // 0  TSTAR [°C]
            -5.0,   // 1  T_out [°C]
            55.0,   // 2  RH_int [%]
            80.0,   // 3  RH_out [%]
            7.2,    // 4  hconv_int (combined) [kJ/hr·m²·K]
            10.8,   // 5  hconv_out [kJ/hr·m²·K]
            3.6,    // 6  q_int [kJ/hr]
            7.2,    // 7  q_out [kJ/hr]
            0.0,    // 8  beta_int = 0 → Lewis fallback fires
            2.8e-8, // 9  beta_out [kg/m²·s·Pa]
            1.8,    // 10 q_userdef [kJ/hr]
        ];
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs);

        // Radiant flag must be false for 11-input deck.
        assert!(!bc.has_separate_radiant_int, "has_separate_radiant_int must be false for 11 inputs");

        let eps_k = 1e-9;
        // t_int_moisture falls back to TSTAR (inputs[0] = 21 °C)
        assert!((bc.temp_inside_air_moisture.value_in(kelvin) - (21.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_air_moisture must fall back to t_int (TSTAR = 21 °C)");
        // tmr_int falls back to TSTAR (inputs[0] = 21 °C)
        assert!((bc.temp_inside_mean_radiant.value_in(kelvin) - (21.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_mean_radiant must fall back to t_int (TSTAR = 21 °C)");

        // hconv_conv_int falls back to combined hconv_int (inputs[4] = 7.2 kJ/hr·m²·K)
        let eps_h = 1e-10;
        let expected_hconv_conv_int = 7.2 * KJ_HR_M2K_TO_W_M2K;
        assert!((bc.heat_transfer_coeff_conv_int.value_in(w_per_m2_k) - expected_hconv_conv_int).abs() < eps_h,
            "heat_transfer_coeff_conv_int must fall back to hconv_int=7.2 kJ/hr·m²·K");

        // Lewis fallback: beta_int==0 → vtc_int = 7e-9 * hconv_conv_int[W/m²K]
        // hconv_conv_int = 7.2 kJ/hr·m²·K = 7.2/3.6 W/m²K = 2.0 W/m²K
        let h_w = 7.2 * KJ_HR_M2K_TO_W_M2K; // = 2.0 W/m²K
        let expected_vtc_int = 7.0e-9 * h_w;
        let eps_beta = 1e-20;
        assert!((bc.vapor_transport_coeff_int.value_in(kg_per_m2_s_pa) - expected_vtc_int).abs() < eps_beta,
            "Lewis fallback: vtc_int = 7e-9 * {h_w} = {expected_vtc_int}, got {}",
            bc.vapor_transport_coeff_int.value_in(kg_per_m2_s_pa));
    }

    /// 12-input deck: input 11 present, inputs 12/13 absent.
    /// - t_int_moisture = inputs[11]
    /// - hconv_conv_int falls back to hconv_int (inputs[4])
    /// - tmr_int falls back to t_int (inputs[0])
    /// - has_separate_radiant_int = false
    #[test]
    fn test_from_trnsys_inputs_12_partial_fallback() {
        let inputs: [f64; 12] = [
            21.0,  // 0  TSTAR
            -5.0,  // 1  T_out
            55.0,  // 2  RH_int [%]
            80.0,  // 3  RH_out [%]
            7.2,   // 4  hconv_int
            10.8,  // 5  hconv_out
            3.6,   // 6  q_int
            7.2,   // 7  q_out
            1.4e-8,// 8  beta_int (non-zero → no Lewis fallback)
            2.8e-8,// 9  beta_out
            1.8,   // 10 q_userdef
            19.0,  // 11 T_int_moisture (TAIR)
        ];
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs);

        assert!(!bc.has_separate_radiant_int, "12 inputs → no separate radiant");
        let eps_k = 1e-9;
        // t_int_moisture comes from inputs[11] = 19 °C
        assert!((bc.temp_inside_air_moisture.value_in(kelvin) - (19.0 + KELVIN_OFFSET)).abs() < eps_k,
            "temp_inside_air_moisture = input[11] = 19 °C");
        // hconv_conv_int falls back to inputs[4] = 7.2
        let eps_h = 1e-10;
        assert!((bc.heat_transfer_coeff_conv_int.value_in(w_per_m2_k) - 7.2 * KJ_HR_M2K_TO_W_M2K).abs() < eps_h,
            "hconv_conv_int falls back to inputs[4]");
        // tmr falls back to TSTAR = 21 °C
        assert!((bc.temp_inside_mean_radiant.value_in(kelvin) - (21.0 + KELVIN_OFFSET)).abs() < eps_k,
            "tmr falls back to TSTAR = 21 °C");
    }

    /// 13-input deck: inputs 11 and 12 present, input 13 absent.
    /// - t_int_moisture = inputs[11]
    /// - hconv_conv_int = inputs[12]
    /// - tmr_int falls back to TSTAR (inputs[0])
    /// - has_separate_radiant_int = false
    #[test]
    fn test_from_trnsys_inputs_13_tmr_fallback() {
        let inputs: [f64; 13] = [
            21.0,  // 0  TSTAR
            -5.0,  // 1  T_out
            55.0,  // 2  RH_int [%]
            80.0,  // 3  RH_out [%]
            7.2,   // 4  hconv_int
            10.8,  // 5  hconv_out
            3.6,   // 6  q_int
            7.2,   // 7  q_out
            1.4e-8,// 8  beta_int
            2.8e-8,// 9  beta_out
            1.8,   // 10 q_userdef
            19.0,  // 11 T_int_moisture
            5.4,   // 12 HCONVI
        ];
        let bc = BoundaryCondition::from_trnsys_inputs(&inputs);

        assert!(!bc.has_separate_radiant_int, "13 inputs → no separate radiant");
        let eps_k = 1e-9;
        let eps_h = 1e-10;
        // inputs[12] is present → hconv_conv_int = 5.4 kJ/hr·m²·K
        assert!((bc.heat_transfer_coeff_conv_int.value_in(w_per_m2_k) - 5.4 * KJ_HR_M2K_TO_W_M2K).abs() < eps_h,
            "hconv_conv_int = input[12] = 5.4");
        // inputs[13] absent → tmr falls back to TSTAR = 21 °C
        assert!((bc.temp_inside_mean_radiant.value_in(kelvin) - (21.0 + KELVIN_OFFSET)).abs() < eps_k,
            "tmr falls back to TSTAR for 13-input deck");
    }
}
