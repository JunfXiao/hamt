//! HAMT unit system built with `diman`.
//!
//! This defines the base and derived dimensions/units commonly needed by the
//! hygrothermal solver, plus a few domain-specific constants in SI.

use diman::unit_system;

unit_system!(
    quantity_type Quantity;
    dimension_type Dimension;

    // Base dimensions
    dimension Dimensionless = 1.0f64;
    dimension Length;
    dimension Time;
    dimension Mass;
    dimension Temperature;

    // Derived dimensions
    dimension Area = Length^2;
    dimension Volume = Length^3;
    dimension Velocity = Length / Time;
    dimension Acceleration = Velocity / Time;
    dimension Pressure = Mass / (Length * Time^2);
    dimension Energy = Pressure * Volume;
    dimension Power = Energy / Time;
    dimension HeatCapacity = Energy / Temperature; // J/K
    dimension SpecificEnergy = Energy / Mass; // J/kg
    dimension SpecificHeatCapacity = HeatCapacity / Mass; // J/(kg·K)
    dimension VolumetricHeatCapacity = SpecificHeatCapacity * Density; // J/(m³·K)
    dimension EnergyFluxDensity = Power / Area; // W/m²
    dimension MassFluxDensity = Mass / Area / Time; // kg/m²·s
    dimension MassFlux = Mass / Time;
    dimension HeatTransferCoeff = Power / (Area * Temperature); // W/(m²·K)
    dimension ThermalConductivity = Power / (Length * Temperature); // W/(m·K)
    dimension ThermalConductance = Power / Temperature;
    dimension VaporTransportCoeff = Mass / (Area * Time * Pressure); // kg/(m²·s·Pa)
    dimension MassConvectionCoeff = Length /  Time; // m/s
    dimension Density = Mass / Volume; // kg/m³
    dimension WaterContent = Mass / Volume; // kg/kg
    dimension LiquidTransportCoeff = Length^2 / Time; // m²/s
    dimension VaporPermeability = Mass/ (Length * Time * Pressure); // m²/s

    dimension ThermalResistance =Temperature / Power; // K/W
    // Vapor resistance (Pa·s/kg), consistent with WVDC-based formulation
    dimension VaporResistance = Pressure * Time / Mass;
    dimension VaporConductance = Dimensionless / VaporResistance;
    dimension MassResistance = Dimensionless / MassFlux;

    // Base units
    #[base(Length)]
    #[symbol(m)]
    unit meter;

    #[base(Time)]
    #[symbol(s)]
    unit seconds;


    #[base(Mass)]
    #[symbol(kg)]
    unit kilograms;

    #[base(Temperature)]
    #[symbol(K)]
    unit kelvin;

    // Derived units
    unit dimensionless: Dimensionless = 1.0;
    unit square_meter: Area = meter^2;
    unit cubic_meter: Volume = meter^3;
    unit m_per_s: Velocity = meter / seconds;
    unit m_per_s_squared: Acceleration = meter / seconds^2;
    unit hours: Time = 3600.0f64 * seconds;
    unit days: Time = 24.0f64 * hours;
    unit pascals: Pressure = kilograms / (meter * seconds^2);
    unit joules: Energy = pascals * cubic_meter;
    unit kj: Energy = 1000.0f64 * joules;
    unit watts: Power = joules / seconds;
    unit kj_per_hr: Power = kj / hours;
    unit j_per_k: HeatCapacity = joules / kelvin;
    unit j_per_kg: SpecificEnergy = joules / kilograms;
    unit j_per_kg_k: SpecificHeatCapacity = j_per_k / kilograms;
    unit j_per_m3_k: VolumetricHeatCapacity = j_per_kg_k * kg_per_m3;
    unit w_per_m2: EnergyFluxDensity = watts / square_meter;
    unit kg_per_m2_s: MassFluxDensity = kilograms / (square_meter * seconds);
    unit kg_per_s: MassFlux = kilograms / seconds;
    unit w_per_m2_k: HeatTransferCoeff = watts / (square_meter * kelvin);
    unit w_per_k: ThermalConductance = watts / kelvin;
    unit kj_per_hr_m2_k: HeatTransferCoeff = kj_per_hr / (square_meter * kelvin);
    unit w_per_m_k: ThermalConductivity = watts / (meter * kelvin);
    unit kg_per_m3: Density = kilograms / cubic_meter;
    unit k_per_w: ThermalResistance =  kelvin / watts;
    unit kg_per_m2_s_pa: VaporTransportCoeff =
        kilograms / (square_meter * seconds * pascals);
    unit m2_per_s: LiquidTransportCoeff = square_meter / seconds;
    unit kg_per_m_s_pa: VaporPermeability = kilograms / (meter * seconds * pascals);
    // Vapor resistance: Pa·s/kg
    unit pa_s_per_kg: VaporResistance = pascals * seconds / kilograms;
    unit s_per_kg: MassResistance = seconds / kilograms;

    // Domain constants (SI)
    constant WATER_DENSITY = 1000.0f64 * kg_per_m3;
    constant WATER_SPECIFIC_HEAT = 4180.0f64 * j_per_kg_k;
    constant WATER_LATENT_HEAT = 2_489_000.0f64 * j_per_kg;
    constant VAPOR_GAS_CONSTANT = 461.524f64 * j_per_kg_k;
);

// Convenience type aliases for external modules
pub type HamtQuantity<S, const D: Dimension> = Quantity<S, D>;
pub type HamtDimension = Dimension;


// Dimension constants for const-generic quantities
//
pub const DIM_DIMENSIONLESS: Dimension = Dimension {
    length: 0,
    time: 0,
    mass: 0,
    temperature: 0,
};
pub const DIM_TIME: Dimension = Dimension {
    length: 0,
    time: 1,
    mass: 0,
    temperature: 0,
};
pub const DIM_TEMPERATURE: Dimension = Dimension {
    length: 0,
    time: 0,
    mass: 0,
    temperature: 1,
};
pub const DIM_PRESSURE: Dimension = Dimension {
    length: -1,
    time: -2,
    mass: 1,
    temperature: 0,
};
pub const DIM_AREA: Dimension = Dimension {
    length: 2,
    time: 0,
    mass: 0,
    temperature: 0,
};
pub const DIM_VOLUME: Dimension = Dimension {
    length: 3,
    time: 0,
    mass: 0,
    temperature: 0,
};
pub const DIM_POWER: Dimension = Dimension {
    length: 2,
    time: -3,
    mass: 1,
    temperature: 0,
};
pub const DIM_DENSITY: Dimension = Dimension {
    length: -3,
    time: 0,
    mass: 1,
    temperature: 0,
};
pub const DIM_WATER_CONTENT: Dimension = Dimension {
    length: -3,
    time: 0,
    mass: 1,
    temperature: 0,
};
pub const DIM_SPECIFIC_HEAT_CAPACITY: Dimension = Dimension {
    length: 2,
    time: -2,
    mass: 0,
    temperature: -1,
};
pub const DIM_THERMAL_CONDUCTIVITY: Dimension = Dimension {
    length: 1,
    time: -3,
    mass: 1,
    temperature: -1,
};

pub const DIM_HEAT_TRANSFER_COEFF: Dimension = Dimension {
    length: 0,
    time: -1,
    mass: 0,
    temperature: -1,
};

pub const DIM_LIQUID_TRANSPORT_COEFF: Dimension = Dimension {
    length: 2,
    time: -1,
    mass: 0,
    temperature: 0,
};

pub const DIM_VAPOR_TRANSPORT_COEFF: Dimension = Dimension {
    length: -2,
    time: -1,
    mass: 1,
    temperature: 0,
};


#[cfg(test)]
mod tests {
    use super::*;
    use crate::uom::dimensions::{HeatTransferCoeff, Power, VaporTransportCoeff};
    use crate::uom::units::{
        dimensionless, kg_per_m2_s_pa, kg_per_m3, kj_per_hr, kj_per_hr_m2_k, m_per_s, pascals,
        w_per_m2_k, watts,
    };

    // Verify that converting from mass convection (m/s) and RH/Psat yields
    // the expected vapor transport coefficient units/value.
    #[test]
    fn vtc_from_mass_convection_has_correct_units() {
        let hmass = 1.0 * m_per_s; // m/s
        let rho_v = 1.2 * kg_per_m3; // kg/m3
        let psat = 2000.0 * pascals; // Pa
        let rh = 0.5 * dimensionless; // [-]

        let vtc: VaporTransportCoeff<f64> = hmass * rho_v / (psat * rh); // should be kg/(m2·s·Pa)
        let expected = 1.0 * 1.2 / (2000.0 * 0.5); // numeric value

        assert!(
            (vtc.value_in(kg_per_m2_s_pa) - expected).abs() < 1e-12,
            "vtc value mismatch: got {}, expected {}",
            vtc.value_in(kg_per_m2_s_pa),
            expected
        );
    }

    // Verify that 1 kJ/(hr·m²·K) converts to the correct W/(m²·K).
    #[test]
    fn heat_transfer_coeff_from_kj_per_hr() {
        let h_kj: HeatTransferCoeff<f64> = 1.0 * kj_per_hr_m2_k;
        let expected = 1000.0 / 3600.0; // W/m2K
        assert!(
            (h_kj.value_in(w_per_m2_k) - expected).abs() < 1e-12,
            "htc value mismatch: got {}, expected {}",
            h_kj.value_in(w_per_m2_k),
            expected
        );
    }

    // Verify that 1 kJ/hr converts to the correct W.
    #[test]
    fn power_from_kj_per_hr() {
        let p_kj_hr: Power<f64> = 1.0 * kj_per_hr;
        let expected = 1000.0 / 3600.0; // W
        assert!(
            (p_kj_hr.value_in(watts) - expected).abs() < 1e-12,
            "power value mismatch: got {}, expected {}",
            p_kj_hr.value_in(watts),
            expected
        );
    }
}
