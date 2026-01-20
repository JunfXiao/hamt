//! HAMT unit system built with `diman`.
//!
//! This defines the base and derived dimensions/units commonly needed by the
//! hygrothermal solver, plus a few domain-specific constants in SI.

use std::ops::Div;

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
    dimension EnergyFlux = Power / Area; // W/m²
    dimension MassFlux = Mass / Area / Time; // kg/m²·s
    dimension HeatTransferCoeff = Power / (Area * Temperature); // W/(m²·K)
    dimension ThermalConductivity = Power / (Length * Temperature); // W/(m·K)
    dimension LiquidTransferCoeff = Length^2 / Time;
    dimension VaporTransportCoeff = Mass / (Area * Time * Pressure); // kg/(m²·s·Pa)
    dimension MassConvectionCoeff = Mass / (Area * Time * Pressure); // kg/(m²·s·Pa)
    dimension Density = Mass / Volume; // kg/m³
    dimension WaterContent = Dimensionless; // kg/kg
    dimension LiquidTransportCoeff = Length^2 / Time; // m²/s
    dimension VaporPermeability = Mass/ (Length * Time * Pressure); // m²/s

    dimension ThermalResistance = Dimensionless / Length / ThermalConductivity; // K/W
    // Vapor resistance (Pa·s/kg), consistent with WVDC-based formulation
    dimension VaporResistance = Pressure * Time / Mass;

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
    unit meters_per_second: Velocity = meter / seconds;
    unit meters_per_second_squared: Acceleration = meter / seconds^2;
    unit hours: Time = 3600.0f64 * seconds;
    unit pascals: Pressure = kilograms / (meter * seconds^2);
    unit joules: Energy = pascals * cubic_meter;
    unit kj: Energy = 1000.0f64 * joules;
    unit watts: Power = joules / seconds;
    unit kj_per_hr: Power = kj / hours;
    unit j_per_k: HeatCapacity = joules / kelvin;
    unit j_per_kg: SpecificEnergy = joules / kilograms;
    unit j_per_kg_k: SpecificHeatCapacity = j_per_k / kilograms;
    unit w_per_m2: EnergyFlux = watts / square_meter;
    unit kg_per_m2_s: MassFlux = kilograms / (square_meter * seconds);
    unit w_per_m2_k: HeatTransferCoeff = watts / (square_meter * kelvin);
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

    // Domain constants (SI)
    constant WATER_DENSITY = 1000.0f64 * kg_per_m3;
    constant WATER_SPECIFIC_HEAT = 4180.0f64 * j_per_kg_k;
    constant WATER_LATENT_HEAT = 2_489_000.0f64 * j_per_kg;
    constant VAPOR_GAS_CONSTANT = 461.52f64 * j_per_kg_k;
);

// Convenience type aliases for external modules
pub type HamtQuantity<S, const D: Dimension> = Quantity<S, D>;
pub type HamtDimension = Dimension;

// Helpful quantity aliases using the dimension consts
pub type DimensionlessQty<S> = Quantity<S, { DIM_DIMENSIONLESS }>;
pub type TimeQty<S> = Quantity<S, { DIM_TIME }>;
pub type Temperature<S> = Quantity<S, { DIM_TEMPERATURE }>;
pub type PressureQty<S> = Quantity<S, { DIM_PRESSURE }>;
pub type AreaQty<S> = Quantity<S, { DIM_AREA }>;
pub type VolumeQty<S> = Quantity<S, { DIM_VOLUME }>;
pub type PowerQty<S> = Quantity<S, { DIM_POWER }>;
pub type DensityQty<S> = Quantity<S, { DIM_DENSITY }>;
pub type WaterContentQty<S> = Quantity<S, { DIM_WATER_CONTENT }>;
pub type SpecificHeatCapacityQty<S> = Quantity<S, { DIM_SPECIFIC_HEAT_CAPACITY }>;
pub type ThermalConductivityQty<S> = Quantity<S, { DIM_THERMAL_CONDUCTIVITY }>;
pub type HeatTransportCoeffQty<S> = Quantity<S, { DIM_HEAT_TRANSFER_COEFF }>;
pub type LiquidTransportCoeffQty<S> = Quantity<S, { DIM_LIQUID_TRANSPORT_COEFF }>;
pub type VaporTransportCoeffQty<S> = Quantity<S, { DIM_VAPOR_TRANSPORT_COEFF }>;

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
