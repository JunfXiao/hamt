//! Defines the data structures for materials and constructions used in the HAMT model.

/// Represents a 2D data curve, typically from material property measurements.
///
/// This is used to store properties that vary with a single independent variable,
/// such as the sorption isotherm (water content vs. RH) or thermal conductivity
/// vs. water content. The `interp` function in the `physics` module will operate
/// on this struct.
#[derive(Debug, Clone)]
pub struct Curve {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Curve {
    /// Creates an empty curve.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
        }
    }
}

/// Holds all the hygrothermal properties of a single material.
///
/// This struct corresponds to the `MaterialProperties` in the Fortran implementation
/// and the `MaterialHAMT` class in the EnergyPlus C++ code.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    // Basic thermal properties
    pub thickness: f64,
    pub conductivity: f64,
    pub density: f64,
    pub spec_heat: f64,

    // HAMT-specific properties
    pub porosity: f64,
    pub initial_water_content: f64, // iwater in Fortran

    // HAMT data curves
    /// Sorption Isotherm curve (Water Content vs. RH).
    pub sorption_isotherm: Curve,
    /// Suction curve (Liquid Transport Coefficient vs. Water Content).
    pub suction: Curve,
    /// Redistribution curve (Liquid Transport Coefficient vs. Water Content for drying).
    pub redistribution: Curve,
    /// Vapor Diffusion factor curve (mu vs. RH).
    pub vapor_diffusion: Curve,
    /// Thermal Conductivity curve (Conductivity vs. Water Content).
    pub thermal_conductivity: Curve,

    // Discretization parameters
    /// Initial temperature for cells of this material [°C].
    pub initial_temp: f64,
    /// Initial relative humidity for cells of this material [0-1].
    pub initial_rh: f64,
    /// Characteristic size for cell division [m].
    pub div_size: f64,
    /// Minimum number of divisions for a layer of this material.
    pub div_min: i32,
    /// Maximum number of divisions for a layer of this material.
    pub div_max: i32,
}

impl Material {
    /// Creates a new material with default values.
    pub fn new(name: String) -> Self {
        Self {
            name,
            thickness: 0.0,
            conductivity: 0.0,
            density: 0.0,
            spec_heat: 0.0,
            porosity: 0.0,
            initial_water_content: 0.0,
            sorption_isotherm: Curve::new(),
            suction: Curve::new(),
            redistribution: Curve::new(),
            vapor_diffusion: Curve::new(),
            thermal_conductivity: Curve::new(),
            initial_temp: 20.0,
            initial_rh: 0.5,
            div_size: 0.005,
            div_min: 3,
            div_max: 50,
        }
    }
}

/// Represents a construction, which is an ordered sequence of material layers.
#[derive(Debug, Clone)]
pub struct Construction {
    pub name: String,
    /// An ordered list of material names that make up the construction.
    pub layer_names: Vec<String>,
}

impl Construction {
    pub fn new(name: String) -> Self {
        Self {
            name,
            layer_names: Vec::new(),
        }
    }
}
