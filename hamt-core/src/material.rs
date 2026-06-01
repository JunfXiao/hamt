//! Defines the data structures for materials and constructions used in the HAMT model.

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    sync::{Arc, RwLock},
};

use crate::{
    boundary::celsius_to_kelvin,
    error::{Error, InvalidInputReason},
    physics::{Curve, LogY},
    uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_SPECIFIC_ENERGY, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
        Quantity,
        dimensions::{Density, Dimensionless, Length, SpecificHeatCapacity, Temperature},
    },
};
use anyhow::{Context, Ok, Result};
use tracing::error;

/// Holds all the hygrothermal properties of a single material.
///
/// This struct corresponds to the `MaterialProperties` in the Fortran implementation
/// and the `MaterialHAMT` class in the EnergyPlus C++ code.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,

    /// Thickness of the material [m].
    pub thickness: Length<f64>,

    // /// Conductivity of the material [W/(m·K)].
    // #[deprecated = "Not used at all"]
    // pub conductivity: ThermalConductivity<f64>,
    /// Density of the material [kg/m³].
    pub density: Density<f64>,

    /// Specific heat of the material [J/(kg·K)].
    pub spec_heat: SpecificHeatCapacity<f64>,

    // Radiation properties
    // #[deprecated = "Not used at all"]
    // pub thermal_absorptance: Dimensionless<f64>,
    // #[deprecated = "Not used at all"]
    // pub solar_absorptance: Dimensionless<f64>,
    // #[deprecated = "Not used at all"]
    // pub visible_absorptance: Dimensionless<f64>,

    // HAMT-specific properties
    /// Porosity of the material [-].
    pub porosity: Dimensionless<f64>,

    /// Initial water content ratio of the material [kg/kg].
    pub initial_water_content: Dimensionless<f64>, // iwater in Fortran

    // HAMT data curves
    /// Sorption Isotherm curve (Water Content vs. RH).
    ///
    /// - X: Relative Humidity [-]
    /// - Y: Water Content [kg/m³]
    pub sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT>,

    /// Water Transport Coefficient Suction curve (Liquid Transport Coefficient vs. Water Content).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Liquid Transport Coefficient [m²/s]
    ///
    /// Uses log-y PCHIP per WUFI convention because D_w spans ~6 decades.
    pub suction: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY>,

    /// Water Transport Coefficient Redistribution curve (Liquid Transport Coefficient vs. Water Content for drying).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Liquid Transport Coefficient [m²/s]
    ///
    /// Uses log-y PCHIP per WUFI convention because D_w spans ~6 decades.
    pub redistribution: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY>,

    /// Vapor Diffusion Resistance curve (mu vs. RH).
    ///
    /// - X: RH [-]
    /// - Y: Vapor Diffusion Resistance $\mu$ [-]
    pub vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS>,

    /// Thermal Conductivity curve (Conductivity vs. Water Content).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Thermal Conductivity [W/mK]
    pub thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY>,

    /// Differential heat of sorption curve (extra heat released on adsorption, beyond L_v).
    ///
    /// - X: Relative Humidity [-]
    /// - Y: Differential heat of sorption [J/kg]
    ///
    /// Optional: when `None`, pure L_v is used (legacy behavior). Only relevant for
    /// hygroscopic materials (wood, gypsum, cellulose); non-hygroscopic materials
    /// (mineral wool, foams) leave this as `None`.
    pub sorption_enthalpy: Option<Curve<DIM_DIMENSIONLESS, DIM_SPECIFIC_ENERGY>>,

    // Discretization parameters
    /// Initial temperature for cells of this material [°C].
    pub initial_temp: Temperature<f64>,

    // Division Parameters
    pub num_div: Option<usize>,

    /// Cell division distribution factor.
    ///
    /// When set to 1.0, the cells are divided by sine distribution.
    /// When set to 0.0, the cells are divided by linear distribution.
    pub div_distribution_factor: f64,
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Material {}

impl Hash for Material {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Unnamed Material".to_string(),
            thickness: Length::<f64>::zero(),
            density: Density::<f64>::zero(),
            spec_heat: SpecificHeatCapacity::<f64>::zero(),
            porosity: Quantity(0.5),
            initial_water_content: Quantity(0.5),
            sorption_isotherm: Curve::default(),
            suction: Curve::default(),
            redistribution: Curve::default(),
            vapor_diffusion_resistance: Curve::default(),
            thermal_conductivity: Curve::default(),
            sorption_enthalpy: None,
            initial_temp: celsius_to_kelvin(20.0),
            num_div: None,
            div_distribution_factor: 0.7,
        }
    }
}

impl Material {
    pub fn new(name: String) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    pub fn initial_water_density(&self) -> Density<f64> {
        let density = self.density;
        self.initial_water_content * density
    }

    pub fn initial_rh(&self) -> Dimensionless<f64> {
        self.sorption_isotherm
            .reverse_interpolate(self.initial_water_density())
            .0
    }
}

/// Represents a construction, which is an ordered sequence of material layers.
#[derive(Debug, Clone, Default)]
pub struct Construction {
    pub name: String,
    /// An ordered list of material names that make up the construction.
    pub layers: Vec<Arc<Material>>,
    pub surface_vapor_coef: SurfaceVaporCoef,
}

impl Construction {
    /// Creates a new construction with the given name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            layers: Vec::new(),
            surface_vapor_coef: SurfaceVaporCoef::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceVaporCoef {
    pub is_external_const: bool,
    pub external_value: f64,
    pub is_internal_const: bool,
    pub internal_value: f64,
}

impl Default for SurfaceVaporCoef {
    fn default() -> Self {
        Self {
            is_external_const: false,
            external_value: 2.2e-08,
            is_internal_const: false,
            internal_value: 9e-09,
        }
    }
}

#[derive(Debug, Default)]
pub struct MaterialLibrary {
    materials: RwLock<HashMap<String, Arc<Material>>>,
    constructions: RwLock<HashMap<String, Arc<Construction>>>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self {
            materials: RwLock::new(HashMap::new()),
            constructions: RwLock::new(HashMap::new()),
        }
    }

    fn get_material_write_lock(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<Material>>>> {
        self.materials
            .write()
            .map_err(|_| Error::LockError("Material Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn get_material_read_lock(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Material>>>> {
        self.materials
            .read()
            .map_err(|_| Error::LockError("Material Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    fn get_construction_write_lock(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<Construction>>>> {
        self.constructions
            .write()
            .map_err(|_| Error::LockError("Construction Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn get_construction_read_lock(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Construction>>>> {
        self.constructions
            .read()
            .map_err(|_| Error::LockError("Construction Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn add_material(&self, material: Material) -> Result<Arc<Material>> {
        let mut materials = self.get_material_write_lock()?;
        if materials.contains_key(&material.name) {
            error!("Material with name '{}' already exists.", material.name);
            Err(Error::InvalidInput(
                "Material".to_string(),
                material.name,
                InvalidInputReason::Duplicate,
            )
            .into())
        } else {
            let arc_material = Arc::new(material);
            materials.insert(arc_material.name.clone(), arc_material.clone());
            Ok(arc_material)
        }
    }

    pub fn add_construction(&self, construction: Construction) -> Result<Arc<Construction>> {
        let mut constructions = self.get_construction_write_lock()?;
        if constructions.contains_key(&construction.name) {
            error!(
                "Construction with name '{}' already exists.",
                construction.name
            );
            Err(Error::InvalidInput(
                "Construction".to_string(),
                construction.name,
                InvalidInputReason::Duplicate,
            )
            .into())
        } else {
            let arc_construction = Arc::new(construction);
            constructions.insert(arc_construction.name.clone(), arc_construction.clone());
            Ok(arc_construction)
        }
    }

    pub fn create_construction(
        &self,
        construction_name: String,
        materials: Vec<String>,
        surface_vapor_coef: SurfaceVaporCoef,
    ) -> Result<Arc<Construction>> {
        let mut layers = Vec::new();
        let materials_lock = self.get_material_read_lock()?;
        for mat_name in materials {
            if let Some(mat) = materials_lock.get(&mat_name) {
                layers.push(mat.clone());
            } else {
                return Err(Error::InvalidInput(
                    "Material".to_string(),
                    mat_name,
                    InvalidInputReason::Missing,
                )
                .into());
            }
        }
        let construction = Construction {
            name: construction_name,
            layers,
            surface_vapor_coef,
        };

        self.add_construction(construction)
    }

    pub fn get_material(&self, name: &str) -> Result<Arc<Material>> {
        let materials = self.get_material_read_lock()?;
        Ok(materials.get(name).cloned().ok_or_else(|| {
            Error::from(Error::InvalidInput(
                "Material".to_string(),
                name.to_string(),
                InvalidInputReason::Missing,
            ))
        })?)
    }

    pub fn get_construction(&self, name: &str) -> Result<Arc<Construction>> {
        let constructions = self.get_construction_read_lock()?;
        Ok(constructions.get(name).cloned().ok_or_else(|| {
            Error::from(Error::InvalidInput(
                "Construction".to_string(),
                name.to_string(),
                InvalidInputReason::Missing,
            ))
        })?)
    }
}
