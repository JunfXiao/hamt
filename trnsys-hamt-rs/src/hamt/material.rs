//! Defines the data structures for materials and constructions used in the HAMT model.

use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, RwLock},
};

use crate::trnsys::error::{InputError, TrnSysError};
use anyhow::{Context, Ok, Result};
use tracing::error;

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

    /// Linearly interpolates a value from a curve.
    /// Also optionally returns the gradient at the interpolation point.
    pub fn interpolate(&self, in_value: f64) -> (f64, f64) {
        if self.x.is_empty() {
            return (0.0, 0.0);
        }

        if in_value.is_nan() {
            return (self.y[0], 0.0);
        }

        let mut upper_index: Option<usize> = None;
        for (i, &xv) in self.x.iter().enumerate() {
            match xv.partial_cmp(&in_value) {
                Some(std::cmp::Ordering::Equal) => return (self.y[i], 0.0),
                Some(std::cmp::Ordering::Greater) => {
                    upper_index = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match upper_index {
            Some(0) => (self.y[0], 0.0),
            Some(i) => {
                let x1 = self.x[i - 1];
                let x2 = self.x[i];
                let y1 = self.y[i - 1];
                let y2 = self.y[i];

                let grad = if (x2 - x1).abs() > 1e-9 {
                    (y2 - y1) / (x2 - x1)
                } else {
                    0.0
                };
                let val = y1 + grad * (in_value - x1);
                (val, grad)
            }
            None => (*self.y.last().unwrap(), 0.0),
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

    /// Thickness of the material [m].
    pub thickness: f64,

    /// Conductivity of the material [W/(m·K)].
    pub conductivity: f64,

    /// Density of the material [kg/m³].
    pub density: f64,

    /// Specific heat of the material [J/(kg·K)].
    pub spec_heat: f64,

    // Radiation properties
    pub thermal_absorptance: f64,
    pub solar_absorptance: f64,
    pub visible_absorptance: f64,

    // HAMT-specific properties
    /// Porosity of the material [-].
    pub porosity: f64,

    /// Initial water content ratio of the material [kg/kg].
    pub initial_water_content: f64, // iwater in Fortran

    // HAMT data curves
    /// Sorption Isotherm curve (Water Content vs. RH).
    ///
    /// - X: Relative Humidity [-]
    /// - Y: Water Content [kg/m³]
    pub sorption_isotherm: Curve,

    /// Water Transport Coefficient Suction curve (Liquid Transport Coefficient vs. Water Content).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Liquid Transport Coefficient [m²/s]
    pub suction: Curve,

    /// Water Transport Coefficient Redistribution curve (Liquid Transport Coefficient vs. Water Content for drying).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Liquid Transport Coefficient [m²/s]
    pub redistribution: Curve,

    /// Vapor Diffusion Resistance curve (mu vs. RH).
    ///
    /// - X: Relative Humidity [-]
    /// - Y: Vapor Diffusion Resistance [-]
    pub vapor_diffusion_resistance: Curve,

    /// Thermal Conductivity curve (Conductivity vs. Water Content).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Thermal Conductivity [W/mK]
    pub thermal_conductivity: Curve,

    // Discretization parameters
    /// Initial temperature for cells of this material [°C].
    pub initial_temp: f64,
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
            thickness: 0.0,
            conductivity: 0.0,
            density: 9999.0,
            spec_heat: 9999.0,
            thermal_absorptance: 0.5,
            solar_absorptance: 0.2,
            visible_absorptance: 0.01,
            porosity: 0.0,
            initial_water_content: 0.0,
            sorption_isotherm: Curve::new(),
            suction: Curve::new(),
            redistribution: Curve::new(),
            vapor_diffusion_resistance: Curve::new(),
            thermal_conductivity: Curve::new(),
            initial_temp: 20.0,
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

    pub fn initiaal_water_density(&self) -> f64 {
        self.initial_water_content * self.density
    }

    pub fn initial_rh(&self) -> f64 {
        self.sorption_isotherm
            .interpolate(self.initiaal_water_density())
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
            .map_err(|_| TrnSysError::LockError("Material Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn get_material_read_lock(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Material>>>> {
        self.materials
            .read()
            .map_err(|_| TrnSysError::LockError("Material Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    fn get_construction_write_lock(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<Construction>>>> {
        self.constructions
            .write()
            .map_err(|_| TrnSysError::LockError("Construction Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn get_construction_read_lock(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Construction>>>> {
        self.constructions
            .read()
            .map_err(|_| TrnSysError::LockError("Construction Library".to_string()))
            .with_context(|| "Failed to acquire write lock on Material Library")
    }

    pub fn add_material(&self, material: Material) -> Result<Arc<Material>> {
        let mut materials = self.get_material_write_lock()?;
        if materials.contains_key(&material.name) {
            error!("Material with name '{}' already exists.", material.name);
            Err(InputError::DuplicateItem {
                item: "Material".to_string(),
                name: material.name,
            }
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
            Err(InputError::DuplicateItem {
                item: "Construction".to_string(),
                name: construction.name,
            }
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
                return Err(InputError::MissingItem {
                    item: "Material".to_string(),
                    name: mat_name,
                }
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
            TrnSysError::from(InputError::MissingItem {
                item: "Material".to_string(),
                name: name.to_string(),
            })
        })?)
    }

    pub fn get_construction(&self, name: &str) -> Result<Arc<Construction>> {
        let constructions = self.get_construction_read_lock()?;
        Ok(constructions.get(name).cloned().ok_or_else(|| {
            TrnSysError::from(InputError::MissingItem {
                item: "Construction".to_string(),
                name: name.to_string(),
            })
        })?)
    }
}
