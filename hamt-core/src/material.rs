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
    uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY, DIM_WATER_CONTENT,
         Dimension, Quantity, 
        dimensions::{Density, Dimensionless, Length, SpecificHeatCapacity, Temperature},
    },
};
use anyhow::{Context, Ok, Result};
use tracing::error;

/// Represents a 2D data curve, typically from material property measurements.
///
/// This is used to store properties that vary with a single independent variable,
/// such as the sorption isotherm (water content vs. RH) or thermal conductivity
/// vs. water content. The `interp` function in the `physics` module will operate
/// on this struct.
#[derive(Debug, Clone, Default)]
pub struct Curve<const DX: Dimension, const DY: Dimension> {
    pub x: Vec<Quantity<f64, DX>>,
    pub y: Vec<Quantity<f64, DY>>,
}

impl<const DX: Dimension, const DY: Dimension> Curve<DX, DY> {
    /// Creates an empty curve.
    pub fn new(x_series: Vec<Quantity<f64, DX>>, y_series: Vec<Quantity<f64, DY>>) -> Self {
        Self {
            x: x_series,
            y: y_series,
        }
    }

    /// Linearly interpolates a value from a curve.
    /// Also optionally returns the gradient at the interpolation point.
    /// Returns the interpolated Y and the gradient (dY/dX) as f64.
    pub fn interpolate(
        &self,
        in_x: Quantity<f64, DX>,
    ) -> (Quantity<f64, DY>, Quantity<f64, { DY.sub(DX) }>) {
        let zero_y = Quantity::<f64, DY>::zero();
        let zero_grad = Quantity::<f64, { DY.sub(DX) }>::zero();

        if self.x.is_empty() {
            return (zero_y, zero_grad);
        }

        let mut upper_index: Option<usize> = None;
        for (i, &xv) in self.x.iter().enumerate() {
            match xv.partial_cmp(&in_x) {
                Some(std::cmp::Ordering::Equal) => return (self.y[i], zero_grad),
                Some(std::cmp::Ordering::Greater) => {
                    upper_index = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match upper_index {
            Some(0) => (self.y[0], zero_grad),
            Some(i) => {
                let x1 = self.x[i - 1];
                let x2 = self.x[i];
                let y1 = self.y[i - 1];
                let y2 = self.y[i];

                let grad = if (x2 - x1).abs() > Quantity::<f64, DX>(1e-9) {
                    (y2 - y1) / (x2 - x1)
                } else {
                    zero_grad
                };
                let val = y1.0 + grad.0 * (in_x.0 - x1.0);
                (Quantity::<f64, DY>(val), grad)
            }
            None => (*self.y.last().unwrap(), zero_grad),
        }
    }

    /// Linearly interpolates a value from a curve.
    /// Also optionally returns the gradient at the interpolation point.
    /// Returns the interpolated X and the gradient (dX/dY) as f64.
    pub fn reverse_interpolate(
        &self,
        in_y: Quantity<f64, DY>,
    ) -> (Quantity<f64, DX>, Quantity<f64, { DX.sub(DY) }>) {
        let zero_x = Quantity::<f64, DX>::zero();
        let zero_grad = Quantity::<f64, { DX.sub(DY) }>::zero();

        if self.y.is_empty() {
            return (zero_x, zero_grad);
        }

        let mut upper_index: Option<usize> = None;
        for (i, &yv) in self.y.iter().enumerate() {
            match yv.partial_cmp(&in_y) {
                Some(std::cmp::Ordering::Equal) => return (self.x[i], zero_grad),
                Some(std::cmp::Ordering::Greater) => {
                    upper_index = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match upper_index {
            Some(0) => (self.x[0], zero_grad),
            Some(i) => {
                let y1 = self.y[i - 1];
                let y2 = self.y[i];
                let x1 = self.x[i - 1];
                let x2 = self.x[i];

                let grad = if (y2 - y1).abs() > Quantity::<f64, DY>(1e-9) {
                    (x2 - x1) / (y2 - y1)
                } else {
                    zero_grad
                };
                let val = x1.0 + grad.0 * (in_y.0 - y1.0);
                (Quantity::<f64, DX>(val), grad)
            }
            None => (*self.x.last().unwrap(), zero_grad),
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
    pub suction: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF>,

    /// Water Transport Coefficient Redistribution curve (Liquid Transport Coefficient vs. Water Content for drying).
    ///
    /// - X: Water Content [kg/m³]
    /// - Y: Liquid Transport Coefficient [m²/s]
    pub redistribution: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF>,

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

    // Discretization parameters
    /// Initial temperature for cells of this material [°C].
    pub initial_temp: Temperature<f64>,

    // Division Parameters
    pub num_div: Option<usize>,
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
            initial_temp: celsius_to_kelvin(20.0),
            num_div: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uom::units::{dimensionless, kg_per_m3};

    #[test]
    fn interpolate_linear_midpoint() {
        let curve = Curve {
            x: vec![0.0 * dimensionless, 1.0 * dimensionless],
            y: vec![0.0 * kg_per_m3, 10.0 * kg_per_m3],
        };
        let (y, grad) = curve.interpolate(0.5 * dimensionless);
        assert!((y.value_in(kg_per_m3) - 5.0).abs() < 1e-12);
        assert!((grad.value_in(kg_per_m3) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn reverse_interpolate_midpoint() {
        let curve = Curve {
            x: vec![0.0 * dimensionless, 1.0 * dimensionless],
            y: vec![0.0 * kg_per_m3, 10.0 * kg_per_m3],
        };
        let (x, grad) = curve.reverse_interpolate(5.0 * kg_per_m3);
        assert!((x.value() - 0.5).abs() < 1e-12);
        assert!((grad.value_in(kg_per_m3) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn interpolate_below_min_returns_first_point() {
        let curve = Curve {
            x: vec![0.2 * dimensionless, 1.0 * dimensionless],
            y: vec![2.0 * kg_per_m3, 10.0 * kg_per_m3],
        };
        let (y, grad) = curve.interpolate(0.0 * dimensionless);
        assert!((y.value_in(kg_per_m3) - 2.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    #[test]
    fn interpolate_above_max_returns_last_point() {
        let curve = Curve {
            x: vec![0.0 * dimensionless, 0.8 * dimensionless],
            y: vec![1.0 * kg_per_m3, 5.0 * kg_per_m3],
        };
        let (y, grad) = curve.interpolate(1.0 * dimensionless);
        assert!((y.value_in(kg_per_m3) - 5.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    #[test]
    fn reverse_interpolate_below_min_returns_first_point() {
        let curve = Curve {
            x: vec![0.0 * dimensionless, 1.0 * dimensionless],
            y: vec![2.0 * kg_per_m3, 6.0 * kg_per_m3],
        };
        let (x, grad) = curve.reverse_interpolate(1.0 * kg_per_m3);
        assert!((x.value() - 0.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    #[test]
    fn reverse_interpolate_above_max_returns_last_point() {
        let curve = Curve {
            x: vec![0.0 * dimensionless, 1.0 * dimensionless],
            y: vec![2.0 * kg_per_m3, 6.0 * kg_per_m3],
        };
        let (x, grad) = curve.reverse_interpolate(10.0 * kg_per_m3);
        assert!((x.value() - 1.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }
}
