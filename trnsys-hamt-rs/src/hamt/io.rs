//! Handles reading and parsing of the HAMT material and construction data file.

use tracing::{info, warn};

use crate::hamt::material::{Curve, Material, MaterialLibrary, SurfaceVaporCoef};
use crate::trnsys::error::{InputError, TrnSysError};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

/// Enum defining the expected properties in the IDF file.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum HamtProperty {
    Material,
    Settings,
    SorptionIsotherm,
    Suction,
    Redistribution,
    Diffusion,
    ThermalConductivity,
    Construction,
    SurfaceVaporCoefs,
}

/// Defines whether a property has a fixed number of parameters or variable.
enum LengthSpec {
    Fixed(usize),
    Variable,
}

impl HamtProperty {
    /// Maps a keyword string to the Enum variant.
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "Material" => Some(Self::Material),
            "MaterialProperty:HeatAndMoistureTransfer:Settings" => Some(Self::Settings),
            "MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm" => {
                Some(Self::SorptionIsotherm)
            }
            "MaterialProperty:HeatAndMoistureTransfer:Suction" => Some(Self::Suction),
            "MaterialProperty:HeatAndMoistureTransfer:Redistribution" => Some(Self::Redistribution),
            "MaterialProperty:HeatAndMoistureTransfer:Diffusion" => Some(Self::Diffusion),
            "MaterialProperty:HeatAndMoistureTransfer:ThermalConductivity" => {
                Some(Self::ThermalConductivity)
            }
            "Construction" => Some(Self::Construction),
            "SurfaceProperties:VaporCoefficients" => Some(Self::SurfaceVaporCoefs),
            _ => None,
        }
    }

    /// Returns the string representation of the property.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Material => "Material",
            Self::Settings => "MaterialProperty:HeatAndMoistureTransfer:Settings",
            Self::SorptionIsotherm => "MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm",
            Self::Suction => "MaterialProperty:HeatAndMoistureTransfer:Suction",
            Self::Redistribution => "MaterialProperty:HeatAndMoistureTransfer:Redistribution",
            Self::Diffusion => "MaterialProperty:HeatAndMoistureTransfer:Diffusion",
            Self::ThermalConductivity => {
                "MaterialProperty:HeatAndMoistureTransfer:ThermalConductivity"
            }
            Self::Construction => "Construction",
            Self::SurfaceVaporCoefs => "SurfaceProperties:VaporCoefficients",
        }
    }

    /// Returns the expected parameter length (excluding property keyword and name).
    fn length_spec(&self) -> LengthSpec {
        match self {
            // Material: Roughness, Thickness, Cond, Dens, SpHeat, ThAbs, SolAbs, VisAbs (8 params)
            Self::Material => LengthSpec::Fixed(8),
            // Settings: Porosity, InitialWaterContent (2 params)
            Self::Settings => LengthSpec::Fixed(2),
            Self::SorptionIsotherm => LengthSpec::Variable,
            Self::Suction => LengthSpec::Variable,
            Self::Redistribution => LengthSpec::Variable,
            Self::Diffusion => LengthSpec::Variable,
            Self::ThermalConductivity => LengthSpec::Variable,
            Self::Construction => LengthSpec::Variable,
            Self::SurfaceVaporCoefs => LengthSpec::Fixed(4),
        }
    }
}

/// Loads all material and construction definitions from a specified HAMT data file.
///
/// The file is parsed handling comments (starting with '!') and multi-line entities
/// ending with ';'.
pub fn load_hamt_data<P: AsRef<Path>>(path: P) -> Result<MaterialLibrary> {
    let mut file = File::open(path.as_ref().clone())?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .with_context(|| format!("Reading file '{:?}'", path.as_ref()))?;

    // 1. Parse content into a raw HashMap representation
    let raw_data = parse_idf_content(&content)
        .with_context(|| format!("Parsing file '{:?}'", path.as_ref()))?;

    // 2. Initialize Material Library
    let library = MaterialLibrary::new();
    let mut materials: HashMap<String, Material> = HashMap::new();
    let mut surface_vapor_coefs: HashMap<String, SurfaceVaporCoef> = HashMap::new();

    // 3. First pass: Create base Material objects
    for ((prop, name), params) in &raw_data {
        if let HamtProperty::Material = prop {
            // Params: [Roughness, Thickness, Conductivity, Density, SpecHeat, ThAbs, SolAbs, VisAbs]
            let mat = Material {
                name: name.clone(),
                thickness: parse_number(&params[1], "Thickness")?,
                conductivity: parse_number(&params[2], "Heat Conductivity")?,
                density: parse_number(&params[3], "Density")?,
                spec_heat: parse_number(&params[4], "Specific Heat")?,
                thermal_absorptance: parse_number(&params[5], "Thermal Absorptance")?,
                solar_absorptance: parse_number(&params[6], "Solar Absorptance")?,
                visible_absorptance: parse_number(&params[7], "Visible Absorptance")?,
                ..Default::default()
            };
            materials.insert(name.clone(), mat);
        }
    }

    // 4. Second pass: Apply additional properties to materials
    for ((prop, name), params) in &raw_data {
        if let Some(mat) = materials.get_mut(name) {
            match prop {
                HamtProperty::Settings => {
                    mat.porosity = parse_number(&params[0], "Porosity")?;
                    mat.initial_water_content = parse_number(&params[1], "Initial Water Content")?;
                }
                HamtProperty::SorptionIsotherm => {
                    mat.sorption_isotherm = parse_curve(params, "SorptionIsotherm")?;
                }
                HamtProperty::Suction => {
                    mat.suction = parse_curve(params, "Suction")?;
                }
                HamtProperty::Redistribution => {
                    mat.redistribution = parse_curve(params, "Redistribution")?;
                }
                HamtProperty::Diffusion => {
                    mat.vapor_diffusion_resistance = parse_curve(params, "Diffusion")?;
                }
                HamtProperty::ThermalConductivity => {
                    mat.thermal_conductivity = parse_curve(params, "ThermalConductivity")?;
                }
                HamtProperty::Construction
                | HamtProperty::Material
                | HamtProperty::SurfaceVaporCoefs => {}
            }
        } else if *prop != HamtProperty::Construction {
            warn!(
                "Property '{}' found for unknown material '{}'",
                prop.as_str(),
                name
            );
        }
    }

    // Register materials in the library
    for mat in materials.into_values() {
        library.add_material(mat)?;
    }

    // 5. Fourth pass: Surface Vapor Coefficients
    for ((prop, name), params) in &raw_data {
        if let HamtProperty::SurfaceVaporCoefs = prop {
            let is_ext = params[0].eq_ignore_ascii_case("Yes");
            let ext_val = parse_number(&params[1], "External Vapor Coefficient")?;
            let is_int = params[2].eq_ignore_ascii_case("Yes");
            let int_val = parse_number(&params[3], "Internal Vapor Coefficient")?;

            let coef = SurfaceVaporCoef {
                is_external_const: is_ext,
                external_value: ext_val,
                is_internal_const: is_int,
                internal_value: int_val,
            };
            if let Some(_) = surface_vapor_coefs.insert(name.clone(), coef) {
                return Err(TrnSysError::from(InputError::DuplicateItem {
                    item: "SurfaceVaporCoef".to_string(),
                    name: name.clone(),
                }))
                .with_context(|| {
                    format!(
                        "adding SurfaceVaporCoef of the construction'{}' to the material library",
                        name
                    )
                });
            }
        }
    }

    // 6. Third pass: Create Constructions
    for ((prop, name), params) in &raw_data {
        if let HamtProperty::Construction = prop {
            if params.is_empty() {
                continue;
            }

            let mut layer_names = Vec::new();
            // Check if first param is a count (legacy format) or just the first layer
            let start_idx = if params[0].parse::<usize>().is_ok() {
                1
            } else {
                0
            };

            for p in &params[start_idx..] {
                layer_names.push(p.clone());
            }

            // Make sure surface vapor coefficients exist for this construction
            // otherwise use default

            let surface_vapor_coef = surface_vapor_coefs.get(name).cloned().unwrap_or_else(||{
                info!("Construction '{}' has no surface vapor coefficient defined. Default value is used.", &name);
                SurfaceVaporCoef::default()
            });

            library.create_construction(name.clone(), layer_names, surface_vapor_coef)?;
        }
    }

    Ok(library)
}

/// Utility function to parse a number from a string, wrapping errors in TrnSysError.
fn parse_number<T: FromStr>(input: &str, field_name: &str) -> Result<T, TrnSysError> {
    input.trim().parse::<T>().map_err(|_| {
        InputError::BadParameter {
            index: -1,
            message: format!("Failed to parse number '{}' for {}", input, field_name),
        }
        .into()
    })
}

/// Helper to parse curve data (count, x, y pairs).
fn parse_curve(params: &[String], name: &str) -> Result<Curve, TrnSysError> {
    if params.is_empty() {
        return Ok(Curve::new());
    }

    let count: usize = parse_number(&params[0], &format!("{} point count", name))?;
    let mut x = Vec::new();
    let mut y = Vec::new();

    // Check strict length: 1 (count) + 2 * count (pairs)
    if params.len() != 1 + 2 * count {
        return Err(InputError::WrongDefinition {
            item: "Curve".to_string(),
            name: name.to_string(),
            reason: format!(
                "Expected {} points ({} values), got {} values",
                count,
                1 + 2 * count,
                params.len()
            ),
        }
        .into());
    }

    for i in 0..count {
        let x_val = parse_number(&params[1 + i * 2], &format!("{} x[{}]", name, i))?;
        let y_val = parse_number(&params[1 + i * 2 + 1], &format!("{} y[{}]", name, i))?;
        x.push(x_val);
        y.push(y_val);
    }

    Ok(Curve { x, y })
}

/// Parses the text content into a structured HashMap.
fn parse_idf_content(content: &str) -> Result<HashMap<(HamtProperty, String), Vec<String>>> {
    let mut clean_text = String::new();

    // Pre-process: Strip comments and flatten lines
    for line in content.lines() {
        let line_content = if let Some(idx) = line.find('!') {
            &line[..idx]
        } else {
            line
        };
        clean_text.push_str(line_content);
        clean_text.push(' '); // Add space to prevent concatenation of tokens across lines
    }

    let mut result = HashMap::new();

    // Split by entity separator ';'
    for entity in clean_text.split(';') {
        let entity = entity.trim();
        if entity.is_empty() {
            continue;
        }

        let tokens: Vec<String> = entity
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if tokens.len() < 2 {
            warn!("Skipping malformed entity (not enough tokens): {}", entity);
            continue;
        }

        let prop_str = &tokens[0];
        let name = &tokens[1];
        let params = tokens[2..].to_vec();

        let property = match HamtProperty::from_str(prop_str) {
            Some(p) => p,
            None => {
                warn!("Unknown Keyword: {}", prop_str);
                continue;
            }
        };

        // Validate Parameter Length
        match property.length_spec() {
            LengthSpec::Fixed(n) => {
                if params.len() != n {
                    return Err(TrnSysError::from(InputError::WrongDefinition {
                        item: prop_str.to_string(),
                        name: name.clone(),
                        reason: format!("Expected {} parameters, found {}", n, params.len()),
                    }))
                    .with_context(|| format!("parsing property '{}' for '{}'", prop_str, name));
                }
            }
            LengthSpec::Variable => {
                // Variable length validation is handled by specific parsers (e.g. parse_curve)
            }
        }

        // Check for Duplicates
        let key = (property, name.clone());
        if result.contains_key(&key) {
            return Err(TrnSysError::from(InputError::DuplicateItem {
                item: prop_str.to_string(),
                name: name.clone(),
            }))
            .with_context(|| {
                format!(
                    "checking duplicates for property '{}' and name '{}'",
                    prop_str, name
                )
            });
        }

        result.insert(key, params);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn test_parse_idf_content() {
        let idf_content = include_str!("../../fixtures/idf_example.idf");
        // create a temporary file
        let temp_path = std::env::temp_dir().join("test_idf.idf");
        std::fs::write(&temp_path, idf_content).unwrap();
        let library = load_hamt_data(&temp_path).unwrap();
        assert!(library.get_material("05_AGEPAN_THD_jr_0.035").is_ok());

        info!("Library: {:#?}", library)
    }
}
