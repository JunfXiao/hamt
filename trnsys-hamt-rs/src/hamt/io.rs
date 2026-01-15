//! Handles reading and parsing of the HAMT material and construction data file.

use crate::hamt::material::{Construction, Material};
use crate::trnsys::error::TrnSysError;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Loads all material and construction definitions from a specified HAMT data file.
///
/// The file is expected to be a comma-separated text file, where each object
/// is defined by a keyword on a line, followed by its data fields on subsequent
/// lines.
///
/// # Arguments
///
/// * `path` - The path to the HAMT data file.
///
/// # Returns
///
/// A tuple containing:
/// 1. A `HashMap` of all parsed `Material`s, keyed by name.
/// 2. A `HashMap` of all parsed `Construction`s, keyed by name.
pub fn load_hamt_data<P: AsRef<Path>>(
    path: P,
) -> Result<(HashMap<String, Material>, HashMap<String, Construction>), TrnSysError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut materials = HashMap::new();
    let mut constructions = HashMap::new();
    let mut current_material: Option<Material> = None;

    for line in reader.lines() {
        let line = line?;
        let record = line.trim();
        if record.is_empty() || record.starts_with('!') {
            continue;
        }

        let parts: Vec<&str> = record.split(',').map(|s| s.trim()).collect();
        let keyword = parts[0];

        match keyword {
            "Material" => {
                if let Some(mat) = current_material.take() {
                    materials.insert(mat.name.clone(), mat);
                }
                let name = parts.get(1).map(|s| s.to_string()).ok_or_else(|| {
                    TrnSysError::HamtError("Material keyword found with no name".to_string())
                })?;
                let mat = Material::new(name);
                current_material = Some(mat);
            }
            "Construction" => {
                if let Some(mat) = current_material.take() {
                    materials.insert(mat.name.clone(), mat);
                }
                let name = parts.get(1).map(|s| s.to_string()).ok_or_else(|| {
                    TrnSysError::HamtError("Construction keyword found with no name".to_string())
                })?;
                let num_layers: usize =
                    parts.get(2).and_then(|s| s.parse().ok()).ok_or_else(|| {
                        TrnSysError::HamtError(format!(
                            "Invalid or missing layer count for Construction '{}'",
                            name
                        ))
                    })?;

                if parts.len() < 3 + num_layers {
                    return Err(TrnSysError::HamtError(format!(
                        "Construction '{}' has {} layers but not enough names provided",
                        name, num_layers
                    )));
                }

                let layer_names = parts[3..3 + num_layers]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();

                let mut con = Construction::new(name.clone());
                con.layer_names = layer_names;
                constructions.insert(name, con);
            }
            // In a full implementation, we would handle all the
            // "MaterialProperty:HeatAndMoistureTransfer:*" keywords here.
            _ => {
                // TODO: Handle other properties or ignore.
            }
        }
    }

    // Add the last material being processed
    if let Some(mat) = current_material.take() {
        materials.insert(mat.name.clone(), mat);
    }

    Ok((materials, constructions))
}
