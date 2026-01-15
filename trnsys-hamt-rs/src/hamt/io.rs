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

        // This is a simplified parser. A more robust solution would use a dedicated
        // CSV parsing library and handle multi-line records properly.
        let parts: Vec<&str> = record.split(',').map(|s| s.trim()).collect();
        let keyword = parts[0];

        match keyword {
            "Material" => {
                if let Some(mat) = current_material.take() {
                    materials.insert(mat.name.clone(), mat);
                }
                let name = parts[1].to_string();
                let mat = Material::new(name);
                // The Fortran code reads subsequent lines for the material data.
                // This simplified parser assumes a more direct format for now.
                // A real implementation needs to handle the multi-line parsing state.
                current_material = Some(mat);
            }
            "Construction" => {
                if let Some(mat) = current_material.take() {
                    materials.insert(mat.name.clone(), mat);
                }
                let name = parts[1].to_string();
                let mut con = Construction::new(name.clone());
                let num_layers: usize = parts[2].parse().map_err(|e| {
                    TrnSysError::HamtError(format!("invalid layer count '{}': {e}", parts[2]))
                })?;
                con.layer_names = parts[3..3 + num_layers]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                constructions.insert(name, con);
            }
            // In a full implementation, we would handle all the
            // "MaterialProperty:HeatAndMoistureTransfer:*" keywords here.
            // For example:
            // "MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm" => {
            //     if let Some(mat) = &mut current_material {
            //         // parse the isotherm data and assign it to mat.sorption_isotherm
            //     }
            // }
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
