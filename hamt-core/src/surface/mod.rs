use anyhow::Ok;
use ndarray::ArcArray3;
use serde::Serialize;
use tracing::debug;

use crate::cell::{BoundaryLocation, Cell, CellType};
use crate::uom::dimensions::{
    Area, Density, Dimensionless, EnergyFluxDensity, Length, Mass, MassFluxDensity, Temperature, VaporTransportCoeff
};
use crate::{boundary::BoundaryCondition, hamt_manager::HamtSettings, physics};
use anyhow::Result;
pub mod init;
pub mod query;
pub mod update;
use std::{collections::HashMap, sync::Arc};

/// Snapshot of on-demand surface reporting values.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SurfaceReport {
    pub temp_surf_out: Temperature<f64>,
    pub temp_surf_in: Temperature<f64>,
    pub rh_surf_out: Dimensionless<f64>,
    pub rh_surf_in: Dimensionless<f64>,
    pub temp_surf_user_def: Temperature<f64>,
    pub rh_surf_user_def: Dimensionless<f64>,
    pub water_surf_user_def: Density<f64>,
    pub q_surf_ext: EnergyFluxDensity<f64>,
    pub q_surf_int: EnergyFluxDensity<f64>,
    pub m_surf_ext: MassFluxDensity<f64>,
    pub m_surf_int: MassFluxDensity<f64>,
    pub water_total: Mass<f64>,
}

/// Holds geometry, cells, and utilities for a single surface.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Surface name.
    pub name: String,
    /// Name of the construction used by this surface.
    pub construction_name: String,
    /// Surface area [m²].
    pub area: Area<f64>,
    /// Cell lattice.
    pub cells: ArcArray3<Cell>,

    // ========= Cell Index Helpers =========
    cells_idx_map: HashMap<CellType, Vec<[usize; 3]>>,


    // ========= Optional user-defined probe =========
    /// Depth (m) at which user-defined reporting is performed.
    pub user_defined_depth: Option<Length<f64>>,

    pub user_defined_cell_idx: Option<[usize; 3]>,
}
