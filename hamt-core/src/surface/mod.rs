use anyhow::Ok;
use ndarray::ArcArray3;
use serde::Serialize;
use tracing::debug;

use crate::cell::{BoundaryLocation, Cell, CellType};
use crate::physics::geometry::Size3D;
use crate::uom::dimensions::{
    Area, Density, Dimensionless, EnergyFluxDensity, Length, Mass, MassFluxDensity, Temperature, VaporTransportCoeff
};
use crate::{boundary::BoundaryCondition, hamt_manager::HamtSettings, physics};
use anyhow::Result;
pub mod init;
pub mod query;
pub mod update;
use std::{collections::HashMap, sync::Arc};

/// Trilinear interpolation weights: 8 corner cells with their indices and weights.
/// Weights sum to 1.0. For 1D grids (ny=nz=1) only the first two entries are
/// non-zero; unused entries have weight 0.0 and index `[0,0,0]`.
pub type InterpolationWeights = [([usize; 3], f64); 8];

/// Snapshot of on-demand surface reporting values.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SurfaceReport {
    pub temp_surf_out: Temperature<f64>,
    pub temp_surf_int: Temperature<f64>,
    pub rh_surf_out: Dimensionless<f64>,
    pub rh_surf_int: Dimensionless<f64>,
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

    /// Surface Size(Dimension) (Thickness, length, height) [m].
    pub size: Size3D,

    /// Surface area [m²].
    pub area: Area<f64>,

    /// Thickness of each cell layer along the x-axis [m].
    pub thickness: Length<f64>,

    /// Cell lattice.
    pub cells: ArcArray3<Cell>,

    // ========= Cell Index Helpers =========
    cells_idx_map: HashMap<CellType, Vec<[usize; 3]>>,


    // ========= Boundary-adjacent material cell indices =========
    /// Index (x-axis) of the outermost interior material cell (last Material cell in x).
    /// Radiation absorbed at the interior surface is deposited here.
    pub interior_surface_material_idx: Option<usize>,

    /// Index (x-axis) of the outermost exterior material cell (first Material cell in x).
    /// Radiation absorbed at the exterior surface is deposited here.
    pub exterior_surface_material_idx: Option<usize>,

    // ========= Optional user-defined probe =========
    /// Depth (m) at which user-defined reporting is performed.
    pub user_defined_depth: Option<Length<f64>>,

    pub user_defined_cell_idx: Option<InterpolationWeights>,
}
