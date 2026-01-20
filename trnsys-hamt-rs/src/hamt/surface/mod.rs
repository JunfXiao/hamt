use anyhow::Ok;
use ndarray::ArcArray3;
use tracing::debug;

use crate::hamt::cell::{BoundaryLocation, Cell, CellType};
use crate::hamt::uom::DensityQty;
use crate::hamt::uom::dimensions::{
    Area, Dimensionless, EnergyFlux, HeatTransferCoeff, Length, Mass, MassFlux, Temperature,
    VaporTransportCoeff,
};
use crate::hamt::{boundary::BoundaryCondition, hamt_manager::HamtSettings, physics};
use anyhow::Result;
pub mod init;
pub mod query;
pub mod update;
use std::{collections::HashMap, sync::Arc};

/// Snapshot of on-demand surface reporting values.
#[derive(Debug, Clone, Default)]
pub struct SurfaceReport {
    pub temp_surf_out: Temperature<f64>,
    pub temp_surf_in: Temperature<f64>,
    pub rh_surf_out: Dimensionless<f64>,
    pub rh_surf_in: Dimensionless<f64>,
    pub temp_surf_user_def: Temperature<f64>,
    pub rh_surf_user_def: Dimensionless<f64>,
    pub water_surf_user_def: DensityQty<f64>,
    pub q_surf_ext: EnergyFlux<f64>,
    pub q_surf_int: EnergyFlux<f64>,
    pub m_surf_ext: MassFlux<f64>,
    pub m_surf_int: MassFlux<f64>,
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

    // ========= Boundary Conditions / Flags =========
    /// Exterior vapor transfer coefficient [kg/(m²·s·Pa)].
    ext_vtc: VaporTransportCoeff<f64>,

    /// Exterior heat transfer coefficient [W/(m²·K)].
    ext_htc: HeatTransferCoeff<f64>,

    /// Interior vapor transfer coefficient [kg/(m²·s·Pa)].
    int_vtc: VaporTransportCoeff<f64>,

    /// Interior heat transfer coefficient [W/(m²·K)].
    int_htc: HeatTransferCoeff<f64>,

    // ========= Optional user-defined probe =========
    /// Depth (m) at which user-defined reporting is performed.
    pub user_defined_depth: Option<Length<f64>>,

    pub user_defined_cell_idx: Option<[usize; 3]>,
}
