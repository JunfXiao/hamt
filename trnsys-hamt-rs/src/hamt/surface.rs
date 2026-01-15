use crate::hamt::{cell::Cell, material::Construction};
use std::sync::{Arc, RwLock};

/// Holds the indices and reporting variables for a single surface.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    /// Surface name (dimensionless).
    pub name: String,
    /// Construction name (dimensionless).
    pub construction_name: String,
    /// Construction reference, if known (dimensionless).
    pub construction: Option<Arc<Construction>>,
    /// Surface area [m²].
    pub area: f64,

    /// Exterior virtual cell
    pub ext_cell: Option<Arc<RwLock<Cell>>>,
    /// Interior virtual cell
    pub int_cell: Option<Arc<RwLock<Cell>>>,
    /// Exterior convection cell
    pub ext_con_cell: Option<Arc<RwLock<Cell>>>,
    /// Interior convection cell
    pub int_con_cell: Option<Arc<RwLock<Cell>>>,
    /// Exterior radiation cell
    pub ext_rad_cell: Option<Arc<RwLock<Cell>>>,
    /// Exterior sky radiation cell
    pub ext_sky_cell: Option<Arc<RwLock<Cell>>>,
    /// Exterior ground radiation cell
    pub ext_grn_cell: Option<Arc<RwLock<Cell>>>,
    /// User-defined depth cell
    pub user_def_cell: Option<Arc<RwLock<Cell>>>,

    // Boundary conditions / Flags
    /// Exterior vapor transfer coefficient [kg/(m²·s·Pa)].
    pub ext_vtc: Option<f64>,
    /// Interior vapor transfer coefficient [kg/(m²·s·Pa)].
    pub int_vtc: Option<f64>,
}

pub struct SurfaceReport {
    /// Exterior surface temperature [°C].
    pub temp_surf_out: f64,
    /// Interior surface temperature [°C].
    pub temp_surf_in: f64,
    /// Exterior surface relative humidity [-].
    pub rh_surf_out: f64,
    /// Interior surface relative humidity [-].
    pub rh_surf_in: f64,
    /// User-defined depth temperature [°C].
    pub temp_surf_user_def: f64,
    /// User-defined depth relative humidity [-].
    pub rh_surf_user_def: f64,
    /// User-defined depth volumetric water content [kg/m³].
    pub water_surf_user_def: f64,
    /// Exterior surface heat flux [W/m²].
    pub q_surf_ext: f64,
    /// Interior surface heat flux [W/m²].
    pub q_surf_int: f64,
    /// Exterior surface moisture flux [kg/(m²·s)].
    pub m_surf_ext: f64,
    /// Interior surface moisture flux [kg/(m²·s)].
    pub m_surf_int: f64,
    /// Total water content over surface cells [kg].
    pub water_total: f64,
}
