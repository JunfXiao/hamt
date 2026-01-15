use crate::hamt::material::Construction;
use std::sync::Arc;

/// Holds the indices and reporting variables for a single surface.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    /// Surface name.
    pub name: String,
    /// Construction name.
    pub construction_name: String,
    /// Shared reference to the construction.
    pub construction: Option<Arc<Construction>>,
    /// Surface area [m²].
    pub area: f64,

    // ========== Cell Indices ==========
    /// Index of the first cell belonging to this surface in the main `cells` vector.
    pub first_cell: usize,
    /// Index of the last cell belonging to this surface.
    pub last_cell: usize,
    /// Index of the exterior virtual surface cell.
    pub ext_cell: usize,
    /// Index of the interior virtual surface cell.
    pub int_cell: usize,
    /// Index of the exterior convection boundary cell.
    pub ext_con_cell: usize,
    /// Index of the interior convection boundary cell.
    pub int_con_cell: usize,
    /// Index of the exterior radiation boundary cell.
    pub ext_rad_cell: usize,
    /// Index of the exterior sky radiation boundary cell.
    pub ext_sky_cell: usize,
    /// Index of the exterior ground radiation boundary cell.
    pub ext_grn_cell: usize,
    /// Index of the user-defined depth cell for reporting.
    pub user_def_cell: usize,

    // ========== Boundary Conditions / Flags ==========
    /// Exterior vapor transfer coefficient [kg/(m²·s·Pa)].
    pub ext_vtc: Option<f64>,
    /// Interior vapor transfer coefficient [kg/(m²·s·Pa)].
    pub int_vtc: Option<f64>,

    // ========== Reporting Variables ==========
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
