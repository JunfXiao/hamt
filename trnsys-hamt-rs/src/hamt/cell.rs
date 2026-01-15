//! Defines the core `Cell` struct, representing a finite volume element in the 1D HAMT model.

use super::material::Material;
use std::sync::Arc;

/// Maximum number of adjacent cells a single cell can have.
pub const ADJACENT_CELLS_MAX: usize = 6;

/// Represents a single discrete control volume within the hygrothermal model.
///
/// This struct holds all the geometric, material, and state information for a
/// single finite difference node. The names are kept consistent with the legacy
/// Fortran and EnergyPlus implementations for easier comparison.
#[derive(Debug, Clone, Default)]
pub struct Cell {
    //
    // ========== Constant Properties (set at initialization) ==========
    //
    /// Shared reference to the material; `None` indicates a virtual/boundary cell.
    pub material: Option<Arc<Material>>,
    /// The volume of the cell [m³].
    pub volume: f64,
    /// The geometric center of the cell along the primary axis [m].
    pub origin: [f64; 3],
    /// The dimensions (length) of the cell in each direction [m]. For 1D, only the first element is used.
    pub length: [f64; 3],
    /// The area of overlap with each adjacent cell [m²].
    pub overlap: [f64; ADJACENT_CELLS_MAX],
    /// The distance from this cell's origin to the face of each adjacent cell [m].
    pub dist: [f64; ADJACENT_CELLS_MAX],
    /// Indices of adjacent cells in the main `cells` vector.
    pub adjs: [i32; ADJACENT_CELLS_MAX],
    /// Indices of the corresponding adjacency slot in the adjacent cell.
    pub adjsl: [i32; ADJACENT_CELLS_MAX],

    //
    // ========== Material Properties (potentially state-dependent) ==========
    //
    /// Density of the dry material [kg/m³].
    pub density: f64,
    /// Specific heat capacity of the dry material [J/kg·K].
    pub spech: f64,
    /// Water vapor diffusion resistance factor [-].
    pub mu: f64,
    /// Moisture-dependent thermal conductivity [W/m·K]. Updated each iteration.
    pub w_thermal_c: f64,
    /// Moisture storage capacity (derivative of sorption isotherm w.r.t. RH) [kg/m³].
    pub dwd_phi: f64,
    /// Liquid water transport coefficient [s].
    pub dw: f64,

    //
    // ========== Boundary Condition Properties ==========
    //
    /// Heat transfer coefficient for boundary cells [W/m²·K].
    pub htc: f64,
    /// Vapor transfer coefficient for boundary cells [kg/m²·s·Pa].
    pub vtc: f64,
    /// Additional heat source term [W].
    pub q_adds: f64,

    //
    // ========== State Variables (evolve over time) ==========
    //
    /// Temperature [°C]. This is the committed value from the previous time step.
    pub temp: f64,
    /// Temperature at the current iteration (`p1` for "plus 1").
    pub temp_p1: f64,
    /// Temperature from the previous iteration (`p2` for "plus 2", though it means `p-1`).
    pub temp_p2: f64,

    /// Relative humidity [0-1]. This is the committed value from the previous time step.
    pub rh: f64,
    /// Relative humidity at the current iteration.
    pub rh_p1: f64,
    /// Relative humidity from the previous iteration.
    pub rh_p2: f64,

    /// Vapor pressure [Pa]. Updated each iteration based on `rh` and `temp`.
    pub vp: f64,
    /// Vapor pressure at the current iteration.
    pub vp_p1: f64,
    /// Saturation vapor pressure [Pa]. Updated each iteration based on `temp_p1`.
    pub vp_sat: f64,

    /// Volumetric water content [kg/m³]. Updated each iteration based on `rh_p1`.
    pub water: f64,
    /// Water content for reporting purposes [kg/kg].
    pub w_report: f64,
    /// Relative humidity for reporting purposes [%].
    pub rh_p: f64,
}

impl Cell {
    /// Creates a new `Cell` for a boundary condition.
    pub fn new_boundary(runor: f64, length: f64) -> Self {
        let mut cell = Self::default();
        cell.length[0] = length;
        cell.origin[0] = length / 2.0 + runor;
        cell
    }

    /// Creates a new `Cell` from a material layer.
    pub fn new_material(
        material: Arc<Material>,
        length: f64,
        origin: f64,
        volume: f64,
        surface_area: f64,
    ) -> Self {
        let mut cell = Self::default();
        cell.material = Some(material.clone());
        cell.density = material.density;
        cell.spech = material.spec_heat;
        cell.length[0] = length;
        cell.origin[0] = origin;
        cell.volume = volume;
        for i in 0..ADJACENT_CELLS_MAX {
            cell.overlap[i] = surface_area;
        }
        cell
    }
}
