//! Defines the core `Cell` struct, representing a finite volume element in the 1D HAMT model.

use crate::hamt::{
    boundary::BoundaryCondition,
    hamt_manager::HamtSettings,
    physics::geometry::{Pt3D, Size3D},
};

use super::material::Material;
use std::sync::{Arc, Mutex};

/// Maximum number of adjacent cells a single cell can have.
pub const ADJACENT_CELLS_MAX: usize = 6;

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum CellType {
    Material(Arc<Material>),
    Boundary(Boundary, BoundaryLocation),
    Virtual(BoundaryLocation),
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Boundary {
    Convection,
    Radiation,
    Sky,
    Ground,
    Virtual,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BoundaryLocation {
    Interior,
    Exterior,
}

/// Represents a single discrete control volume within the hygrothermal model.
///
/// This struct holds all the geometric, material, and state information for a
/// single finite difference node. The names are kept consistent with the legacy
/// Fortran and EnergyPlus implementations for easier comparison.
#[derive(Debug, Clone, Default)]
pub struct Cell {
    //
    // ========== Constant Properties (set at initialization) ==========
    /// If is a cell is a boundary cell or a material cell
    pub cell_type: CellType,
    /// Shared reference to the material; `None` indicates a virtual/boundary cell.
    pub material: Option<Arc<Material>>,

    /// The volume of the cell [m³].
    pub volume: f64,

    /// The geometric center of the cell along the primary axis [m].
    /// ## Coordinates:
    /// - `x`: The surface depth/thickness direction.
    /// - `y`: The surface width direction.
    /// - `z`: The surface height direction.
    pub origin: Pt3D,

    /// The dimensions (length) of the cell in each direction [m]. For 1D, only the first element is used.
    /// ## Coordinates:
    /// - `x`: The surface depth/thickness direction.
    /// - `y`: The surface width direction.
    /// - `z`: The surface height direction.
    pub size: Size3D,

    /// The area of overlap with adjacent cell in x, y and z direction [m²].
    #[deprecated = "Lagecy FORTRAN Code"]
    pub overlap: [f64; ADJACENT_CELLS_MAX],

    /// The distance from this cell's origin to the face of each adjacent cell [m].
    #[deprecated = "Lagecy FORTRAN Code"]
    pub dist: [f64; ADJACENT_CELLS_MAX],

    /// Indices of adjacent cells in the main `cells` vector.
    #[deprecated = "Lagecy FORTRAN Code"]
    pub adjs: [i32; ADJACENT_CELLS_MAX],

    /// Indices of the corresponding adjacency slot in the adjacent cell.
    #[deprecated = "Lagecy FORTRAN Code"]
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
    /// The calculated temperature at the current iteration.
    pub cell_temp: f64,

    /// Cell Temperature [°C].
    #[deprecated = "Use `cells_prev.cell_temp` Surface State"]
    pub temp: f64,

    /// Temperature after the current iteration (`p1` for "plus 1").
    #[deprecated = "Use `cells_next.cell_temp` variable"]
    pub temp_p1: f64,

    /// Temperature from the previous iteration (`p2` for "plus 2", though it means `p-1`).
    #[deprecated = "Use `cells_curr.cell_temp` Surface State"]
    pub temp_p2: f64,

    /// Cell Relative Humidity [0-1].
    pub cell_rh: f64,

    /// Relative humidity [0-1]. This is the committed value from the previous **time step**.
    #[deprecated = "Use `cells_prev.cell_rh` Surface State"]
    pub rh: f64,

    /// Relative Humidity [0-1]. Value after the current iteration.
    #[deprecated = "Use `cells_next.cell_rh` Surface State"]
    pub rh_p1: f64,

    /// Relative Humidity [0-1]. This is the committed value from the previous iteration .
    #[deprecated = "Use `cells_curr.cell_rh` Surface State"]
    pub rh_p2: f64,

    /// Vapor pressure [Pa].
    pub cell_vp: f64,

    #[deprecated = "Use `cells_curr.cell_vp` Surface State"]
    /// Vapor pressure [Pa]. Updated each iteration.
    pub vp: f64,

    #[deprecated = "Use `cells_next.cell_rh` Surface State"]
    /// Vapor pressure at the end of current iteration.
    pub vp_p1: f64,

    /// Saturation vapor pressure [Pa]. Updated each iteration.
    pub vp_sat: f64,

    /// Volumetric water content [kg/m³]. Updated each iteration.
    pub volumetric_water: f64,

    /// Water content for reporting purposes [kg/kg].
    pub w_report: f64,

    /// Relative humidity for reporting purposes [%].
    pub rh_p: f64, // TODO: Check the unit
}

impl Cell {
    pub fn new(cell_type: CellType, origin: Pt3D, size: Size3D) -> Self {
        let mut c = Self {
            cell_type,
            size,
            origin,
            ..Default::default()
        };
        c.read_from_material();

        c
    }

    fn read_from_material(&mut self) {
        match &self.cell_type {
            CellType::Material(material) => {
                // Link the material reference and initialize geometry and state from it
                self.material = Some(material.clone());
                self.volume = self.size.volume();

                // Constant properties
                self.density = material.density;
                self.spech = material.spec_heat;

                // Initial conditions (temperature and RH)
                self.cell_temp = material.initial_temp;
                self.temp = material.initial_temp;
                self.temp_p1 = material.initial_temp;
                self.temp_p2 = material.initial_temp;

                self.cell_rh = material.initial_rh();
                self.rh = material.initial_rh();
                self.rh_p1 = material.initial_rh();
                self.rh_p2 = material.initial_rh();

                // Initial water content: kg/kg -> kg/m³ using material density
                self.volumetric_water = material.initial_water_content * material.density;
                self.w_report = material.initial_water_content;

                // Reported RH [%]
                self.rh_p = self.cell_rh * 100.0;
            }
            _ => {
                // Do nothing for non-material cells (boundary & virtual)
            }
        }
    }
}
