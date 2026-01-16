//! Defines the core `Cell` struct, representing a finite volume element in the 1D HAMT model.

use crate::hamt::{boundary::BoundaryCondition, hamt_manager::HamtSettings, physics::geometry::Pt};

use super::material::Material;
use std::sync::{Arc, Mutex};

/// Maximum number of adjacent cells a single cell can have.
pub const ADJACENT_CELLS_MAX: usize = 6;

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum CellType {
    Material(Arc<Material>),
    Boundary(Boundary, BoundaryLocation),
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
    pub origin: Pt,

    /// The dimensions (length) of the cell in each direction [m]. For 1D, only the first element is used.
    /// ## Coordinates:
    /// - `x`: The surface depth/thickness direction.
    /// - `y`: The surface width direction.
    /// - `z`: The surface height direction.
    pub length: Pt,

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

    /// Vapor pressure [Pa]. Updated each iteration.
    pub vp: f64,

    /// Vapor pressure at the current iteration.
    pub vp_p1: f64,

    /// Saturation vapor pressure [Pa]. Updated each iteration.
    pub vp_sat: f64,

    /// Volumetric water content [kg/m³]. Updated each iteration.
    pub water: f64,

    /// Water content for reporting purposes [kg/kg].
    pub w_report: f64,

    /// Relative humidity for reporting purposes [%].
    pub rh_p: f64, // TODO: Check the unit
}

impl Cell {
    /// Creates a new `Cell` for a boundary condition.
    /// This will move the base point to the surface of the newly created cell.
    pub fn new_boundary(
        boundary: Boundary,
        location: BoundaryLocation,
        base_pt: &mut Pt,
        length: f64,
        area: f64,
    ) -> Self {
        let dummy_length = area.sqrt();
        let mut cell = Self {
            cell_type: CellType::Boundary(boundary, location),
            length: Pt(length, dummy_length, dummy_length),
            origin: base_pt.clone(),
            ..Self::default()
        };
        cell.origin += cell.length / 2.0;
        *base_pt += cell.length;
        cell
    }

    /// Creates a new `Cell` from a material layer.
    pub fn new_material_by_area(
        material: Arc<Material>,
        depth: f64,
        origin: f64,
        area: f64,
    ) -> Self {
        let dummy_length = area.sqrt();
        let mut cell = Self {
            cell_type: CellType::Material(material.clone()),
            length: Pt(depth, dummy_length, dummy_length),
            origin: Pt(origin, 0.0, 0.0),
            density: material.density,
            ..Self::default()
        };
        cell.density = material.density;
        cell.spech = material.spec_heat;

        cell.volume = cell.length.volume();

        cell
    }

    /// Update the cell's calculated properties based on the current `cell_temp` and `cell_rh`.
    pub fn update(&mut self, settings: &HamtSettings, boundary: &BoundaryCondition) {
        // 1. Update Vapor Pressure and Saturation Vapor Pressure
        self.vp_sat = crate::hamt::physics::psy_psat_fn_temp(self.cell_temp);
        self.vp = crate::hamt::physics::rh_to_vp(self.cell_rh, self.cell_temp);

        // 2. If it is a Material cell, update material properties
        if let Some(mat) = &self.material {
            // Interpolate Sorption Curve: RH -> Water Content [kg/m³] and dwdphi
            let (water_content, dwdphi) = mat.sorption_isotherm.interpolate(self.cell_rh);
            self.water = water_content;
            self.dwd_phi = dwdphi;

            // Interpolate Thermal Conductivity: Water Content -> Lambda [W/m·K]
            let (tc, _) = mat.thermal_conductivity.interpolate(self.water);
            self.w_thermal_c = tc;

            // Interpolate Vapor Diffusion Resistance Factor: RH -> Mu [-]
            let (mu, _) = mat.vapor_diffusion_resistance.interpolate(self.cell_rh);
            self.mu = mu;

            // Interpolate Liquid Transport Coefficient: Water Content -> Dw [s]
            let (dw, _) = if settings.rain_enabled && boundary.is_raining {
                mat.suction.interpolate(self.water)
            } else {
                mat.redistribution.interpolate(self.water)
            };

            self.dw = dw;

            // Update reporting variables
            self.w_report = self.water / self.density;
            self.rh_p = self.cell_rh * 100.0;
        }
    }
}
