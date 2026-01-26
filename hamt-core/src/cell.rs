//! Defines the core `Cell` struct, representing a finite volume element in the 1D HAMT model.

use tracing::warn;

use crate::boundary::celsius_to_kelvin;
use crate::uom::dimensions::{
    Density, Dimensionless, HeatTransferCoeff, LiquidTransportCoeff, Power, Pressure,
    SpecificHeatCapacity, Temperature, ThermalConductivity, VaporTransportCoeff, Volume,
};
use crate::uom::units::{kelvin, kg_per_m3, pascals};
use crate::{
    boundary::BoundaryCondition,
    hamt_manager::HamtSettings,
    physics,
    physics::geometry::{Pt3D, Size3D},
};

use super::material::Material;
use std::ffi::os_str::Display;
use std::sync::Arc;

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

    /// The volume of the cell [m³].
    pub volume: Volume<f64>,

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

    //
    // ========== Material Properties (potentially state-dependent) ==========
    //
    /// Density of the dry material [kg/m³].
    pub density: Density<f64>,

    /// Specific heat capacity of the dry material [J/kg·K].
    pub spech: SpecificHeatCapacity<f64>,

    /// Water vapor diffusion resistance factor [-]. It's used to calculate
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub mu: Dimensionless<f64>,

    /// Moisture-dependent heat transfer coefficient [W/m²·K]. Updated each iteration.
    pub htc: HeatTransferCoeff<f64>,

    /// Additional Power from external sources [W]
    pub additional_power: Power<f64>,

    /// Vapor Transfer Coefficient [kg/(m²·s·Pa)].
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub vtc: VaporTransportCoeff<f64>,

    ///

    /// Updated by [Cell::calc_dependent_state]
    pub w_thermal_c: ThermalConductivity<f64>,

    /// Moisture storage capacity (derivative of sorption isotherm w.r.t. RH) [kg/m³].

    ///

    /// Updated by [Cell::calc_dependent_state]
    pub dwd_phi: Density<f64>,

    /// Liquid water transport coefficient [m²/s].
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub dw: LiquidTransportCoeff<f64>,

    //
    // ========== State Variables (evolve over time) ==========
    //
    /// The calculated temperature at the current iteration.
    pub cell_temp: Temperature<f64>,

    /// Cell Relative Humidity [0-1].
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub cell_rh: Dimensionless<f64>,

    /// Vapor pressure [Pa].
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub cell_vp: Pressure<f64>,

    /// Saturation vapor pressure [Pa]. Updated each iteration.
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub vp_sat: Pressure<f64>,

    /// Volumetric water content [kg/m³ Material]. Updated each iteration.
    ///
    /// Updated by [Cell::calc_dependent_state]
    pub volumetric_water: Density<f64>,
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

    /// Returns `true` if the cell is a boundary cell.
    pub fn is_boundary(&self) -> bool {
        matches!(self.cell_type, CellType::Boundary(_, _))
    }

    /// Returns `true` if the cell is an exterior boundary cell.
    pub fn is_exterior(&self) -> bool {
        matches!(
            self.cell_type,
            CellType::Boundary(_, BoundaryLocation::Exterior)
                | CellType::Virtual(BoundaryLocation::Exterior)
        )
    }

    fn read_from_material(&mut self) {
        match &self.cell_type {
            CellType::Material(material) => {
                // Link the material reference and initialize geometry and state from it
                self.volume = self.size.volume();

                // Constant properties
                self.density = material.density;
                self.spech = material.spec_heat;

                // Initial conditions (temperature and RH)
                self.cell_temp = material.initial_temp;
                self.cell_rh = material.initial_rh();

                // Initial water content: kg/kg -> kg/m³ using material density
                self.volumetric_water = material.density * material.initial_water_content;
            }
            _ => {
                // Do nothing for non-material cells (boundary & virtual)
                self.cell_temp = celsius_to_kelvin(20.0);
            }
        }
    }

    pub fn calc_dependent_state(&mut self, bc: &BoundaryCondition, settings: &HamtSettings) {
        let t_self = self.cell_temp;
        let rh_self = self.cell_rh;

        let vp_sat_pa = physics::psy_psat_fn_temp(t_self);

        self.vp_sat = vp_sat_pa;

        self.cell_vp = rh_self * vp_sat_pa;

        if let CellType::Material(mat) = &self.cell_type {
            let (water, dwdphi) = mat.sorption_isotherm.interpolate(rh_self);

            self.volumetric_water = water;

            self.dwd_phi = dwdphi;

            let (dw, _) = if bc.is_raining && settings.rain_enabled {
                mat.suction.interpolate(water)
            } else {
                mat.redistribution.interpolate(water)
            };

            // make sure dw is positive
            self.dw = dw.abs();

            let (mu, _) = mat.vapor_diffusion_resistance.interpolate(rh_self);

            self.mu = mu;

            let (k, _) = mat.thermal_conductivity.interpolate(water);

            self.w_thermal_c = k;
        }
    }

    pub fn sync_state_from(&mut self, ref_cell: &Cell) {
        self.mu = ref_cell.mu;
        self.htc = ref_cell.htc;
        self.additional_power = ref_cell.additional_power;
        self.vtc = ref_cell.vtc;
        self.w_thermal_c = ref_cell.w_thermal_c;
        self.dwd_phi = ref_cell.dwd_phi;
        self.dw = ref_cell.dw;
        self.cell_temp = ref_cell.cell_temp;
        self.cell_rh = ref_cell.cell_rh;
        self.cell_vp = ref_cell.cell_vp;
        self.vp_sat = ref_cell.vp_sat;
        self.volumetric_water = ref_cell.volumetric_water;
    }
}
