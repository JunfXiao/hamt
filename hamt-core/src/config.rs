//! Simulation-wide configuration types shared across `surface`, `solver`,
//! `accel`, and `manager`.
//!
//! This is a *leaf* module: it depends only on `crate::uom` and
//! `crate::physics::geometry`. No `manager`, `surface`, or `solver` imports
//! are allowed here — the purpose of this module is to break the cycle where
//! `surface`/`solver` previously pulled in `manager` just for these types.

use crate::physics::geometry::{Pt3D, Size3D};
use crate::uom::dimensions::Length;
use crate::uom::dimensions::{Dimensionless, Power, Temperature, Time};
use crate::uom::units;
use crate::uom::units::{dimensionless, kelvin};

/// Boundary condition on one of the 6 axis-aligned faces of the wall box.
/// Side faces (y/z-normal) default to Adiabatic; the two x-normal faces carry
/// the existing interior/exterior Convection+Radiation sheets. Designed to admit
/// future variants (thermal bridges) without a topology rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceBc {
    /// No flux across this face (the stencil has no neighbour across it).
    Adiabatic,
    /// Active boundary face: this face participates in boundary exchange via the
    /// per-face boundary sheets (Convection / Radiation / Sky / Ground). Currently
    /// used for the two x-normal faces (interior and exterior). Future thermal-bridge
    /// variants (e.g. a y-face with a specified temperature or convective coefficient)
    /// should be added as NEW enum variants rather than overloading `Active`.
    Active,
}

/// Indices into `Surface::face_bcs`: x-min (exterior), x-max (interior), y-min, y-max, z-min, z-max.
pub const FACE_X_MIN: usize = 0;
pub const FACE_X_MAX: usize = 1;
pub const FACE_Y_MIN: usize = 2;
pub const FACE_Y_MAX: usize = 3;
pub const FACE_Z_MIN: usize = 4;
pub const FACE_Z_MAX: usize = 5;

/// Selects the inner linear solver used by `solve_timestep`.
///
/// The legacy `BlockJacobi` path runs block-Jacobi sweeps until per-iter δ
/// falls below tolerance (with the option-2 min-iter guard for
/// correctness). The `BlockTridiagonal` path assembles the implicit-Euler
/// step into a block-tridiagonal system and solves it directly via the
/// Thomas algorithm — O(N) per outer iteration and free of the
/// premature-convergence pathology (see `solver::block_tridiag`).
///
/// `BlockTridiagonal` requires `coupled_heat_moisture_enabled = true`.
/// If false, `solve_timestep` falls back to `BlockJacobi` and emits a warn!.
///
/// `Adi` (Alternating-Direction Implicit) extends `BlockTridiagonal` to
/// multi-dimensional grids via direction-by-direction sweeps. For a 1-D grid
/// (ny=nz=1) it delegates to the existing block-tridiag path. The genuine 2-D/3-D
/// ADI step is implemented in Phase 2a Task 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    BlockJacobi,
    BlockTridiagonal,
    Adi,
}

/// Tunable gains for the PI adaptive sub-stepper (`solve_timestep_adaptive`).
///
/// All fields have `Default` values that are identical to the hard-coded
/// literals that existed before this struct was introduced, so the default
/// configuration is behaviour-preserving.
#[derive(Debug, Clone)]
pub struct PiControllerSettings {
    /// Integral gain applied to `(1/LTE)^k_i`.
    pub k_i: f64,
    /// Proportional gain applied to `(prev_LTE/LTE)^k_p`.
    pub k_p: f64,
    /// Safety factor applied to the raw amplification before clamping.
    pub safety: f64,
    /// Lower clamp on the amplification (minimum step-size growth/shrink factor).
    pub amp_min: f64,
    /// Upper clamp on amplification for the very first accepted step after a
    /// failure (when `prev_lte` is `None`).
    pub amp_max_first: f64,
    /// Upper clamp on amplification for subsequent accepted steps.
    pub amp_max: f64,
    /// Floor for the temperature integration tolerance used in the LTE check
    /// (independent of the solver convergence tolerance).
    pub integ_tol_t_floor: Temperature<f64>,
    /// Floor for the relative-humidity integration tolerance used in the LTE
    /// check (independent of the solver convergence tolerance).
    pub integ_tol_rh_floor: Dimensionless<f64>,
}

impl Default for PiControllerSettings {
    fn default() -> Self {
        Self {
            k_i: 0.15,
            k_p: 0.05,
            safety: 0.8,
            amp_min: 0.25,
            amp_max_first: 2.0,
            amp_max: 1.5,
            integ_tol_t_floor: 0.25 * kelvin,
            integ_tol_rh_floor: 0.02 * dimensionless,
        }
    }
}

/// Global settings for the HAMT simulation.
#[derive(Debug, Clone)]
pub struct HamtSettings {
    /// Timestep [s].
    pub timestep: Time<f64>,
    /// Maximum nonlinear iterations (count).
    pub max_iterations: i32,
    /// Temperature convergence tolerance [K or °C].
    pub temp_convergence_tolerance: Temperature<f64>,
    /// Relative humidity convergence tolerance [-].
    pub rh_convergence_tolerance: Dimensionless<f64>,
    /// Base Cell Depth [m]
    pub base_cell_size: Size3D,
    /// Flag to enable latent heat coupling (dimensionless).
    pub latent_heat_enabled: bool,
    /// Flag to indicate if rain is present on the exterior surface.
    pub rain_enabled: bool,
    /// Clamp for latent heat contribution [W].
    pub max_latent_heat: Power<f64>,
    /// Cell division size [m]
    pub div_length: Length<f64>,
    /// Minimum cell amount for a material after division
    pub min_div: i32,
    /// Maximum cell amount for a material after division
    pub max_div: i32,
    /// Allowed length for adjacent check [m]
    pub adj_length: Length<f64>,
    /// Cell division distribution factor.
    ///
    /// When set to 1.0, the cells are divided by sine distribution.
    /// When set to 0.0, the cells are divided by linear distribution.
    pub div_distribution_factor: f64,
    /// user defined depth [m]
    pub user_defined_depth: Option<Length<f64>>,
    /// Minimum per-cell under-relaxation factor (floor) for nonlinear iteration.
    ///
    /// Each cell starts with α=1.0 (no relaxation). When a cell's iteration
    /// residuals indicate divergence, α is reduced adaptively but never below
    /// this floor. Lower values allow more aggressive damping at the cost of
    /// slower convergence.
    pub min_relaxation_factor: f64,

    /// Enable symmetric 2×2 heat–moisture block solve.
    ///
    /// When `true`, the energy equation is linearized to include the T↔φ cross-coupling
    /// term through P_sat, giving symmetric first-order coupling in both directions.
    /// Set to `false` to reproduce the legacy staggered (lagged) scheme exactly.
    pub coupled_heat_moisture_enabled: bool,

    /// Enable differential sorption-enthalpy contribution to latent heat.
    ///
    /// When `true` and a material provides a `sorption_enthalpy` curve, the effective
    /// latent heat used in the energy balance is `L_v + H_diff(φ)` instead of `L_v` alone.
    /// Has no effect if no material in the construction provides the curve.
    pub sorption_enthalpy_enabled: bool,

    /// Enable Type-II Anderson acceleration on the outer fixed-point loop.
    ///
    /// When `true`, the solver wraps the per-iteration BlockJacobi map with an AA
    /// extrapolation step using the last `anderson_history_depth` iterates.
    /// When `false`, the loop is byte-identical to the unaccelerated baseline.
    pub anderson_enabled: bool,

    /// Maximum AA history depth (mixing parameter `m`). Typical values 3..=5.
    pub anderson_history_depth: usize,

    /// Inner linear-solver method. See `SolverMethod`.
    pub solver_method: SolverMethod,

    /// Settings for the PI adaptive sub-stepper (`solve_timestep_adaptive`).
    pub pi: PiControllerSettings,
}

impl Default for HamtSettings {
    fn default() -> Self {
        Self {
            timestep: 3600.0 * units::seconds,
            max_iterations: 1500,
            temp_convergence_tolerance: 0.001 * units::kelvin,
            rh_convergence_tolerance: 0.0001 * units::dimensionless,
            base_cell_size: Pt3D(0.1 * units::meter, 0.0 * units::meter, 0.0 * units::meter),
            latent_heat_enabled: true,
            rain_enabled: false,
            max_latent_heat: 100_000.0 * units::watts,
            div_length: 0.05 * units::meter,
            min_div: 3,
            max_div: 20,
            adj_length: 0.00005 * units::meter,
            div_distribution_factor: 0.5,
            user_defined_depth: None,
            min_relaxation_factor: 0.1,
            coupled_heat_moisture_enabled: true,
            sorption_enthalpy_enabled: true,
            anderson_enabled: true,
            anderson_history_depth: 3,
            solver_method: SolverMethod::BlockTridiagonal,
            pi: PiControllerSettings::default(),
        }
    }
}

