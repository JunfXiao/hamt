//! Manager-internal constants and type aliases.
//!
//! The public config types (`HamtSettings`, `SolverMethod`, `PiControllerSettings`)
//! have moved to `crate::config` (a leaf module). Re-exports here preserve the
//! existing `crate::manager::*` paths so that downstream consumers and the sibling
//! DLL crate continue to resolve them unchanged.

pub use crate::config::{HamtSettings, PiControllerSettings, SolverMethod};

use crate::uom::dimensions::{Dimensionless, Temperature, Time};
use std::ops::Div;

/// Maximum allowed per-cell over-relaxation factor (upper clamp for α).
/// Manager-internal; not consumed by `surface` or `solver`.
pub const MAX_ALPHA: f64 = 1.5;

/// Temperature change rate [K/s]. Used by `AdaptiveState` in `adaptive.rs`.
pub type TempRate<S> = <Temperature<S> as Div<Time<S>>>::Output;

/// Relative-humidity change rate [1/s]. Used by `AdaptiveState` in `adaptive.rs`.
pub type RhRate<S> = <Dimensionless<S> as Div<Time<S>>>::Output;
