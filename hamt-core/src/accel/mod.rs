//! Fixed-point accelerators for the nonlinear outer loop.
use crate::surface::Surface;

/// Bare scalar knobs the relaxation needs, so `accel` does not depend on `manager`.
#[derive(Debug, Clone, Copy)]
pub struct RelaxKnobs {
    pub min_alpha: f64,
    pub max_alpha: f64,
}

/// Acts on the outer fixed-point iterate G(x). Reusable across any solver step.
///
/// Two roles are composed in the outer loop: a *relaxer* applies per-cell
/// damping (`relax`, default no-op) every iteration, and an optional
/// *extrapolator* (`extrapolate`, default `None`) refines the post-swap iterate.
/// Aitken plays the relaxer role; Anderson plays the extrapolator role.
pub trait Accelerator {
    /// Per-cell Aitken Δ² relaxation-factor update; mutates `next.relaxation_factor`.
    /// Default no-op for accelerators that only extrapolate.
    fn relax(&mut self, _next: &mut Surface, _curr: &Surface, _iter: i32, _knobs: RelaxKnobs) {}
    /// Global fixed-point extrapolation on the flattened SI iterate. Given the
    /// previous iterate `x_curr` and the fresh map output `g_x`, returns an
    /// extrapolated next iterate, or `None` to keep `g_x`. Default `None`.
    fn extrapolate(&mut self, _x_curr: &[f64], _g_x: &[f64]) -> Option<Vec<f64>> {
        None
    }
    /// Reset per-timestep history.
    fn reset(&mut self) {}
}

pub mod aitken;
pub use aitken::Aitken;

pub mod anderson;
