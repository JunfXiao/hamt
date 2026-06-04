//! Fixed-point accelerators for the nonlinear outer loop.
use crate::surface::Surface;

/// Bare scalar knobs the relaxation needs, so `accel` does not depend on `manager`.
#[derive(Debug, Clone, Copy)]
pub struct RelaxKnobs {
    pub min_alpha: f64,
    pub max_alpha: f64,
}

/// Acts on the outer fixed-point iterate G(x). Reusable across any solver step.
pub trait Accelerator {
    /// Per-cell Aitken Δ² relaxation-factor update; mutates `next.relaxation_factor`.
    fn relax(&mut self, next: &mut Surface, curr: &Surface, iter: i32, knobs: RelaxKnobs);
    /// Reset per-timestep history.
    fn reset(&mut self) {}
}

pub mod aitken;
pub use aitken::Aitken;
