//! Direct linear solvers for the implicit-Euler step.
//!
//! Two layers:
//! * `block_2x2`     — 2×2 dense matrix primitives operating on raw f64.
//! * `block_tridiag` — block-Thomas algorithm for block-tridiagonal systems
//!                      with 2×2 dense blocks.
//!
//! These kernels are dimension-agnostic. The 1-D wall assembly in
//! `assembly` builds the (L, D, U, b) arrays from a `Surface` state and
//! calls `block_tridiag::solve`. Future 2-D / 3-D extensions will provide
//! ADI-style assemblers that build a 1-D line per direction and reuse the
//! same kernel.
//!
//! All numerics in this module operate on raw `f64`. Typed-quantity
//! boundaries live in the `assembly` submodule per CLAUDE.md's
//! linear-algebra-kernel exception.

pub mod block_2x2;
pub mod block_tridiag;
pub mod assembly;

#[cfg(test)]
pub(crate) mod test_fixtures;
