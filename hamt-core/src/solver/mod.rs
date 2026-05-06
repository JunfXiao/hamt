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
//!
//! ## Extending to higher dimensions
//!
//! For a 2-D or 3-D wall, the cell array stops being 1-D and the
//! neighbour stencil fans out beyond left/right. The natural extension is
//! ADI (Alternating-Direction Implicit), which keeps the implicit-Euler
//! step but solves it as a sequence of 1-D direction-line solves:
//!
//!   X-sweep:  for each (j, k), assemble the line of cells along x and
//!             solve via `block_tridiag::solve`. y- and z-neighbours
//!             enter as RHS contributions (lagged once).
//!   Y-sweep:  same but along y; x- and z- as RHS.
//!   Z-sweep:  same along z.
//!   Iterate the three sweeps to convergence on the lagged
//!             cross-direction couplings (typically 2–4 outer iterations).
//!
//! The kernel `block_tridiag::solve` does not change. Only `assembly`
//! gains sibling functions (`assemble_x_sweep`, `assemble_y_sweep`,
//! `assemble_z_sweep`) and the `Surface` API gains line iterators.
//! The legacy 1-D path stays as the special case `assemble_1d`.

pub mod block_2x2;
pub mod block_tridiag;
pub mod assembly;

#[cfg(test)]
pub(crate) mod test_fixtures;
