//! Block-Thomas algorithm for block-tridiagonal systems with 2×2 blocks.
//!
//! Solves
//!   L_i · x_{i-1} + D_i · x_i + U_i · x_{i+1} = b_i,    i = 0..N-1
//! with `L_0` and `U_{N-1}` ignored (no out-of-range neighbour). All
//! blocks are 2×2; both `b` and `x` carry one (T, φ) pair per row in the
//! kernel's f64 sense (T in K, φ dimensionless — re-tagged at the
//! assembly boundary).
//!
//! The algorithm is the standard scalar Thomas recurrence with scalar
//! division replaced by 2×2 inverse:
//!
//!   forward:   D'_0 = D_0
//!              b'_0 = b_0
//!              for i in 1..N:
//!                  M_i  = L_i · (D'_{i-1})^-1
//!                  D'_i = D_i − M_i · U_{i-1}
//!                  b'_i = b_i − M_i · b'_{i-1}
//!
//!   back-sub:  x_{N-1} = (D'_{N-1})^-1 · b'_{N-1}
//!              for i in (0..N-1).rev():
//!                  x_i = (D'_i)^-1 · (b'_i − U_i · x_{i+1})
//!
//! On any singular `D'_i` the solver returns `Err(SolverError::Singular)`.
//! Callers are expected to treat this as a numerical failure and fall
//! back to per-cell decoupled handling (consistent with the existing
//! 2×2 path in `calculate_physics_step`).

use super::block_2x2::Block2x2;

#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// At least one diagonal block became singular during elimination.
    /// Includes the `i` of the failing block to aid diagnostics.
    Singular { row: usize },
    /// `lower.len()`, `upper.len()` must equal `diag.len()`. Mismatched
    /// sizes from a buggy assembler.
    LengthMismatch,
    /// Empty system.
    Empty,
}

/// Tolerance below which a 2×2 block is treated as singular during
/// elimination. Calibrated to the magnitude of the dominant diagonal
/// terms (~tcap/dt ≈ 10⁵, wcap/dt ≈ 10⁻⁴ for typical building cells).
pub const DEFAULT_SINGULARITY_EPS: f64 = 1e-30;

/// Solve the block-tridiagonal system. `lower[0]` is unused; `upper[N-1]`
/// is unused (no neighbour past the ends). Both must still be present in
/// the slice (e.g. zero-filled) so indexing matches `diag` and `rhs`.
pub fn solve(
    lower: &[Block2x2],
    diag: &[Block2x2],
    upper: &[Block2x2],
    rhs: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>, SolverError> {
    let n = diag.len();
    if n == 0 {
        return Err(SolverError::Empty);
    }
    if lower.len() != n || upper.len() != n || rhs.len() != n {
        return Err(SolverError::LengthMismatch);
    }

    if n == 1 {
        // Degenerate: no off-diagonals contribute; just invert D_0.
        let inv = diag[0]
            .try_inv(DEFAULT_SINGULARITY_EPS)
            .ok_or(SolverError::Singular { row: 0 })?;
        return Ok(vec![inv.mul_vec(rhs[0])]);
    }

    // Forward + back-sub implemented in subsequent tasks.
    todo!("forward sweep (Task 2.2)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_pair(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol
    }

    #[test]
    fn n1_identity_returns_rhs() {
        // D_0 = I, b_0 = (3, 5)  =>  x_0 = (3, 5)
        let d = vec![Block2x2::identity()];
        let l = vec![Block2x2::zero()];
        let u = vec![Block2x2::zero()];
        let b = vec![(3.0, 5.0)];
        let x = solve(&l, &d, &u, &b).unwrap();
        assert!(approx_pair(x[0], (3.0, 5.0), 1e-12));
    }

    #[test]
    fn n1_singular_diagonal_errors() {
        let d = vec![Block2x2::zero()];
        let l = vec![Block2x2::zero()];
        let u = vec![Block2x2::zero()];
        let b = vec![(1.0, 1.0)];
        assert!(matches!(solve(&l, &d, &u, &b), Err(SolverError::Singular { row: 0 })));
    }

    #[test]
    fn empty_input_errors() {
        let res = solve(&[], &[], &[], &[]);
        assert!(matches!(res, Err(SolverError::Empty)));
    }
}
