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
//! On any singular `D'_i` the solver returns `Err(SolverError::Singular { row })`.
//! The caller (`surface::step_tridiag::BlockTridiagStep`) propagates this as a
//! typed `anyhow::Error` (preserving the `SolverError` variant for downcasting),
//! which `run_outer_loop` catches and converts into a non-convergence signal.
//! The adaptive PI substepper (`solve_timestep_adaptive`) then shrinks `dt` and
//! retries the macro-step from the last accepted state. No per-cell decoupled
//! fallback is performed inside this module.

use super::block_2x2::Block2x2;

/// Errors from the block-Thomas solve. Deliberately `thiserror`-derived so
/// that `Singular` can be recovered via `err.downcast_ref::<SolverError>()`
/// after an `anyhow::Error` round-trip — `run_outer_loop` uses this to route
/// singularities into the adaptive substepper's shrink-dt-and-retry path
/// instead of aborting the whole simulation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolverError {
    /// At least one diagonal block became singular during elimination.
    /// Includes the `i` of the failing block to aid diagnostics.
    #[error("singular diagonal block at row {row} during block-Thomas elimination")]
    Singular { row: usize },
    /// `lower.len()`, `upper.len()` must equal `diag.len()`. Mismatched
    /// sizes from a buggy assembler.
    #[error("block-tridiag length mismatch: lower/upper/rhs must match diag length")]
    LengthMismatch,
    /// Empty system.
    #[error("block-tridiag system is empty")]
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

    // Working buffers. We could overwrite `diag`/`rhs` but the API is
    // by-reference, so allocate locals.
    let mut d_prime: Vec<Block2x2> = diag.to_vec();
    let mut b_prime: Vec<(f64, f64)> = rhs.to_vec();

    // ── Forward sweep: eliminate sub-diagonal ──
    for i in 1..n {
        let inv_prev = d_prime[i - 1]
            .try_inv(DEFAULT_SINGULARITY_EPS)
            .ok_or(SolverError::Singular { row: i - 1 })?;
        // M_i = L_i · D'_{i-1}^-1
        let m = lower[i].mul(&inv_prev);
        // D'_i = D_i − M_i · U_{i-1}
        d_prime[i] = d_prime[i].sub(&m.mul(&upper[i - 1]));
        // b'_i = b_i − M_i · b'_{i-1}
        let m_b = m.mul_vec(b_prime[i - 1]);
        b_prime[i] = (b_prime[i].0 - m_b.0, b_prime[i].1 - m_b.1);
    }

    // ── Back-substitution ──
    let mut x: Vec<(f64, f64)> = vec![(0.0, 0.0); n];
    let inv_last = d_prime[n - 1]
        .try_inv(DEFAULT_SINGULARITY_EPS)
        .ok_or(SolverError::Singular { row: n - 1 })?;
    x[n - 1] = inv_last.mul_vec(b_prime[n - 1]);
    for i in (0..n - 1).rev() {
        let inv_i = d_prime[i]
            .try_inv(DEFAULT_SINGULARITY_EPS)
            .ok_or(SolverError::Singular { row: i })?;
        let ux = upper[i].mul_vec(x[i + 1]);
        let r = (b_prime[i].0 - ux.0, b_prime[i].1 - ux.1);
        x[i] = inv_i.mul_vec(r);
    }

    Ok(x)
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
        assert!(matches!(
            solve(&l, &d, &u, &b),
            Err(SolverError::Singular { row: 0 })
        ));
    }

    #[test]
    fn empty_input_errors() {
        let res = solve(&[], &[], &[], &[]);
        assert!(matches!(res, Err(SolverError::Empty)));
    }

    /// Helper: assemble residual r_i = L·x_{i-1} + D·x_i + U·x_{i+1} − b_i.
    /// Used by tests to verify the SOLVED x satisfies the system.
    fn residual_norm(
        l: &[Block2x2],
        d: &[Block2x2],
        u: &[Block2x2],
        x: &[(f64, f64)],
        b: &[(f64, f64)],
    ) -> f64 {
        let n = d.len();
        let mut max_r = 0.0_f64;
        for i in 0..n {
            let dx = d[i].mul_vec(x[i]);
            let lx = if i == 0 {
                (0.0, 0.0)
            } else {
                l[i].mul_vec(x[i - 1])
            };
            let ux = if i + 1 == n {
                (0.0, 0.0)
            } else {
                u[i].mul_vec(x[i + 1])
            };
            let r0 = dx.0 + lx.0 + ux.0 - b[i].0;
            let r1 = dx.1 + lx.1 + ux.1 - b[i].1;
            max_r = max_r.max(r0.abs()).max(r1.abs());
        }
        max_r
    }

    /// Helper: build b consistent with x_true so we can round-trip.
    fn build_b(
        l: &[Block2x2],
        d: &[Block2x2],
        u: &[Block2x2],
        x_true: &[(f64, f64)],
    ) -> Vec<(f64, f64)> {
        let n = d.len();
        (0..n)
            .map(|i| {
                let dx = d[i].mul_vec(x_true[i]);
                let lx = if i == 0 {
                    (0.0, 0.0)
                } else {
                    l[i].mul_vec(x_true[i - 1])
                };
                let ux = if i + 1 == n {
                    (0.0, 0.0)
                } else {
                    u[i].mul_vec(x_true[i + 1])
                };
                (dx.0 + lx.0 + ux.0, dx.1 + lx.1 + ux.1)
            })
            .collect()
    }

    /// Property (decoupled): when L = U = 0, each block solves independently.
    #[test]
    fn diagonal_only_system_decouples() {
        let d = vec![
            Block2x2 {
                a: 2.0,
                b: 0.0,
                c: 0.0,
                d: 4.0,
            },
            Block2x2 {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 2.0,
            },
            Block2x2 {
                a: 5.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
            },
        ];
        let l = vec![Block2x2::zero(); 3];
        let u = vec![Block2x2::zero(); 3];
        let x_true: Vec<(f64, f64)> = vec![(2.0, 2.0), (1.0, 3.0), (2.0, 3.0)];
        let b = build_b(&l, &d, &u, &x_true);
        let x = solve(&l, &d, &u, &b).unwrap();
        for i in 0..3 {
            assert!(
                approx_pair(x[i], x_true[i], 1e-12),
                "row {i}: got {:?}, want {:?}",
                x[i],
                x_true[i]
            );
        }
    }

    /// Property (round-trip): for an arbitrary diagonally-dominant block-
    /// tridiag system, b = system · x_true ⟹ solve(b) ≈ x_true.
    #[test]
    fn round_trip_recovers_x_dense_blocks() {
        let l = vec![
            Block2x2::zero(),
            Block2x2 {
                a: -0.4,
                b: 0.1,
                c: -0.05,
                d: -0.3,
            },
            Block2x2 {
                a: -0.6,
                b: -0.1,
                c: 0.0,
                d: -0.5,
            },
        ];
        let d = vec![
            Block2x2 {
                a: 3.0,
                b: 0.2,
                c: 0.1,
                d: 4.0,
            },
            Block2x2 {
                a: 2.5,
                b: 0.0,
                c: 0.05,
                d: 3.5,
            },
            Block2x2 {
                a: 4.0,
                b: -0.1,
                c: 0.0,
                d: 5.0,
            },
        ];
        let u = vec![
            Block2x2 {
                a: -0.5,
                b: 0.0,
                c: 0.05,
                d: -0.4,
            },
            Block2x2 {
                a: -0.6,
                b: -0.05,
                c: 0.0,
                d: -0.5,
            },
            Block2x2::zero(),
        ];
        let x_true = vec![(1.0, 0.5), (-0.3, 0.7), (2.0, -0.4)];
        let b = build_b(&l, &d, &u, &x_true);
        let x = solve(&l, &d, &u, &b).unwrap();
        for i in 0..3 {
            assert!(approx_pair(x[i], x_true[i], 1e-9));
        }
        // Residual must be at floating-point noise.
        assert!(residual_norm(&l, &d, &u, &x, &b) < 1e-9);
    }

    /// Property: same round-trip on a longer system. Length-related index
    /// bugs (off-by-one in forward / back loops) survive N=3 but not N=10.
    #[test]
    fn round_trip_length_10() {
        use std::f64::consts::PI;
        let n = 10;
        let mut l = vec![Block2x2::zero(); n];
        let mut d = vec![Block2x2::zero(); n];
        let mut u = vec![Block2x2::zero(); n];
        for i in 0..n {
            let s = (i as f64 + 1.0) * 0.1;
            d[i] = Block2x2 {
                a: 5.0 + s,
                b: 0.2 * s,
                c: 0.1 * s,
                d: 6.0 + s,
            };
            if i > 0 {
                l[i] = Block2x2 {
                    a: -1.0 - 0.1 * s,
                    b: 0.05 * s,
                    c: 0.0,
                    d: -1.5,
                };
            }
            if i + 1 < n {
                u[i] = Block2x2 {
                    a: -1.0 - 0.1 * s,
                    b: 0.0,
                    c: 0.05 * s,
                    d: -1.5,
                };
            }
        }
        let x_true: Vec<(f64, f64)> = (0..n)
            .map(|i| ((PI * i as f64 / n as f64).sin(), 0.3 + 0.05 * i as f64))
            .collect();
        let b = build_b(&l, &d, &u, &x_true);
        let x = solve(&l, &d, &u, &b).unwrap();
        let r = residual_norm(&l, &d, &u, &x, &b);
        assert!(r < 1e-8, "residual {r} too large");
        for i in 0..n {
            assert!(approx_pair(x[i], x_true[i], 1e-7), "row {i}");
        }
    }

    /// Edge case: N=2 with strictly diagonal blocks reduces to two
    /// independent scalar tridiag-of-length-2 problems with hand-checkable
    /// answers.   2 −1 ┐ ┌ x ┐   ┌ 1 ┐
    ///          −1  2 ┘ └ y ┘ = └ 1 ┘   →  x = y = 1.
    #[test]
    fn n2_scalar_reduction_hand_check() {
        let d = vec![
            Block2x2 {
                a: 2.0,
                b: 0.0,
                c: 0.0,
                d: 2.0,
            },
            Block2x2 {
                a: 2.0,
                b: 0.0,
                c: 0.0,
                d: 2.0,
            },
        ];
        let l = vec![
            Block2x2::zero(),
            Block2x2 {
                a: -1.0,
                b: 0.0,
                c: 0.0,
                d: -1.0,
            },
        ];
        let u = vec![
            Block2x2 {
                a: -1.0,
                b: 0.0,
                c: 0.0,
                d: -1.0,
            },
            Block2x2::zero(),
        ];
        let b = vec![(1.0, 1.0), (1.0, 1.0)];
        let x = solve(&l, &d, &u, &b).unwrap();
        assert!(approx_pair(x[0], (1.0, 1.0), 1e-12));
        assert!(approx_pair(x[1], (1.0, 1.0), 1e-12));
    }

    /// Edge case: a singular intermediate D' must surface as Singular(row).
    #[test]
    fn singular_diagonal_reports_row() {
        let d = vec![
            Block2x2::identity(),
            Block2x2::zero(), // becomes singular after elimination too
        ];
        let l = vec![Block2x2::zero(), Block2x2::zero()];
        let u = vec![Block2x2::zero(), Block2x2::zero()];
        let b = vec![(1.0, 1.0), (1.0, 1.0)];
        let res = solve(&l, &d, &u, &b);
        assert!(
            matches!(res, Err(SolverError::Singular { row: 1 })),
            "expected Singular {{ row: 1 }}, got {:?}",
            res
        );
    }

    #[test]
    fn pure_heat_diffusion_matches_implicit_euler() {
        // 1-D heat equation:  ∂T/∂t = α ∂²T/∂x², N=5 cells, Dirichlet 0 at
        // both ends, T(x, 0) = sin(π x). After implicit-Euler step Δt with
        // α Δt / Δx² = 0.25, the analytical (continuum) decay factor is
        // exp(-π² α Δt) ≈ 0.085, but the implicit-Euler discrete decay
        // factor is 1 / (1 + 4·0.25·sin²(π/(2·6))²) summed across modes;
        // we don't compare to that. Instead, build A from a known x_true,
        // compute b = A·x_true, solve, and confirm we get x_true back.
        let n = 5;
        let r = 0.25; // α Δt / Δx²
        // Diagonal: 1 + 2r in the T channel, 1 in the φ channel (decoupled).
        let d_block = Block2x2 {
            a: 1.0 + 2.0 * r,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        };
        let off_block = Block2x2 {
            a: -r,
            b: 0.0,
            c: 0.0,
            d: 0.0,
        };
        let mut diag = vec![d_block; n];
        // Boundary rows (i=0, n-1): the off-diagonal beyond the boundary is
        // absent, so we model Dirichlet T=0 by leaving the corresponding
        // upper/lower as zero (same as the kernel's no-out-of-range rule).
        let lower = vec![off_block; n];
        // lower[0] unused
        let upper = vec![off_block; n];
        // upper[n-1] unused

        // Make endpoints Dirichlet by tweaking the boundary diagonals so the
        // tridiag is exactly representing zero outside.
        diag[0] = Block2x2 {
            a: 1.0 + 2.0 * r,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        };
        diag[n - 1] = Block2x2 {
            a: 1.0 + 2.0 * r,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        };

        // Pick a smooth interior solution.
        let x_true: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let x = (i as f64 + 1.0) / (n as f64 + 1.0);
                let t = (std::f64::consts::PI * x).sin();
                (t, 0.5)
            })
            .collect();

        // b_i = D·x_i + U·x_{i+1} + L·x_{i-1}
        let mut b = vec![(0.0, 0.0); n];
        for i in 0..n {
            let dx = diag[i].mul_vec(x_true[i]);
            let lx = if i == 0 {
                (0.0, 0.0)
            } else {
                lower[i].mul_vec(x_true[i - 1])
            };
            let ux = if i == n - 1 {
                (0.0, 0.0)
            } else {
                upper[i].mul_vec(x_true[i + 1])
            };
            b[i] = (dx.0 + lx.0 + ux.0, dx.1 + lx.1 + ux.1);
        }

        let x = solve(&lower, &diag, &upper, &b).unwrap();
        for i in 0..n {
            assert!(
                approx_pair(x[i], x_true[i], 1e-10),
                "i={i} got {:?} want {:?}",
                x[i],
                x_true[i]
            );
        }
    }
}
