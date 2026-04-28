//! Type-II Anderson acceleration for the outer fixed-point loop.
//!
//! The HAMT solver iterates a map `G : x ↦ G(x)` (a single
//! `Surface::calculate_iteration` plus the per-cell-α write-back). For stiff
//! parabolic operators the spectral radius approaches 1 and Picard convergence
//! stalls. AA(m) extrapolates from the last `m+1` iterates and their residuals
//! to find a better next guess.
//!
//! ## Math choices
//!
//! - **Type-II formulation.** Solve `min_α ‖f_k − ΔF α‖²` then update
//!   `x_{k+1} = x_k + f_k − (ΔX + ΔF) α`. We keep `ΔX` and `ΔF` as histories of
//!   *consecutive differences*.
//! - **Normal equations + Tikhonov.** With `m ≤ 5` the LS system is tiny, so we
//!   form `G = ΔF^T ΔF` (size m×m) and add `λ · trace(G)/m · I` before
//!   Gauss-Jordan inversion. `λ = 1e-10` is small enough not to bias the
//!   solution but large enough to keep the matrix well-conditioned when columns
//!   become near-parallel (which happens as iterates converge).
//! - **Monotonic-improvement gate.** AA can amplify divergence — particularly
//!   in the first few iterations when the linearisation is poor. We skip the
//!   AA step when the new residual norm fails to decrease vs the previous one
//!   (after the first three iterations, where transients are expected). The
//!   per-cell α relaxation inside `G` continues to provide damping during the
//!   skip, and AA re-engages once residuals start decreasing.
//!
//! The accelerator works on raw `Vec<f64>` SI scalars — the HAMT manager strips
//! `(T_kelvin, φ)` at the boundary (a linear-algebra-kernel boundary per the
//! UOM Discipline) and re-tags on the way back.

use std::collections::VecDeque;

/// Tikhonov regularisation strength relative to `trace(G)/m`.
const TIKHONOV_LAMBDA: f64 = 1.0e-10;

/// First N iterations are exempt from the monotonic-improvement gate
/// (early transients can legitimately oscillate).
const GATE_GRACE_ITERATIONS: usize = 3;

/// Type-II Anderson accelerator with bounded history.
pub struct AndersonAccelerator {
    /// Maximum history depth (mixing parameter `m`).
    m: usize,
    /// State vector length, set on first push.
    n: usize,
    /// Recent iterates `x_k` (raw SI scalars, length `n`).
    x_history: VecDeque<Vec<f64>>,
    /// Recent residuals `f_k = G(x_k) − x_k`.
    f_history: VecDeque<Vec<f64>>,
    /// Previous iteration's residual L2 norm — used by the improvement gate.
    last_residual_norm: f64,
    /// Number of iterations seen so far in the current solve (reset by `reset`).
    iter_count: usize,
}

impl AndersonAccelerator {
    pub fn new(m: usize) -> Self {
        Self {
            m: m.max(1),
            n: 0,
            x_history: VecDeque::new(),
            f_history: VecDeque::new(),
            last_residual_norm: f64::INFINITY,
            iter_count: 0,
        }
    }

    /// Clears all history. Called once per timestep so each macro-step starts
    /// fresh — an old history from a different state would mislead AA.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.f_history.clear();
        self.last_residual_norm = f64::INFINITY;
        self.iter_count = 0;
        self.n = 0;
    }

    /// Push the latest `(x_k, G(x_k))` pair and return an AA-extrapolated next
    /// iterate, or `None` to indicate the caller should keep `g_x_curr`.
    ///
    /// `None` is returned when:
    ///   - history has fewer than 2 entries (no `ΔF` columns yet),
    ///   - any input contains a non-finite value,
    ///   - the residual norm fails the monotonic-improvement gate,
    ///   - the regularised normal-equations solve is numerically singular.
    pub fn accelerate(&mut self, x_curr: &[f64], g_x_curr: &[f64]) -> Option<Vec<f64>> {
        debug_assert_eq!(x_curr.len(), g_x_curr.len());

        if self.n == 0 {
            self.n = x_curr.len();
        } else if self.n != x_curr.len() {
            // Shape changed unexpectedly; reset and bail out for this iteration.
            self.reset();
            self.n = x_curr.len();
        }

        // f_k = G(x_k) − x_k
        let mut f_k = vec![0.0; self.n];
        for i in 0..self.n {
            f_k[i] = g_x_curr[i] - x_curr[i];
        }

        if !f_k.iter().all(|v| v.is_finite()) || !x_curr.iter().all(|v| v.is_finite()) {
            return None;
        }

        let f_norm = l2_norm(&f_k);

        // Always grow history before deciding whether to extrapolate — even on
        // a skipped iteration we want the next iteration's ΔF to see the
        // current pair.
        self.push_history(x_curr.to_vec(), f_k.clone());
        self.iter_count += 1;

        // Monotonic-improvement gate. After the grace window, if the residual
        // is not strictly decreasing we let the per-cell α handle damping
        // and skip AA for this iteration.
        let prev_norm = self.last_residual_norm;
        self.last_residual_norm = f_norm;
        if self.iter_count > GATE_GRACE_ITERATIONS && f_norm >= prev_norm {
            return None;
        }

        let m_eff = self.x_history.len().saturating_sub(1);
        if m_eff == 0 {
            // Need at least two entries to form a single ΔF column.
            return None;
        }

        // Build ΔX and ΔF as flattened column-major (n × m_eff).
        let n = self.n;
        let mut delta_x = vec![0.0; n * m_eff];
        let mut delta_f = vec![0.0; n * m_eff];
        for j in 0..m_eff {
            // Column j corresponds to history pair (j, j+1).
            let xj0 = &self.x_history[j];
            let xj1 = &self.x_history[j + 1];
            let fj0 = &self.f_history[j];
            let fj1 = &self.f_history[j + 1];
            for i in 0..n {
                delta_x[j * n + i] = xj1[i] - xj0[i];
                delta_f[j * n + i] = fj1[i] - fj0[i];
            }
        }

        // Normal equations: solve (ΔF^T ΔF + λ I) α = ΔF^T f_k.
        let mut g = vec![0.0; m_eff * m_eff];
        for j1 in 0..m_eff {
            for j2 in 0..m_eff {
                let mut s = 0.0;
                for i in 0..n {
                    s += delta_f[j1 * n + i] * delta_f[j2 * n + i];
                }
                g[j1 * m_eff + j2] = s;
            }
        }
        let trace: f64 = (0..m_eff).map(|j| g[j * m_eff + j]).sum();
        let reg = if trace > 0.0 {
            TIKHONOV_LAMBDA * trace / m_eff as f64
        } else {
            return None;
        };
        for j in 0..m_eff {
            g[j * m_eff + j] += reg;
        }

        let mut rhs = vec![0.0; m_eff];
        for j in 0..m_eff {
            let mut s = 0.0;
            for i in 0..n {
                s += delta_f[j * n + i] * f_k[i];
            }
            rhs[j] = s;
        }

        let alpha = match solve_gauss_jordan(&mut g, &mut rhs, m_eff) {
            Some(a) => a,
            None => return None,
        };

        // x_{k+1} = x_k + f_k − (ΔX + ΔF) α
        let mut x_next = vec![0.0; n];
        for i in 0..n {
            let mut correction = 0.0;
            for j in 0..m_eff {
                correction += (delta_x[j * n + i] + delta_f[j * n + i]) * alpha[j];
            }
            x_next[i] = x_curr[i] + f_k[i] - correction;
        }

        if !x_next.iter().all(|v| v.is_finite()) {
            return None;
        }

        Some(x_next)
    }

    fn push_history(&mut self, x: Vec<f64>, f: Vec<f64>) {
        if self.x_history.len() >= self.m + 1 {
            self.x_history.pop_front();
            self.f_history.pop_front();
        }
        self.x_history.push_back(x);
        self.f_history.push_back(f);
    }
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// In-place Gauss-Jordan with partial pivoting for an `m×m` system stored
/// row-major in `a` and rhs in `b`. Returns the solution vector or `None` on
/// near-zero pivot. `m` is small (≤ 5) so this is cheap and dependency-free.
fn solve_gauss_jordan(a: &mut [f64], b: &mut [f64], m: usize) -> Option<Vec<f64>> {
    const PIVOT_EPS: f64 = 1.0e-30;
    for k in 0..m {
        // Find pivot row.
        let mut piv_row = k;
        let mut piv_val = a[k * m + k].abs();
        for r in (k + 1)..m {
            let v = a[r * m + k].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = r;
            }
        }
        if piv_val < PIVOT_EPS {
            return None;
        }
        if piv_row != k {
            for c in 0..m {
                a.swap(k * m + c, piv_row * m + c);
            }
            b.swap(k, piv_row);
        }
        // Normalise pivot row.
        let inv = 1.0 / a[k * m + k];
        for c in 0..m {
            a[k * m + c] *= inv;
        }
        b[k] *= inv;
        // Eliminate other rows.
        for r in 0..m {
            if r == k {
                continue;
            }
            let factor = a[r * m + k];
            if factor == 0.0 {
                continue;
            }
            for c in 0..m {
                a[r * m + c] -= factor * a[k * m + c];
            }
            b[r] -= factor * b[k];
        }
    }
    Some(b.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow linear iteration `x ← 0.95·x + 0.05` (fixed point at 1, contraction
    /// ratio 0.95). Plain Picard needs ~540 iterations to reach 1e-12; AA(m≥1)
    /// solves an affine 1-D map exactly after the first usable iteration, so it
    /// converges in 2-3 iterations. This is the regime AA is designed for —
    /// stiff parabolic operators with spectral radius near 1, like the HAMT
    /// outer loop. (Heron / quadratically-convergent iterations are a poor
    /// benchmark: residual columns become near-collinear, the LS system
    /// degenerates, and AA can over-extrapolate — exactly the failure mode the
    /// monotonic-improvement gate guards in production.)
    #[test]
    fn anderson_accelerates_slow_linear_iteration() {
        fn g(x: f64) -> f64 {
            0.95 * x + 0.05
        }
        let target: f64 = 1.0;
        let max_iters = 1000;
        let tol: f64 = 1e-12;

        // Plain Picard.
        let mut plain_iters = 0;
        let mut x = 0.0;
        while (x - target).abs() > tol && plain_iters < max_iters {
            x = g(x);
            plain_iters += 1;
        }
        let plain_err = (x - target).abs();

        // AA(3) from the same starting point.
        let mut aa = AndersonAccelerator::new(3);
        let mut aa_iters = 0;
        let mut x_aa = 0.0;
        while (x_aa - target).abs() > tol && aa_iters < max_iters {
            let gx = g(x_aa);
            x_aa = match aa.accelerate(&[x_aa], &[gx]) {
                Some(v) => v[0],
                None => gx,
            };
            aa_iters += 1;
        }
        let aa_err = (x_aa - target).abs();

        assert!(plain_err <= tol, "plain Picard should converge");
        assert!(aa_err <= tol, "AA should converge to within tolerance");
        assert!(
            plain_iters > 200,
            "test fixture should be slow (got {} plain iterations)",
            plain_iters
        );
        assert!(
            aa_iters * 10 < plain_iters,
            "AA should be at least 10× faster on this fixture (AA: {}, plain: {})",
            aa_iters,
            plain_iters
        );
    }

    /// The monotonic-improvement gate must skip AA once residual norms stop
    /// decreasing (after the grace window). Without this gate AA can amplify
    /// divergence on ill-conditioned residual histories.
    #[test]
    fn anderson_gate_skips_on_non_decreasing_residuals() {
        let mut aa = AndersonAccelerator::new(3);
        // Feed iterates with monotonically *growing* residuals.
        // After GATE_GRACE_ITERATIONS calls, the gate must engage.
        let mut all_after_grace_returned_none = true;
        let mut calls_after_grace = 0;
        for i in 0..(GATE_GRACE_ITERATIONS + 5) {
            let x = i as f64;
            // f_k = G(x) − x = (i + 1) — strictly increasing in i.
            let gx = x + (i as f64 + 1.0);
            let result = aa.accelerate(&[x], &[gx]);
            if i > GATE_GRACE_ITERATIONS {
                calls_after_grace += 1;
                if result.is_some() {
                    all_after_grace_returned_none = false;
                }
            }
        }
        assert!(calls_after_grace > 0);
        assert!(
            all_after_grace_returned_none,
            "monotonic-improvement gate failed to skip AA on growing residuals",
        );
    }

    #[test]
    fn first_call_returns_none() {
        let mut aa = AndersonAccelerator::new(3);
        assert!(aa.accelerate(&[1.0, 2.0], &[1.5, 2.5]).is_none());
    }

    #[test]
    fn non_finite_returns_none() {
        let mut aa = AndersonAccelerator::new(3);
        let _ = aa.accelerate(&[1.0], &[1.1]);
        let _ = aa.accelerate(&[1.1], &[1.05]);
        assert!(aa.accelerate(&[f64::NAN], &[1.0]).is_none());
    }

    #[test]
    fn history_capped_at_m_plus_1() {
        let mut aa = AndersonAccelerator::new(3);
        for i in 0..10 {
            let x = i as f64;
            let gx = 0.5 * x;
            let _ = aa.accelerate(&[x], &[gx]);
        }
        assert!(aa.x_history.len() <= 4);
        assert!(aa.f_history.len() <= 4);
    }
}
