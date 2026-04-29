//! 2×2 dense matrix operating on raw f64.
//!
//! Layout convention: row-major
//!   M = ┌ a  b ┐
//!       └ c  d ┘

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block2x2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Block2x2 {
    pub const fn zero() -> Self {
        Self { a: 0.0, b: 0.0, c: 0.0, d: 0.0 }
    }

    pub const fn identity() -> Self {
        Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0 }
    }

    /// Apply the matrix to a 2-vector `(x, y)`.
    #[inline]
    pub fn mul_vec(&self, v: (f64, f64)) -> (f64, f64) {
        (self.a * v.0 + self.b * v.1, self.c * v.0 + self.d * v.1)
    }

    /// Matrix-matrix product `self * rhs`.
    #[inline]
    pub fn mul(&self, rhs: &Block2x2) -> Block2x2 {
        Block2x2 {
            a: self.a * rhs.a + self.b * rhs.c,
            b: self.a * rhs.b + self.b * rhs.d,
            c: self.c * rhs.a + self.d * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
        }
    }

    #[inline]
    pub fn sub(&self, rhs: &Block2x2) -> Block2x2 {
        Block2x2 {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_block(a: &Block2x2, b: &Block2x2, tol: f64) -> bool {
        approx(a.a, b.a, tol) && approx(a.b, b.b, tol)
            && approx(a.c, b.c, tol) && approx(a.d, b.d, tol)
    }

    /// Property: I · v = v for every v.
    #[test]
    fn identity_acts_as_identity_on_vectors() {
        let i = Block2x2::identity();
        for &(x, y) in &[(3.0, 5.0), (-2.0, 0.0), (1e-9, 1e9), (-7.5, 4.25)] {
            let (rx, ry) = i.mul_vec((x, y));
            assert!(approx(rx, x, 1e-12) && approx(ry, y, 1e-12));
        }
    }

    /// Property: I · M = M and M · I = M for any M.
    #[test]
    fn identity_acts_as_identity_on_matrices() {
        let i = Block2x2::identity();
        let m = Block2x2 { a: 1.7, b: -2.3, c: 0.9, d: 4.1 };
        assert!(approx_block(&i.mul(&m), &m, 1e-12));
        assert!(approx_block(&m.mul(&i), &m, 1e-12));
    }

    /// Property: zero matrix annihilates: 0 · M = 0, 0 · v = 0.
    #[test]
    fn zero_annihilates() {
        let z = Block2x2::zero();
        let m = Block2x2 { a: 7.0, b: -3.0, c: 2.5, d: 9.1 };
        assert!(approx_block(&z.mul(&m), &z, 1e-12));
        assert_eq!(z.mul_vec((4.0, -1.0)), (0.0, 0.0));
    }

    /// One independently-computed reference product. Hand-derived from
    /// the textbook formula  (M·N)_ij = Σ_k M_ik N_kj  (NOT from the
    /// implementation's `a*rhs.a + b*rhs.c` expansion).
    ///   ┌ 1 2 ┐ × ┌ 5 6 ┐ = ┌ 19 22 ┐
    ///   └ 3 4 ┘   └ 7 8 ┘   └ 43 50 ┘
    #[test]
    fn mul_textbook_reference() {
        let lhs = Block2x2 { a: 1.0, b: 2.0, c: 3.0, d: 4.0 };
        let rhs = Block2x2 { a: 5.0, b: 6.0, c: 7.0, d: 8.0 };
        let expected = Block2x2 { a: 19.0, b: 22.0, c: 43.0, d: 50.0 };
        assert!(approx_block(&lhs.mul(&rhs), &expected, 1e-12));
    }

    /// Property: multiplication is associative on a generic triple.
    /// Catches any sub-product mistake without prescribing the order.
    #[test]
    fn mul_is_associative() {
        let a = Block2x2 { a: 1.1, b: 2.2, c: 3.3, d: 4.4 };
        let b = Block2x2 { a: -0.5, b: 0.7, c: 1.3, d: -1.1 };
        let c = Block2x2 { a: 2.0, b: -3.0, c: 0.4, d: 1.5 };
        let lhs = a.mul(&b).mul(&c);   // (A·B)·C
        let rhs = a.mul(&b.mul(&c));   // A·(B·C)
        assert!(approx_block(&lhs, &rhs, 1e-9));
    }

    /// Property: sub is the inverse of itself.  M − M = 0 for any M.
    #[test]
    fn sub_self_is_zero() {
        let m = Block2x2 { a: 4.0, b: 5.0, c: 6.0, d: 7.0 };
        assert!(approx_block(&m.sub(&m), &Block2x2::zero(), 1e-12));
    }
}
