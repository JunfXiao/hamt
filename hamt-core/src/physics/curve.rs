//! Defines the data structures for materials and constructions used in the HAMT model.

use std::fmt::Debug;
use std::marker::PhantomData;

use crate::uom::{Dimension, Quantity};

/// Errors that can arise when constructing a [`Curve`] via [`Curve::try_new`].
///
/// `LinearY` curves never raise these; they can only originate from `LogY`
/// preprocessing, where the y-axis must be strictly positive (with optional
/// leading zero rows interpreted as "below threshold => y = 0").
#[derive(Debug, Clone, PartialEq)]
pub enum CurveError {
    /// `x` and `y` slices passed to `try_new` had different lengths.
    LengthMismatch { x_len: usize, y_len: usize },
    /// A `LogY` curve had a zero y in a non-leading row (i.e. after a strictly
    /// positive y has been seen). Such tables are ambiguous: the WUFI
    /// leading-zero convention applies only to the prefix.
    MidTableZero { row: usize },
    /// A negative y value appeared anywhere in a `LogY` curve.
    NegativeY { row: usize },
}

impl std::fmt::Display for CurveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurveError::LengthMismatch { x_len, y_len } => write!(
                f,
                "Curve length mismatch: x has {} entries, y has {}",
                x_len, y_len
            ),
            CurveError::MidTableZero { row } => write!(
                f,
                "Curve has zero y at row {} after non-zero entries (LogY accepts \
                 zero y only in leading rows)",
                row
            ),
            CurveError::NegativeY { row } => {
                write!(f, "Curve has negative y at row {} (LogY requires y >= 0)", row)
            }
        }
    }
}

impl std::error::Error for CurveError {}

/// Interpolation scale strategy for [`Curve`].
///
/// `preprocess` runs once at construction. The other two transform functions
/// are hot: every `interpolate` call decodes one (or zero) PCHIP samples
/// through them. All methods are `#[inline]`; for `LinearY` they are true
/// identities so LLVM collapses them to no-ops.
///
/// The `Extra` associated type carries any per-scale runtime metadata that
/// `Curve` needs to store. `LinearY` uses `()` — zero-sized, so it costs
/// nothing in the struct layout — while `LogY` uses `Option<f64>` for the
/// optional below-threshold cutoff. Threading the threshold through this
/// associated type (rather than a hard-coded `Option<f64>` field on `Curve`)
/// lets LLVM constant-fold `LinearY::threshold(()) -> None` and eliminate
/// the threshold check entirely on `LinearY` curves.
pub trait CurveScale {
    /// Per-scale runtime metadata stored on the [`Curve`].
    ///
    /// * `LinearY::Extra = ()` — zero-sized, elided from the struct.
    /// * `LogY::Extra = Option<f64>` — the below-threshold cutoff (when a
    ///   leading-zero prefix was present in the table).
    type Extra: Copy + Default;

    /// Preprocess raw `(x, y)` before PCHIP setup. Returns
    /// `(x_pchip, y_internal, extra)` where `extra` is the scale-specific
    /// runtime metadata stored on the curve. For `LogY`, leading zero-y rows
    /// are stripped and their last x becomes the threshold; below the
    /// threshold, `interpolate` short-circuits to `(0, 0)`.
    fn preprocess(
        x: &[f64],
        y: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Self::Extra), CurveError>;

    /// Return the below-threshold cutoff, if any. `LinearY` always returns
    /// `None` (inlined identity → LLVM eliminates the check).
    fn threshold(extra: Self::Extra) -> Option<f64>;

    /// Inverse transform of the internal y back to the physical y.
    fn from_internal(t: f64) -> f64;

    /// Chain-rule conversion of the internal gradient `dt/dx` back to the
    /// physical `dy/dx`, given the already-decoded physical y.
    fn to_physical_grad(dt_dx: f64, y_physical: f64) -> f64;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LinearY;

#[derive(Debug, Clone, Copy, Default)]
pub struct LogY;

impl CurveScale for LinearY {
    type Extra = ();

    #[inline]
    fn preprocess(
        x: &[f64],
        y: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Self::Extra), CurveError> {
        if x.len() != y.len() {
            return Err(CurveError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        Ok((x.to_vec(), y.to_vec(), ()))
    }

    #[inline]
    fn threshold(_extra: Self::Extra) -> Option<f64> {
        None
    }

    #[inline]
    fn from_internal(t: f64) -> f64 {
        t
    }

    #[inline]
    fn to_physical_grad(dt_dx: f64, _y_physical: f64) -> f64 {
        dt_dx
    }
}

impl CurveScale for LogY {
    type Extra = Option<f64>;

    #[inline]
    fn preprocess(
        x: &[f64],
        y: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Self::Extra), CurveError> {
        if x.len() != y.len() {
            return Err(CurveError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }

        // Walk the leading zero-y prefix. Negative y is always an error;
        // a zero y after the first non-zero is an error (mid-table zero).
        let mut first_nonzero: Option<usize> = None;
        for (i, &yi) in y.iter().enumerate() {
            if yi < 0.0 {
                return Err(CurveError::NegativeY { row: i });
            }
            if yi > 0.0 && first_nonzero.is_none() {
                first_nonzero = Some(i);
            }
            if yi == 0.0 && first_nonzero.is_some() {
                return Err(CurveError::MidTableZero { row: i });
            }
        }

        // All-zero non-empty input is the WUFI convention for "no liquid
        // transport at any state" — used by vapour foils and non-hygroscopic
        // insulations. Return empty PCHIP arrays so `interpolate`'s empty
        // short-circuit yields (0, 0) for any x.
        if first_nonzero.is_none() {
            return Ok((Vec::new(), Vec::new(), None));
        }

        let start = first_nonzero.unwrap();
        let x_pchip: Vec<f64> = x[start..].to_vec();
        let y_internal: Vec<f64> = y[start..].iter().map(|v| v.ln()).collect();
        // Threshold = first x at which y becomes positive. If there were no
        // leading zeros, no threshold is needed (caller may still extrapolate
        // below x[0]; PCHIP returns the boundary value flat there).
        let x_threshold = if start > 0 { Some(x[start]) } else { None };

        Ok((x_pchip, y_internal, x_threshold))
    }

    #[inline]
    fn threshold(extra: Self::Extra) -> Option<f64> {
        extra
    }

    #[inline]
    fn from_internal(t: f64) -> f64 {
        t.exp()
    }

    #[inline]
    fn to_physical_grad(dt_dx: f64, y_physical: f64) -> f64 {
        // dy/dx = d(exp(ln y))/dx = y · d(ln y)/dx
        y_physical * dt_dx
    }
}

/// Represents a 2D data curve, typically from material property measurements.
///
/// Generic over an interpolation scale `S` (default `LinearY`). Choose `LogY`
/// for quantities that span many decades in y where linear PCHIP smears the
/// transition into the final segment — notably the WUFI liquid-transport
/// coefficients `suction` and `redistribution`. `LogY` also recognises the
/// WUFI leading-zero-row convention: a prefix of `(x, 0)` rows defines a
/// threshold; `interpolate` returns `(0, 0)` below it.
///
/// Construction is fallible via [`Curve::try_new`] (only `LogY` can fail —
/// see [`CurveError`]). A `Curve<_, _, LinearY>::new` infallible alias is
/// provided for legacy call sites with hand-curated monotone data.
#[derive(Debug, Clone)]
pub struct Curve<const DX: Dimension, const DY: Dimension, S: CurveScale = LinearY>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    // PCHIP-space (post-preprocess). For LinearY identical to (x_physical,
    // y_physical) stripped to raw f64; for LogY y_internal = ln(y_physical).
    x_pchip: Vec<f64>,
    y_internal: Vec<f64>,
    tangents: Vec<f64>,
    // Per-scale runtime metadata (e.g. LogY's optional below-threshold
    // cutoff). For `LinearY` this is `()` — zero-sized, elided from the
    // struct layout — so a default LinearY `Curve` does not pay for LogY's
    // threshold field.
    extra: S::Extra,

    // Retained for reverse_interpolate and Default-empty bookkeeping.
    x_physical: Vec<Quantity<f64, DX>>,
    y_physical: Vec<Quantity<f64, DY>>,

    _scale: PhantomData<S>,
}

impl<const DX: Dimension, const DY: Dimension, S: CurveScale> Default for Curve<DX, DY, S>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    fn default() -> Self {
        Self {
            x_pchip: Vec::new(),
            y_internal: Vec::new(),
            tangents: Vec::new(),
            extra: S::Extra::default(),
            x_physical: Vec::new(),
            y_physical: Vec::new(),
            _scale: PhantomData,
        }
    }
}

// -------- Generic constructor + interpolation --------
impl<const DX: Dimension, const DY: Dimension, S: CurveScale> Curve<DX, DY, S>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    /// Fallible constructor. `S::preprocess` may reject malformed input
    /// (e.g. a `LogY` curve with a mid-table zero or a negative y).
    /// Passing two empty vectors yields an empty curve and never runs
    /// preprocessing — this keeps `Material::default()` infallible.
    pub fn try_new(
        x_series: Vec<Quantity<f64, DX>>,
        y_series: Vec<Quantity<f64, DY>>,
    ) -> Result<Self, CurveError> {
        if x_series.is_empty() && y_series.is_empty() {
            return Ok(Self::default());
        }

        // strip: PCHIP arithmetic operates on raw f64 internals; the typed
        // wrappers are reinstated at the interpolate boundary.
        let x_raw: Vec<f64> = x_series.iter().map(|q| q.0).collect();
        let y_raw: Vec<f64> = y_series.iter().map(|q| q.0).collect();

        let (x_pchip, y_internal, extra) = S::preprocess(&x_raw, &y_raw)?;
        let tangents = calculate_pchip_tangents(&x_pchip, &y_internal);

        Ok(Self {
            x_pchip,
            y_internal,
            tangents,
            extra,
            x_physical: x_series,
            y_physical: y_series,
            _scale: PhantomData,
        })
    }

    /// Interpolates a value from the curve using PCHIP.
    /// Returns the interpolated Y and the gradient (dY/dX).
    ///
    /// For a `LogY` curve with a leading-zero prefix, inputs below the
    /// threshold short-circuit to `(0, 0)`; above threshold the PCHIP is
    /// evaluated on `(x, ln y)` and decoded via `exp` plus chain rule.
    pub fn interpolate(
        &self,
        in_x: Quantity<f64, DX>,
    ) -> (Quantity<f64, DY>, Quantity<f64, { DY.sub(DX) }>) {
        let zero_y = Quantity::<f64, DY>::zero();
        let zero_grad = Quantity::<f64, { DY.sub(DX) }>::zero();

        // Empty curve: return (0, 0). Routing this through `S::from_internal`
        // would yield `exp(0) = 1` for `LogY`, breaking `Curve::default()`.
        if self.x_pchip.is_empty() {
            return (zero_y, zero_grad);
        }

        // For `LinearY` curves `S::threshold(())` is `None`, which LLVM
        // constant-folds away — the whole conditional becomes dead code.
        // `LogY` paths preserve the existing below-threshold short-circuit.
        if let Some(thr) = S::threshold(self.extra) {
            if in_x.0 < thr {
                return (zero_y, zero_grad);
            }
        }

        let (t, dt_dx) = pchip_interp_raw(in_x.0, &self.x_pchip, &self.y_internal, &self.tangents);
        let y_phys = S::from_internal(t);
        let dy_dx = S::to_physical_grad(dt_dx, y_phys);
        (Quantity(y_phys), Quantity(dy_dx))
    }

    /// Linearly interpolates a value from a curve (Reverse: Y -> X).
    /// Note: This remains linear as analytic inversion of cubic PCHIP is complex.
    /// Operates on the *physical* (x, y) arrays, so it is correct for `LogY`
    /// curves too — though no current call site uses reverse-interpolation on
    /// a `LogY` curve (only `sorption_isotherm`, which is `LinearY`).
    pub fn reverse_interpolate(
        &self,
        in_y: Quantity<f64, DY>,
    ) -> (Quantity<f64, DX>, Quantity<f64, { DX.sub(DY) }>) {
        let zero_x = Quantity::<f64, DX>::zero();
        let zero_grad = Quantity::<f64, { DX.sub(DY) }>::zero();

        if self.y_physical.is_empty() {
            return (zero_x, zero_grad);
        }

        if !in_y.0.is_finite() {
            return (self.x_physical[0], zero_grad);
        }

        let mut upper_index: Option<usize> = None;
        for (i, &yv) in self.y_physical.iter().enumerate() {
            match yv.partial_cmp(&in_y) {
                Some(std::cmp::Ordering::Equal) => return (self.x_physical[i], zero_grad),
                Some(std::cmp::Ordering::Greater) => {
                    upper_index = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match upper_index {
            Some(0) => (self.x_physical[0], zero_grad),
            Some(i) => {
                let y1 = self.y_physical[i - 1];
                let y2 = self.y_physical[i];
                let x1 = self.x_physical[i - 1];
                let x2 = self.x_physical[i];

                let grad = if (y2 - y1).abs() > Quantity::<f64, DY>(1e-9) {
                    (x2 - x1) / (y2 - y1)
                } else {
                    zero_grad
                };
                let val = x1.0 + grad.0 * (in_y.0 - y1.0);
                (Quantity::<f64, DX>(val), grad)
            }
            None => (*self.x_physical.last().unwrap(), zero_grad),
        }
    }
}

// -------- Backward-compat infallible constructor for LinearY --------
impl<const DX: Dimension, const DY: Dimension> Curve<DX, DY, LinearY>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    /// Infallible `LinearY` constructor — `LinearY::preprocess` only fails on
    /// length mismatch, which is a programmer error. Existing call sites that
    /// pass hand-picked monotone data continue to use this.
    pub fn new(
        x_series: Vec<Quantity<f64, DX>>,
        y_series: Vec<Quantity<f64, DY>>,
    ) -> Self {
        Self::try_new(x_series, y_series).expect("valid LinearY curve inputs")
    }
}

// ---- PCHIP primitives (raw-f64 space, scale-agnostic) ----

/// Calculates derivatives (tangents) using the PCHIP (Monotone Cubic Hermite)
/// algorithm. Inputs are post-preprocess; for `LogY` the y array is `ln(y)`.
fn calculate_pchip_tangents(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut tangents = Vec::with_capacity(n);
    let mut slopes = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        let dy = y[i + 1] - y[i];
        if dx.abs() < 1e-9 {
            slopes.push(0.0);
        } else {
            slopes.push(dy / dx);
        }
    }

    // Start point
    let m0 = if n == 2 {
        slopes[0]
    } else {
        let h0 = x[1] - x[0];
        let h1 = x[2] - x[1];
        let d0 = ((2.0 * h0 + h1) * slopes[0] - h0 * slopes[1]) / (h0 + h1);
        if d0 * slopes[0] <= 0.0 {
            0.0
        } else if (slopes[0] * slopes[1] <= 0.0) && (d0.abs() > 3.0 * slopes[0].abs()) {
            3.0 * slopes[0]
        } else {
            d0
        }
    };
    tangents.push(m0);

    // Interior points
    for i in 0..n - 2 {
        let s0 = slopes[i];
        let s1 = slopes[i + 1];
        if s0 * s1 <= 0.0 {
            tangents.push(0.0);
        } else {
            let h0 = x[i + 1] - x[i];
            let h1 = x[i + 2] - x[i + 1];
            let w1 = 2.0 * h1 + h0;
            let w2 = h1 + 2.0 * h0;
            let harmonic_mean = (w1 + w2) / (w1 / s0 + w2 / s1);
            tangents.push(harmonic_mean);
        }
    }

    // End point
    let m_last = if n == 2 {
        slopes[0]
    } else {
        let h_n_2 = x[n - 2] - x[n - 3];
        let h_n_1 = x[n - 1] - x[n - 2];
        let d_last =
            ((2.0 * h_n_1 + h_n_2) * slopes[n - 2] - h_n_1 * slopes[n - 3]) / (h_n_1 + h_n_2);
        if d_last * slopes[n - 2] <= 0.0 {
            0.0
        } else if (slopes[n - 2] * slopes[n - 3] <= 0.0)
            && (d_last.abs() > 3.0 * slopes[n - 2].abs())
        {
            3.0 * slopes[n - 2]
        } else {
            d_last
        }
    };
    tangents.push(m_last);

    tangents
}

/// PCHIP interpolation on raw `f64` slices. Returns `(y, dy/dx)` in internal
/// space; for `LogY` the caller decodes through `exp` + chain rule.
fn pchip_interp_raw(
    target_x: f64,
    x_data: &[f64],
    y_data: &[f64],
    tangents: &[f64],
) -> (f64, f64) {
    if x_data.is_empty() {
        return (0.0, 0.0);
    } else if x_data.len() == 1 {
        return (y_data[0], 0.0);
    }

    // Guard against non-finite inputs (e.g. NaN from unstable upstream state)
    // to avoid invalid comparisons inside the interval search.
    if !target_x.is_finite() {
        if target_x > 0.0 {
            return (y_data[y_data.len() - 1], 0.0);
        } else {
            return (y_data[0], 0.0);
        }
    }

    let n = x_data.len();

    // Constant extrapolation outside the table.
    if target_x < x_data[0] {
        return (y_data[0], 0.0);
    }
    if target_x > x_data[n - 1] {
        return (y_data[n - 1], 0.0);
    }

    // Find index i such that x[i] <= target_x < x[i+1].
    let mut low = 0usize;
    let mut high = n;
    while low < high {
        let mid = low + (high - low) / 2;
        if x_data[mid] <= target_x {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let i = low.saturating_sub(1).min(n - 2);

    let h = x_data[i + 1] - x_data[i];
    if h.abs() < 1e-9 {
        return (y_data[i], tangents[i]);
    }

    let t = (target_x - x_data[i]) / h;
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    let y_k = y_data[i];
    let y_k1 = y_data[i + 1];
    let m_k = tangents[i];
    let m_k1 = tangents[i + 1];

    let val = h00 * y_k + h10 * h * m_k + h01 * y_k1 + h11 * h * m_k1;

    let d_h00 = 6.0 * t2 - 6.0 * t;
    let d_h10 = 3.0 * t2 - 4.0 * t + 1.0;
    let d_h01 = -6.0 * t2 + 6.0 * t;
    let d_h11 = 3.0 * t2 - 2.0 * t;

    let grad = (d_h00 * y_k + d_h10 * h * m_k + d_h01 * y_k1 + d_h11 * h * m_k1) / h;

    (val, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uom::DIM_DIMENSIONLESS;
    use crate::uom::units::{dimensionless, kg_per_m3};

    #[test]
    fn interpolate_linear_midpoint() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![0.0 * kg_per_m3, 10.0 * kg_per_m3],
        );
        let (y, grad) = curve.interpolate(0.5 * dimensionless);
        assert!((y.value_in(kg_per_m3) - 5.0).abs() < 1e-12);
        assert!((grad.value_in(kg_per_m3) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn reverse_interpolate_midpoint() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![0.0 * kg_per_m3, 10.0 * kg_per_m3],
        );
        let (x, grad) = curve.reverse_interpolate(5.0 * kg_per_m3);
        assert!((x.value() - 0.5).abs() < 1e-12);
        assert!((grad.value_in(kg_per_m3) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn interpolate_below_min_returns_first_point() {
        let curve = Curve::new(
            vec![0.2 * dimensionless, 1.0 * dimensionless],
            vec![2.0 * kg_per_m3, 10.0 * kg_per_m3],
        );
        let (y, grad) = curve.interpolate(0.0 * dimensionless);
        assert!(
            (y.value_in(kg_per_m3) - 2.0).abs() < 1e-12,
            "Assert y value {:.2} to be 2.0",
            y.value_in(kg_per_m3)
        );
        assert!(
            grad.value_in(kg_per_m3).abs() < 1e-12,
            "Assert grad value {:.2} to be 0.0",
            grad.value_in(kg_per_m3)
        );
    }

    #[test]
    fn interpolate_above_max_returns_last_point() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 0.8 * dimensionless],
            vec![1.0 * kg_per_m3, 5.0 * kg_per_m3],
        );
        let (y, grad) = curve.interpolate(1.0 * dimensionless);
        assert!(
            (y.value_in(kg_per_m3) - 5.0).abs() < 1e-12,
            "Assert y value {:.2} to be 5.0",
            y.value_in(kg_per_m3)
        );
        assert!(
            grad.value_in(kg_per_m3).abs() < 1e-12,
            "Assert grad value {:.2} to be 0.0",
            grad.value_in(kg_per_m3)
        );
    }

    #[test]
    fn reverse_interpolate_below_min_returns_first_point() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![2.0 * kg_per_m3, 6.0 * kg_per_m3],
        );
        let (x, grad) = curve.reverse_interpolate(1.0 * kg_per_m3);
        assert!(
            (x.value() - 0.0).abs() < 1e-12,
            "Assert x value {:.2} to be 1.0",
            x.value()
        );
        assert!(
            grad.value_in(kg_per_m3).abs() < 1e-12,
            "Assert grad value {:.2} to be 0.0",
            grad.value_in(kg_per_m3)
        );
    }

    #[test]
    fn reverse_interpolate_above_max_returns_last_point() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![2.0 * kg_per_m3, 6.0 * kg_per_m3],
        );
        let (x, grad) = curve.reverse_interpolate(10.0 * kg_per_m3);
        assert!((x.value() - 1.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    #[test]
    fn interpolate_nan_input_returns_first_point_without_panic() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 0.8 * dimensionless],
            vec![1.0 * kg_per_m3, 5.0 * kg_per_m3],
        );
        let (y, grad) = curve.interpolate(f64::NAN * dimensionless);
        assert!((y.value_in(kg_per_m3) - 1.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    #[test]
    fn reverse_interpolate_nan_input_returns_first_point_without_panic() {
        let curve = Curve::new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![2.0 * kg_per_m3, 6.0 * kg_per_m3],
        );
        let (x, grad) = curve.reverse_interpolate(f64::NAN * kg_per_m3);
        assert!((x.value() - 0.0).abs() < 1e-12);
        assert!(grad.value_in(kg_per_m3).abs() < 1e-12);
    }

    // ---------- LogY behavioural tests ----------

    #[test]
    fn logy_interpolate_midpoint_is_geometric_mean() {
        // Two-point LogY curve: PCHIP on (x, ln y) reduces to a straight line
        // through (1, ln 1e-13) and (2, ln 1e-7), so y at x=1.5 is the
        // geometric mean = sqrt(1e-13 * 1e-7) = 1e-10.
        let logy = Curve::<_, _, LogY>::try_new(
            vec![1.0 * dimensionless, 2.0 * dimensionless],
            vec![1.0e-13 * kg_per_m3, 1.0e-7 * kg_per_m3],
        )
        .expect("valid LogY input");
        let (y_log, _) = logy.interpolate(1.5 * dimensionless);
        let y_log_val = y_log.value_in(kg_per_m3);
        let geom_mean = 1e-10_f64;
        assert!(
            ((y_log_val - geom_mean).abs() / geom_mean) < 0.01,
            "LogY midpoint {:.3e} should be ~geometric mean {:.3e}",
            y_log_val,
            geom_mean
        );

        // Sanity: LinearY on the same data gives ~arithmetic mean
        // (≈ 5e-8), three orders of magnitude away.
        let lin = Curve::new(
            vec![1.0 * dimensionless, 2.0 * dimensionless],
            vec![1.0e-13 * kg_per_m3, 1.0e-7 * kg_per_m3],
        );
        let (y_lin, _) = lin.interpolate(1.5 * dimensionless);
        let y_lin_val = y_lin.value_in(kg_per_m3);
        let arith_mean = 0.5 * (1.0e-13 + 1.0e-7);
        assert!(
            ((y_lin_val - arith_mean).abs() / arith_mean) < 0.01,
            "LinearY midpoint {:.3e} should be ~arithmetic mean {:.3e}",
            y_lin_val,
            arith_mean
        );
        // And the two scales must differ by orders of magnitude.
        assert!(
            y_lin_val / y_log_val > 100.0,
            "LinearY ({:.3e}) and LogY ({:.3e}) midpoints should differ \
             by orders of magnitude",
            y_lin_val,
            y_log_val
        );
    }

    #[test]
    fn logy_gradient_chain_rule() {
        // At a two-point PCHIP node the tangent equals the secant in internal
        // space, so at the midpoint dt/dx = (ln y2 - ln y1)/(x2 - x1).
        // Chain rule: dy/dx = y * dt/dx.
        let y1 = 1.0e-13_f64;
        let y2 = 1.0e-7_f64;
        let x1 = 1.0_f64;
        let x2 = 2.0_f64;
        let logy = Curve::<_, _, LogY>::try_new(
            vec![x1 * dimensionless, x2 * dimensionless],
            vec![y1 * kg_per_m3, y2 * kg_per_m3],
        )
        .expect("valid LogY input");
        let (y_mid, grad) = logy.interpolate(1.5 * dimensionless);
        let y_mid_val = y_mid.value_in(kg_per_m3);
        let dt_dx = (y2.ln() - y1.ln()) / (x2 - x1);
        let expected_grad = y_mid_val * dt_dx;
        let actual_grad = grad.value_in(kg_per_m3);
        let rel = (actual_grad - expected_grad).abs() / expected_grad.abs();
        assert!(
            rel < 1e-9,
            "LogY chain-rule gradient mismatch: got {:.4e}, expected {:.4e} (rel {:.2e})",
            actual_grad,
            expected_grad,
            rel
        );
    }

    #[test]
    fn logy_zero_pinning_below_threshold() {
        let logy = Curve::<_, _, LogY>::try_new(
            vec![
                0.0 * dimensionless,
                1.0 * dimensionless,
                2.0 * dimensionless,
            ],
            vec![0.0 * kg_per_m3, 1.0e-10 * kg_per_m3, 1.0e-7 * kg_per_m3],
        )
        .expect("valid LogY input with leading zero");

        // Below threshold (threshold = x of first non-zero = 1.0).
        let (y, grad) = logy.interpolate(0.5 * dimensionless);
        assert_eq!(y.value_in(kg_per_m3), 0.0);
        assert_eq!(grad.value_in(kg_per_m3), 0.0);

        // Above threshold: PCHIP in log space, must be strictly positive.
        let (y, _) = logy.interpolate(1.5 * dimensionless);
        assert!(y.value_in(kg_per_m3) > 0.0);
    }

    #[test]
    fn logy_rejects_interior_zero() {
        let err = Curve::<_, _, LogY>::try_new(
            vec![
                0.0 * dimensionless,
                1.0 * dimensionless,
                2.0 * dimensionless,
                3.0 * dimensionless,
            ],
            vec![
                0.0 * kg_per_m3,
                1.0e-10 * kg_per_m3,
                0.0 * kg_per_m3,
                1.0e-7 * kg_per_m3,
            ],
        )
        .expect_err("mid-table zero should be rejected");
        match err {
            CurveError::MidTableZero { row } => assert_eq!(row, 2),
            other => panic!("expected MidTableZero, got {:?}", other),
        }
    }

    #[test]
    fn logy_all_zero_input_is_no_transport_curve() {
        // WUFI convention: an all-zero D_w table declares "no liquid
        // transport for this material" (vapour foils, non-hygroscopic
        // insulations). The curve must construct cleanly and return
        // (0, 0) at every x — including outside the table's x range.
        for (xs, ys) in [
            // Multi-row all-zero (e.g. PE_Folie Suction with one entry).
            (
                vec![0.0 * dimensionless, 1.0 * dimensionless, 2.0 * dimensionless],
                vec![0.0 * kg_per_m3, 0.0 * kg_per_m3, 0.0 * kg_per_m3],
            ),
            // Single-row all-zero (the IDF "1 entry" case that triggered the bug).
            (vec![0.0 * dimensionless], vec![0.0 * kg_per_m3]),
        ] {
            let curve = Curve::<_, _, LogY>::try_new(xs, ys)
                .expect("all-zero LogY is the WUFI no-transport convention");
            for x in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
                let (y, grad) = curve.interpolate(x * dimensionless);
                assert_eq!(y.value_in(kg_per_m3), 0.0, "y != 0 at x = {x}");
                assert_eq!(grad.value_in(kg_per_m3), 0.0, "grad != 0 at x = {x}");
            }
        }
    }

    #[test]
    fn logy_rejects_negative_y() {
        let err = Curve::<_, _, LogY>::try_new(
            vec![0.0 * dimensionless, 1.0 * dimensionless],
            vec![-1.0 * kg_per_m3, 1.0e-7 * kg_per_m3],
        )
        .expect_err("negative y should be rejected");
        match err {
            CurveError::NegativeY { row } => assert_eq!(row, 0),
            other => panic!("expected NegativeY, got {:?}", other),
        }
    }

    #[test]
    fn linear_y_curve_omits_logy_threshold_storage() {
        // The whole point of S::Extra: LinearY curves don't pay for LogY's
        // threshold field. With `LinearY::Extra = ()` the struct is strictly
        // smaller than the `Option<f64>`-bearing LogY variant. Guards against
        // an accidental regression where the scale's extra data leaks into
        // every curve.
        use std::mem::size_of;
        assert!(
            size_of::<Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS, LinearY>>()
                < size_of::<Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS, LogY>>(),
            "LinearY curve should be strictly smaller than LogY curve"
        );
    }

    #[test]
    fn default_param_resolves_to_linear_y() {
        // Curve<X, Y> and Curve<X, Y, LinearY> are the same type — locks in
        // that the default scale parameter is LinearY.
        use std::mem::size_of;
        assert_eq!(
            size_of::<Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS>>(),
            size_of::<Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS, LinearY>>(),
        );
    }

    #[test]
    fn default_logy_curve_is_empty_and_safe() {
        // Material::default() builds empty LogY curves for suction and
        // redistribution. Interpolating on those must return (0, 0) without
        // attempting ln() on anything.
        let curve: Curve<DIM_DIMENSIONLESS, { crate::uom::DIM_WATER_CONTENT }, LogY> =
            Curve::default();
        let (y, grad) = curve.interpolate(0.5 * dimensionless);
        assert_eq!(y.value_in(kg_per_m3), 0.0);
        assert_eq!(grad.value_in(kg_per_m3), 0.0);
    }
}
