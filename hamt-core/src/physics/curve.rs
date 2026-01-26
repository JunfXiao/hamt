//! Defines the data structures for materials and constructions used in the HAMT model.

use std::fmt::Debug;

use crate::uom::{Dimension, Quantity};

/// Represents a 2D data curve, typically from material property measurements.
///
/// This is used to store properties that vary with a single independent variable,
/// such as the sorption isotherm (water content vs. RH) or thermal conductivity
/// vs. water content. The `interp` function in the `physics` module will operate
/// on this struct.
#[derive(Debug, Clone, Default)]
pub struct Curve<const DX: Dimension, const DY: Dimension>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    pub x: Vec<Quantity<f64, DX>>,
    pub y: Vec<Quantity<f64, DY>>,
    /// Cached tangents (derivatives dy/dx) calculated using the PCHIP algorithm.
    /// These are computed once during creation to ensure efficient interpolation.
    pub tangents: Vec<Quantity<f64, { DY.sub(DX) }>>,
}

impl<const DX: Dimension, const DY: Dimension> Curve<DX, DY>
where
    Quantity<f64, { DY.sub(DX) }>: Sized,
{
    /// Creates a new curve and pre-calculates PCHIP tangents.
    pub fn new(x_series: Vec<Quantity<f64, DX>>, y_series: Vec<Quantity<f64, DY>>) -> Self {
        let tangents = Self::calculate_pchip_tangents(&x_series, &y_series);
        Self {
            x: x_series,
            y: y_series,
            tangents,
        }
    }

    /// Calculates derivatives (tangents) using the PCHIP (Monotone Cubic Hermite) algorithm.
    /// This ensures that the interpolation remains monotonic (no overshoots) if the data is monotonic.
    fn calculate_pchip_tangents(
        x: &[Quantity<f64, DX>],
        y: &[Quantity<f64, DY>],
    ) -> Vec<Quantity<f64, { DY.sub(DX) }>> {
        let n = x.len();
        if n < 2 {
            return vec![Quantity::<f64, { DY.sub(DX) }>::zero(); n];
        }

        let mut tangents = Vec::with_capacity(n);
        // Calculate secant slopes (linear slopes between points)
        let mut slopes = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dx = x[i + 1].0 - x[i].0;
            let dy = y[i + 1].0 - y[i].0;
            if dx.abs() < 1e-9 {
                slopes.push(0.0);
            } else {
                slopes.push(dy / dx);
            }
        }

        // --- Start Point ---
        // Use one-sided 3-point difference for the boundary, or linear slope if n=2
        let m0 = if n == 2 {
            slopes[0]
        } else {
            // Non-centered 3-point formula for end point
            let h0 = x[1].0 - x[0].0;
            let h1 = x[2].0 - x[1].0;
            let d0 = ((2.0 * h0 + h1) * slopes[0] - h0 * slopes[1]) / (h0 + h1);
            // Sign check to preserve monotonicity at the boundary
            if d0 * slopes[0] <= 0.0 {
                0.0
            } else if (slopes[0] * slopes[1] <= 0.0) && (d0.abs() > 3.0 * slopes[0].abs()) {
                3.0 * slopes[0]
            } else {
                d0
            }
        };
        tangents.push(Quantity::<f64, { DY.sub(DX) }>(m0));

        // --- Internal Points ---
        for i in 0..n - 2 {
            let s0 = slopes[i];
            let s1 = slopes[i + 1];

            // If slopes have different signs, derivative must be 0 to enforce monotonicity
            if s0 * s1 <= 0.0 {
                tangents.push(Quantity::<f64, { DY.sub(DX) }>::zero());
            } else {
                // Weighted harmonic mean
                let h0 = x[i + 1].0 - x[i].0;
                let h1 = x[i + 2].0 - x[i + 1].0;
                let w1 = 2.0 * h1 + h0;
                let w2 = h1 + 2.0 * h0;
                let harmonic_mean = (w1 + w2) / (w1 / s0 + w2 / s1);
                tangents.push(Quantity::<f64, { DY.sub(DX) }>(harmonic_mean));
            }
        }

        // --- End Point ---
        let m_last = if n == 2 {
            slopes[0]
        } else {
            let h_n_2 = x[n - 2].0 - x[n - 3].0;
            let h_n_1 = x[n - 1].0 - x[n - 2].0;
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
        tangents.push(Quantity::<f64, { DY.sub(DX) }>(m_last));

        tangents
    }
    /// Public static method to perform PCHIP interpolation on raw data arrays.
    ///
    /// # Arguments
    /// * `target_x` - The X value to interpolate at.
    /// * `x_data` - The array of X coordinates (must be sorted).
    /// * `y_data` - The array of Y coordinates.
    /// * `tangents` - The array of derivatives at each point (must match length of x/y).
    ///
    /// # Returns
    /// A tuple containing (Interpolated Y, Gradient dY/dX).
    pub fn interpolate_pchip_static(
        target_x: Quantity<f64, DX>,
        x_data: &[Quantity<f64, DX>],
        y_data: &[Quantity<f64, DY>],
        tangents: &[Quantity<f64, { DY.sub(DX) }>],
    ) -> (Quantity<f64, DY>, Quantity<f64, { DY.sub(DX) }>) {
        let zero_y = Quantity::<f64, DY>::zero();
        let zero_grad = Quantity::<f64, { DY.sub(DX) }>::zero();

        if x_data.is_empty() {
            return (zero_y, zero_grad);
        }

        let n = x_data.len();

        // Fix: Use strictly less/greater comparisons for constant extrapolation.
        // If strictly out of bounds, return the boundary Y value and ZERO gradient (flat line).
        if target_x < x_data[0] {
            return (y_data[0], zero_grad);
        }
        if target_x > x_data[n - 1] {
            return (y_data[n - 1], zero_grad);
        }

        // Binary search to find the interval [x_i, x_{i+1}]
        // Handle exact boundary matches or values within range
        // If target_x == x_data[0] or target_x == x_data[n-1], the logic below correctly
        // uses the PCHIP formula to return the boundary value and the boundary tangent.

        // Find index i such that x[i] <= target_x < x[i+1]
        let mut i = 0;
        let mut low = 0;
        let mut high = n - 1;

        while low <= high {
            let mid = (low + high) / 2;
            if x_data[mid] <= target_x {
                i = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        // Ensure i is not the last element (if target_x is exactly x_data[n-1])
        if i >= n - 1 {
            i = n - 2;
        }

        // Hermite Basis Functions
        let h = x_data[i + 1].0 - x_data[i].0;
        if h.abs() < 1e-9 {
            return (y_data[i], tangents[i]);
        } // Protect against zero-length segments

        let t = (target_x.0 - x_data[i].0) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let y_k = y_data[i].0;
        let y_k1 = y_data[i + 1].0;
        let m_k = tangents[i].0;
        let m_k1 = tangents[i + 1].0;

        // Interpolated Value
        let val = h00 * y_k + h10 * h * m_k + h01 * y_k1 + h11 * h * m_k1;

        // Derivative (dY/dX) = (dY/dt) * (dt/dX) = (dY/dt) / h
        let d_h00 = 6.0 * t2 - 6.0 * t;
        let d_h10 = 3.0 * t2 - 4.0 * t + 1.0;
        let d_h01 = -6.0 * t2 + 6.0 * t;
        let d_h11 = 3.0 * t2 - 2.0 * t;

        let grad = (d_h00 * y_k + d_h10 * h * m_k + d_h01 * y_k1 + d_h11 * h * m_k1) / h;

        (
            Quantity::<f64, DY>(val),
            Quantity::<f64, { DY.sub(DX) }>(grad),
        )
    }

    /// Interpolates a value from the curve using PCHIP.
    /// Returns the interpolated Y and the gradient (dY/dX).
    pub fn interpolate(
        &self,
        in_x: Quantity<f64, DX>,
    ) -> (Quantity<f64, DY>, Quantity<f64, { DY.sub(DX) }>) {
        Self::interpolate_pchip_static(in_x, &self.x, &self.y, &self.tangents)
    }

    /// Linearly interpolates a value from a curve (Reverse: Y -> X).
    /// Note: This remains linear as analytic inversion of cubic PCHIP is complex.
    /// Also optionally returns the gradient at the interpolation point.
    /// Returns the interpolated X and the gradient (dX/dY) as f64.
    pub fn reverse_interpolate(
        &self,
        in_y: Quantity<f64, DY>,
    ) -> (Quantity<f64, DX>, Quantity<f64, { DX.sub(DY) }>) {
        let zero_x = Quantity::<f64, DX>::zero();
        let zero_grad = Quantity::<f64, { DX.sub(DY) }>::zero();

        if self.y.is_empty() {
            return (zero_x, zero_grad);
        }

        let mut upper_index: Option<usize> = None;
        for (i, &yv) in self.y.iter().enumerate() {
            match yv.partial_cmp(&in_y) {
                Some(std::cmp::Ordering::Equal) => return (self.x[i], zero_grad),
                Some(std::cmp::Ordering::Greater) => {
                    upper_index = Some(i);
                    break;
                }
                _ => {}
            }
        }

        match upper_index {
            Some(0) => (self.x[0], zero_grad),
            Some(i) => {
                let y1 = self.y[i - 1];
                let y2 = self.y[i];
                let x1 = self.x[i - 1];
                let x2 = self.x[i];

                let grad = if (y2 - y1).abs() > Quantity::<f64, DY>(1e-9) {
                    (x2 - x1) / (y2 - y1)
                } else {
                    zero_grad
                };
                let val = x1.0 + grad.0 * (in_y.0 - y1.0);
                (Quantity::<f64, DX>(val), grad)
            }
            None => (*self.x.last().unwrap(), zero_grad),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
