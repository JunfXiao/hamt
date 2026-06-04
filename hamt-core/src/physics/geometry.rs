use std::ops;

use tracing::error;

use crate::uom::dimensions::{Area, Length, Volume};

/// A point or vector in 3D space.
/// ## Coordinates:
/// - `x`: The surface depth/thickness direction.
/// - `y`: The surface width direction.
/// - `z`: The surface height direction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pt3D(pub Length<f64>, pub Length<f64>, pub Length<f64>);

/// Size3 is a type alias for Pt, representing a 3D size.
pub type Size3D = Pt3D;

impl Pt3D {
    pub fn new(x: Length<f64>, y: Length<f64>, z: Length<f64>) -> Self {
        Pt3D(x, y, z)
    }

    pub fn x(&self) -> Length<f64> {
        self.0
    }
    pub fn y(&self) -> Length<f64> {
        self.1
    }
    pub fn z(&self) -> Length<f64> {
        self.2
    }

    pub fn area_along(&self, axis: i32) -> Area<f64> {
        match axis {
            0 => self.1 * self.2,
            1 => self.0 * self.2,
            2 => self.0 * self.1,
            _ => {
                error!("Invalid axis for area: {}", axis);
                Area::<f64>::zero()
            }
        }
    }

    /// Return the extent (size component) along the given axis.
    ///
    /// - axis 0 → x (depth / thickness direction)
    /// - axis 1 → y (width direction)
    /// - axis 2 → z (height direction)
    ///
    /// Mirrors [`area_along`]; used by the axis-aware resistance helpers
    /// so half-distances are computed along the transport direction.
    pub fn extent_along(&self, axis: usize) -> Length<f64> {
        match axis {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            _ => {
                error!("Invalid axis for extent: {}", axis);
                Length::<f64>::zero()
            }
        }
    }

    pub fn volume(&self) -> Volume<f64> {
        self.0 * self.1 * self.2
    }
}

impl ops::Add for Pt3D {
    type Output = Pt3D;

    fn add(self, other: Pt3D) -> Pt3D {
        Pt3D(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl ops::AddAssign for Pt3D {
    fn add_assign(&mut self, other: Pt3D) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
    }
}

impl ops::Sub for Pt3D {
    type Output = Pt3D;

    fn sub(self, other: Pt3D) -> Pt3D {
        Pt3D(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl ops::SubAssign for Pt3D {
    fn sub_assign(&mut self, other: Pt3D) {
        self.0 -= other.0;
        self.1 -= other.1;
        self.2 -= other.2;
    }
}

impl ops::Mul<f64> for Pt3D {
    type Output = Pt3D;

    fn mul(self, scalar: f64) -> Pt3D {
        Pt3D(self.0 * scalar, self.1 * scalar, self.2 * scalar)
    }
}

impl ops::MulAssign<f64> for Pt3D {
    fn mul_assign(&mut self, scalar: f64) {
        self.0 *= scalar;
        self.1 *= scalar;
        self.2 *= scalar;
    }
}

impl ops::Div<f64> for Pt3D {
    type Output = Pt3D;

    fn div(self, scalar: f64) -> Pt3D {
        Pt3D(self.0 / scalar, self.1 / scalar, self.2 / scalar)
    }
}

impl ops::DivAssign<f64> for Pt3D {
    fn div_assign(&mut self, scalar: f64) {
        self.0 /= scalar;
        self.1 /= scalar;
        self.2 /= scalar;
    }
}

impl ops::Neg for Pt3D {
    type Output = Pt3D;

    fn neg(self) -> Pt3D {
        Pt3D(-self.0, -self.1, -self.2)
    }
}
