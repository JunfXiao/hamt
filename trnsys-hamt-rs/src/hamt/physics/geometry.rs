use std::ops;

use tracing::error;

/// A point or vector in 3D space.
/// ## Coordinates:
/// - `x`: The surface depth/thickness direction.
/// - `y`: The surface width direction.
/// - `z`: The surface height direction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pt3D(pub f64, pub f64, pub f64);

/// Size3 is a type alias for Pt, representing a 3D size.
pub type Size3D = Pt3D;

impl Pt3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Pt3D(x, y, z)
    }

    pub fn x(&self) -> f64 {
        self.0
    }
    pub fn y(&self) -> f64 {
        self.1
    }
    pub fn z(&self) -> f64 {
        self.2
    }

    pub fn area_along(&self, axis: i32) -> f64 {
        match axis {
            0 => self.1 * self.2,
            1 => self.0 * self.2,
            2 => self.0 * self.1,
            _ => {
                error!("Invalid axis for area: {}", axis);
                0.0
            }
        }
    }

    pub fn volume(&self) -> f64 {
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
