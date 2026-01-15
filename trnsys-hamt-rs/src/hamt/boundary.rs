/// Boundary conditions for the simulation.
#[derive(Debug, Clone)]
pub struct BoundaryCondition {
    /// Temperature of the outside air [°C].
    pub temp_outside_air: f64,
    /// Temperature of the inside air [°C].
    pub temp_inside_air: f64,
    /// Relative humidity of the outside air [-].
    pub rh_outside_air: f64,
    /// Relative humidity of the inside air [-].
    pub rh_inside_air: f64,
    /// Pressure of the air [Pa].
    pub pressure_air: f64,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition {
            temp_outside_air: 20.0,
            temp_inside_air: 20.0,
            rh_outside_air: 0.5,
            rh_inside_air: 0.5,
            pressure_air: 101325.0,
        }
    }
}

impl BoundaryCondition {
    pub fn new(
        temp_outside_air: f64,
        temp_inside_air: f64,
        rh_outside_air: f64,
        rh_inside_air: f64,
        pressure_air: f64,
    ) -> Self {
        BoundaryCondition {
            temp_outside_air,
            temp_inside_air,
            rh_outside_air,
            rh_inside_air,
            pressure_air,
        }
    }
}
