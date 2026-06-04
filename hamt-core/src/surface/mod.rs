use anyhow::Ok;
use ndarray::ArcArray3;
use serde::Serialize;

use crate::cell::{BoundaryLocation, Cell, CellType};
use crate::physics::geometry::Size3D;
use crate::uom::dimensions::{
    Area, Density, Dimensionless, EnergyFluxDensity, Length, Mass, MassFluxDensity, Temperature,
    VaporTransportCoeff,
};
use crate::{boundary::BoundaryCondition, config::HamtSettings, physics};
use anyhow::Result;
pub mod boundary;
pub mod init;
pub mod query;
pub mod step_jacobi;
pub mod step_tridiag;
use std::{collections::HashMap, sync::Arc};

use crate::parallel::{par_or_seq, par_or_seq_mut};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Re-export step types so external code can use crate::surface::BlockJacobiStep etc.
pub use step_jacobi::{BlockJacobiStep, CellInertia, CellSources, NeighborSums};
pub use step_tridiag::BlockTridiagStep;

/// Trilinear interpolation weights: 8 corner cells with their indices and weights.
/// Weights sum to 1.0. For 1D grids (ny=nz=1) only the first two entries are
/// non-zero; unused entries have weight 0.0 and index `[0,0,0]`.
pub type InterpolationWeights = [([usize; 3], f64); 8];

/// Snapshot of on-demand surface reporting values.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SurfaceReport {
    pub temp_surf_out: Temperature<f64>,
    pub temp_surf_int: Temperature<f64>,
    pub rh_surf_out: Dimensionless<f64>,
    pub rh_surf_int: Dimensionless<f64>,
    pub temp_surf_user_def: Temperature<f64>,
    pub rh_surf_user_def: Dimensionless<f64>,
    pub water_surf_user_def: Density<f64>,
    pub q_surf_ext: EnergyFluxDensity<f64>,
    pub q_surf_int: EnergyFluxDensity<f64>,
    pub m_surf_ext: MassFluxDensity<f64>,
    pub m_surf_int: MassFluxDensity<f64>,
    pub water_total: Mass<f64>,
}

/// Holds geometry, cells, and utilities for a single surface.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Surface name.
    pub name: String,
    /// Name of the construction used by this surface.
    pub construction_name: String,

    /// Surface Size(Dimension) (Thickness, length, height) [m].
    pub size: Size3D,

    /// Surface area [m²].
    pub area: Area<f64>,

    /// Thickness of each cell layer along the x-axis [m].
    pub thickness: Length<f64>,

    /// Cell lattice.
    pub cells: ArcArray3<Cell>,

    // ========= Cell Index Helpers =========
    cells_idx_map: HashMap<CellType, Vec<[usize; 3]>>,

    // ========= Boundary-adjacent material cell indices =========
    /// Index (x-axis) of the outermost interior material cell (last Material cell in x).
    /// Radiation absorbed at the interior surface is deposited here.
    pub interior_surface_material_idx: Option<usize>,

    /// Index (x-axis) of the outermost exterior material cell (first Material cell in x).
    /// Radiation absorbed at the exterior surface is deposited here.
    pub exterior_surface_material_idx: Option<usize>,

    // ========= Optional user-defined probe =========
    /// Depth (m) at which user-defined reporting is performed.
    pub user_defined_depth: Option<Length<f64>>,

    pub user_defined_cell_idx: Option<InterpolationWeights>,

    /// Per-face boundary condition: [x-min, x-max, y-min, y-max, z-min, z-max].
    pub face_bcs: [crate::config::FaceBc; 6],
}

/// A single inner-iteration step of the nonlinear outer loop.
///
/// Both inner solvers (BlockJacobi and BlockTridiagonal) share an outer-loop
/// skeleton (step → convergence_stats → aitken.relax → swap → exit test). The
/// only thing that differs per iteration is *how* the next iterate is computed.
/// This trait abstracts that step so a single generic driver
/// (`HamtManager::run_outer_loop`) can host the shared skeleton.
///
/// Implementors compute the k+1 iterate into `next`, reading the physical
/// previous-timestep state from `surface_prev` (inertial term) and the current
/// iteration-k state from `surface_curr` (nonlinear coefficients / Jacobian).
pub trait IterativeStep {
    fn step(
        &self,
        next: &mut Surface,
        settings: &crate::config::HamtSettings,
        bc: &crate::boundary::BoundaryCondition,
        surface_prev: &Surface,
        surface_curr: &Surface,
    ) -> anyhow::Result<()>;
}

impl Surface {
    /// Copy the current iteration state into `surface_next` before property updates.
    pub fn sync_state_from(&mut self, surface_curr: &Surface) -> Result<()> {
        if let (Some(dst), Some(src)) = (self.cells.as_slice_mut(), surface_curr.cells.as_slice()) {
            par_or_seq_mut(dst)
                .zip(par_or_seq(src))
                .for_each(|(cell, ref_cell)| {
                    *cell = ref_cell.clone();
                    cell.sync_state_from(ref_cell);
                });
        } else {
            for (cell, ref_cell) in self.cells.iter_mut().zip(surface_curr.cells.iter()) {
                *cell = ref_cell.clone();
                cell.sync_state_from(ref_cell);
            }
        }
        Ok(())
    }

    /// Update dependent cell properties based on the current iteration's Temp and RH.
    pub fn update_cell_properties(
        &mut self,
        bc: &BoundaryCondition,
        settings: &HamtSettings,
    ) -> Result<()> {
        if let Some(cells) = self.cells.as_slice_mut() {
            par_or_seq_mut(cells).for_each(|cell| cell.calc_dependent_state(bc, settings));
        } else {
            for cell in self.cells.iter_mut() {
                cell.calc_dependent_state(bc, settings);
            }
        }
        Ok(())
    }

    /// Get neighbor indices for a cell along `axis`.
    ///
    /// For `axis == 0` this is the existing x-axis walk. The Virtual-cell
    /// gather (all-of-wall fan-out) is x-only by design; Phase 2 will
    /// generalize it for y/z once the multi-D cell topology is defined.
    pub(crate) fn neighbor_indices(
        &self,
        i: usize,
        n_along: usize,
        axis: usize,
    ) -> impl Iterator<Item = usize> {
        debug_assert!(
            axis == 0,
            "neighbor_indices: only axis 0 (x) is implemented; 3-D walks land in Phase 2"
        );
        let self_cell = self.cells.get([i, 0, 0]).unwrap();

        let left: Vec<usize> = if i > 0 {
            if matches!(
                self_cell.cell_type,
                CellType::Virtual(BoundaryLocation::Exterior)
            ) {
                (0..i).collect()
            } else if i > 1 {
                vec![i - 1]
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        let right = if matches!(
            self_cell.cell_type,
            CellType::Virtual(BoundaryLocation::Interior)
        ) {
            (i + 1..n_along).collect()
        } else if i + 1 < n_along {
            vec![i + 1]
        } else {
            vec![]
        };

        left.into_iter().chain(right)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn one_d_surface_has_active_x_faces_adiabatic_sides() {
        use crate::config::{FaceBc, FACE_X_MIN, FACE_X_MAX, FACE_Y_MIN, FACE_Z_MAX};
        let mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let f = mgr.surface_curr.face_bcs;
        assert_eq!(f[FACE_X_MIN], FaceBc::Active);
        assert_eq!(f[FACE_X_MAX], FaceBc::Active);
        assert_eq!(f[FACE_Y_MIN], FaceBc::Adiabatic);
        assert_eq!(f[FACE_Z_MAX], FaceBc::Adiabatic);
    }
}
