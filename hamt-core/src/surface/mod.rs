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
pub mod grid;
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
    /// For a multi-column (ny>1) grid this is the first element of
    /// `interior_surface_material_idx_set`; kept for backward-compatibility with
    /// callers that need only a single x-index.
    pub interior_surface_material_idx: Option<usize>,

    /// Index (x-axis) of the outermost exterior material cell (first Material cell in x).
    /// Radiation absorbed at the exterior surface is deposited here.
    /// For a multi-column (ny>1) grid this is the first element of
    /// `exterior_surface_material_idx_set`; kept for backward-compatibility.
    pub exterior_surface_material_idx: Option<usize>,

    /// Interior-most material cell per y/z column (the interior surface sheet).
    /// For a 1-D wall (ny=1, nz=1) this contains exactly one element `[x,0,0]`
    /// whose `x` equals `interior_surface_material_idx.unwrap()`.
    pub interior_surface_material_idx_set: Vec<[usize; 3]>,

    /// Exterior-most material cell per y/z column (the exterior surface sheet).
    /// For a 1-D wall (ny=1, nz=1) this contains exactly one element `[x,0,0]`
    /// whose `x` equals `exterior_surface_material_idx.unwrap()`.
    pub exterior_surface_material_idx_set: Vec<[usize; 3]>,

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

    /// In-bounds neighbours of `idx` along ONE axis (0=x, 1=y, 2=z), as [i,j,k].
    ///
    /// * **Axis 0 (x):** replicates the existing 1-D walk with full Virtual-cell
    ///   fan-out semantics:
    ///   - `Virtual(Exterior)` fans to ALL lower-x cells `[0..i, j, k]`.
    ///   - `Virtual(Interior)` fans to ALL higher-x cells `[i+1..nx, j, k]`.
    ///   - Normal material/boundary cells: the ±1 x-neighbours (if in-bounds).
    /// * **Axes 1 and 2 (y, z):** simple ±1 step; no Virtual fan-out. A step is
    ///   omitted if it goes out-of-bounds **or** if the corresponding face's
    ///   `face_bcs` is `Adiabatic`.  For a 1-D wall (ny=1, nz=1) the dims
    ///   check naturally yields no neighbours here without ever touching face_bcs.
    pub(crate) fn axis_neighbors(&self, idx: [usize; 3], axis: usize) -> Vec<[usize; 3]> {
        use crate::config::{FaceBc, FACE_Y_MAX, FACE_Y_MIN, FACE_Z_MAX, FACE_Z_MIN};

        let dims = self.cells.raw_dim();
        let [i, j, k] = idx;

        if axis == 0 {
            // x-axis: replicate original Virtual fan-out logic.
            let nx = dims[0];
            let self_cell = &self.cells[[i, j, k]];
            let mut nbrs: Vec<[usize; 3]> = Vec::new();

            // Left (lower-x) neighbours.
            if i > 0 {
                if matches!(
                    self_cell.cell_type,
                    CellType::Virtual(BoundaryLocation::Exterior)
                ) {
                    for xi in 0..i {
                        nbrs.push([xi, j, k]);
                    }
                } else if i > 1 {
                    nbrs.push([i - 1, j, k]);
                }
            }

            // Right (higher-x) neighbours.
            if matches!(
                self_cell.cell_type,
                CellType::Virtual(BoundaryLocation::Interior)
            ) {
                for xi in (i + 1)..nx {
                    nbrs.push([xi, j, k]);
                }
            } else if i + 1 < nx {
                nbrs.push([i + 1, j, k]);
            }

            nbrs
        } else if axis == 1 {
            // y-axis: ±1 step, gated by face_bcs and bounds.
            let ny = dims[1];
            let mut nbrs: Vec<[usize; 3]> = Vec::new();

            // j-1 (y-min face).
            if j > 0 && self.face_bcs[FACE_Y_MIN] != FaceBc::Adiabatic {
                nbrs.push([i, j - 1, k]);
            }
            // j+1 (y-max face).
            if j + 1 < ny && self.face_bcs[FACE_Y_MAX] != FaceBc::Adiabatic {
                nbrs.push([i, j + 1, k]);
            }

            nbrs
        } else {
            // axis == 2: z-axis: ±1 step, gated by face_bcs and bounds.
            let nz = dims[2];
            let mut nbrs: Vec<[usize; 3]> = Vec::new();

            // k-1 (z-min face).
            if k > 0 && self.face_bcs[FACE_Z_MIN] != FaceBc::Adiabatic {
                nbrs.push([i, j, k - 1]);
            }
            // k+1 (z-max face).
            if k + 1 < nz && self.face_bcs[FACE_Z_MAX] != FaceBc::Adiabatic {
                nbrs.push([i, j, k + 1]);
            }

            nbrs
        }
    }

    /// ALL in-bounds face neighbours of `idx` across every spatial axis (x always;
    /// y and z when their dimension > 1), each tagged with the axis it lies on.
    ///
    /// This is the union of `axis_neighbors` over every active axis and is used
    /// by `accumulate_neighbor_sums` to accumulate the implicit diagonal
    /// conductance sums for multi-dimensional stencils.
    ///
    /// For a 1-D wall (ny=1, nz=1) the result is identical to
    /// `axis_neighbors(idx, 0)` mapped to `(nbr, 0)`.
    pub(crate) fn all_face_neighbors(&self, idx: [usize; 3]) -> Vec<([usize; 3], usize)> {
        let dims = self.cells.raw_dim();
        let mut result: Vec<([usize; 3], usize)> = Vec::new();

        // Axis 0 (x) is always included.
        for nbr in self.axis_neighbors(idx, 0) {
            result.push((nbr, 0));
        }
        // Axis 1 (y) only when ny > 1.
        if dims[1] > 1 {
            for nbr in self.axis_neighbors(idx, 1) {
                result.push((nbr, 1));
            }
        }
        // Axis 2 (z) only when nz > 1.
        if dims[2] > 1 {
            for nbr in self.axis_neighbors(idx, 2) {
                result.push((nbr, 2));
            }
        }

        result
    }

    /// Get neighbor indices for a cell along `axis`.
    ///
    /// **Thin shim** — delegates to [`Self::axis_neighbors`] and strips the
    /// `[i,j,k]` index back to a flat `usize` for the existing 1-D callers
    /// (`assemble_1d`, the BlockTridiag virtual-cell refresh). On a 1-D wall
    /// (ny=1, nz=1) the behaviour is byte-identical to the previous
    /// implementation. Do **not** delete this method; callers outside this
    /// module rely on it.
    pub(crate) fn neighbor_indices(
        &self,
        i: usize,
        n_along: usize,
        axis: usize,
    ) -> impl Iterator<Item = usize> {
        let _ = n_along; // retained in signature for call-site compatibility
        self.axis_neighbors([i, 0, 0], axis)
            .into_iter()
            .map(|idx| idx[0])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn one_d_surface_has_active_x_faces_adiabatic_sides() {
        use crate::config::{FaceBc, FACE_X_MAX, FACE_X_MIN, FACE_Y_MIN, FACE_Z_MAX};
        let mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let f = mgr.surface_curr.face_bcs;
        assert_eq!(f[FACE_X_MIN], FaceBc::Active);
        assert_eq!(f[FACE_X_MAX], FaceBc::Active);
        assert_eq!(f[FACE_Y_MIN], FaceBc::Adiabatic);
        assert_eq!(f[FACE_Z_MAX], FaceBc::Adiabatic);
    }

    /// Pin (N,1,1) byte-identity: `axis_neighbors([i,0,0], 0)` must match the
    /// old `neighbor_indices`-equivalent walk for every cell index.
    ///
    /// The expected set is computed inline, mirroring the pre-refactor logic:
    /// * Virtual(Exterior) → `0..i`
    /// * Virtual(Interior) → `i+1..nx`
    /// * Normal cell: `i-1` if `i>1`, `i+1` if `i+1 < nx`
    ///   (Note: `i>1` mirrors the original guard; index 1 is a boundary cell
    ///    that is reached only via the boundary cell's own neighbour walk, not
    ///    as a left-neighbour of the cell at index 2.)
    #[test]
    fn stencil_1d_matches_old_neighbor_walk() {
        use crate::cell::{BoundaryLocation, CellType};

        let mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let surf = &mgr.surface_curr;
        let nx = surf.cells.raw_dim()[0];

        for i in 0..nx {
            let cell = &surf.cells[[i, 0, 0]];

            // ── Compute expected set (inline replication of the old logic) ──
            let mut expected: Vec<usize> = Vec::new();

            // Left (lower-x)
            if i > 0 {
                if matches!(
                    cell.cell_type,
                    CellType::Virtual(BoundaryLocation::Exterior)
                ) {
                    expected.extend(0..i);
                } else if i > 1 {
                    expected.push(i - 1);
                }
            }

            // Right (higher-x)
            if matches!(
                cell.cell_type,
                CellType::Virtual(BoundaryLocation::Interior)
            ) {
                expected.extend((i + 1)..nx);
            } else if i + 1 < nx {
                expected.push(i + 1);
            }

            // ── Actual from axis_neighbors ──
            let actual: Vec<usize> = surf
                .axis_neighbors([i, 0, 0], 0)
                .iter()
                .map(|idx| idx[0])
                .collect();

            assert_eq!(
                actual, expected,
                "axis_neighbors([{i},0,0], 0) mismatch: got {actual:?}, expected {expected:?}"
            );
        }
    }

    // NOTE: genuine 2-D stencil verified in T6/T9/T10 (2-D grid generator not
    // yet available in Task 3; hand-rolling a full Surface literal here would be
    // fragile).

    /// For a (N,1,1) 1-D wall the interior/exterior sets must each contain
    /// exactly one element, and that element's x-index must equal the scalar
    /// `interior_surface_material_idx` / `exterior_surface_material_idx`.
    #[test]
    fn interior_set_is_single_cell_in_1d() {
        let mgr = crate::solver::test_fixtures::build_fixture_manager(6);
        let surf = &mgr.surface_curr;

        // Interior set: single element whose [0] == scalar idx.
        let set = &surf.interior_surface_material_idx_set;
        assert_eq!(set.len(), 1, "1-D wall must have exactly one interior sheet cell");
        assert_eq!(
            Some(set[0][0]),
            surf.interior_surface_material_idx,
            "set element x-index must equal scalar interior_surface_material_idx"
        );

        // Exterior set: same invariant.
        let eset = &surf.exterior_surface_material_idx_set;
        assert_eq!(eset.len(), 1, "1-D wall must have exactly one exterior sheet cell");
        assert_eq!(
            Some(eset[0][0]),
            surf.exterior_surface_material_idx,
            "set element x-index must equal scalar exterior_surface_material_idx"
        );
    }

    /// For a (nx, ny=4, 1) 2-D surface the interior/exterior sets must each
    /// contain exactly `ny` elements — one per y-column.
    #[test]
    fn interior_set_has_ny_entries_in_2d() {
        use super::Surface;
        use crate::boundary::{BoundaryCondition, celsius_to_kelvin};
        use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
        use crate::physics::{Curve, LogY};
        use crate::uom::units::{
            dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, meter, square_meter, w_per_m_k,
        };
        use crate::uom::{
            DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY,
            DIM_WATER_CONTENT, Quantity,
        };
        use std::sync::Arc;

        let ny = 4usize;
        let n_div = 3usize;

        // ── Minimal material (same curves as build_fixture_manager) ──────────
        let w_sat = 146.0 * kg_per_m3;
        let n_pts = 20usize;
        let si_x: Vec<_> = (0..=n_pts)
            .map(|k| (k as f64 / n_pts as f64) * dimensionless)
            .collect();
        let si_y: Vec<_> = si_x.iter().map(|&rh| w_sat * rh.value().max(0.0).sqrt()).collect();
        let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
            Curve::new(si_x.clone(), si_y);
        let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
        let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
            Curve::new(si_x.clone(), mu_y);
        let wc_pts = 10usize;
        let wc_x: Vec<_> = (0..=wc_pts)
            .map(|k| (k as f64 / wc_pts as f64) * w_sat)
            .collect();
        let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
        let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
            Curve::new(wc_x.clone(), k_y);
        let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
        let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
            Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid Dw");

        let mat = Material {
            name: "2dMat".to_string(),
            thickness: 0.01 * meter,
            density: 200.0 * kg_per_m3,
            spec_heat: 1000.0 * j_per_kg_k,
            porosity: Quantity(0.146),
            initial_water_content: Quantity(0.073),
            sorption_isotherm,
            suction: liquid_tc.clone(),
            redistribution: liquid_tc,
            vapor_diffusion_resistance,
            thermal_conductivity,
            sorption_enthalpy: None,
            initial_temp: celsius_to_kelvin(20.0),
            num_div: Some(n_div),
            div_distribution_factor: 0.5,
        };

        let lib = MaterialLibrary::new();
        lib.add_material(mat).expect("add material");
        let sv = SurfaceVaporCoef {
            is_external_const: false,
            external_value: 1e-12,
            is_internal_const: false,
            internal_value: 1e-8,
        };
        lib.create_construction("2dConst".to_string(), vec!["2dMat".to_string()], sv)
            .expect("create construction");
        let construction = lib.get_construction("2dConst").unwrap();

        let settings = {
            let mut s = crate::config::HamtSettings::default();
            s.min_div = n_div as i32;
            s.max_div = n_div as i32;
            s
        };
        let bc = BoundaryCondition::default();
        let total_area = 4.0 * square_meter;
        let y_extent = 2.0 * meter; // ny=4 columns each 0.5 m tall

        let surf = Surface::from_construction_2d(
            &settings,
            Arc::clone(&construction),
            "2dSurface".to_string(),
            total_area,
            ny,
            y_extent,
            &bc,
        )
        .expect("from_construction_2d");

        assert_eq!(
            surf.interior_surface_material_idx_set.len(),
            ny,
            "interior set must have one entry per y-column (ny={ny})"
        );
        assert_eq!(
            surf.exterior_surface_material_idx_set.len(),
            ny,
            "exterior set must have one entry per y-column (ny={ny})"
        );

        // All elements in each set must refer to actual Material cells.
        use crate::cell::CellType;
        for &idx in &surf.interior_surface_material_idx_set {
            let cell = surf.cells.get(idx).expect("valid index");
            assert!(
                matches!(cell.cell_type, CellType::Material(_)),
                "interior set element {idx:?} is not a Material cell"
            );
        }
        for &idx in &surf.exterior_surface_material_idx_set {
            let cell = surf.cells.get(idx).expect("valid index");
            assert!(
                matches!(cell.cell_type, CellType::Material(_)),
                "exterior set element {idx:?} is not a Material cell"
            );
        }
    }
}
