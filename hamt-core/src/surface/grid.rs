
use super::*;
use crate::uom::dimensions::{Area, Length};
use crate::uom::units::meter;
use crate::{
    cell::Boundary,
    material::Construction,
    physics::geometry::{Pt3D, Size3D},
};

/// Hard ceiling on the per-`factor` growth-ratio cap `r_max(factor)` used by
/// [`geometric_norm_pos`]. The two-sided geometric grading grows each cell by a
/// factor `r` relative to its inward neighbour. `r` is no longer a fixed
/// function of `factor`; it is SOLVED so the first cell hits a target size (see
/// [`geometric_norm_pos`]), then CAPPED at the legacy fixed-ratio formula
/// `r_max(factor) = 1 + factor·(MAX - 1)`. This constant is the `factor = 1`
/// end of that cap. Capping `r` keeps the size jump between adjacent cells
/// modest (a finite-volume accuracy concern — large size ratios degrade the
/// second-order spatial truncation). 1.5 (a 50 % max growth per cell) resolves
/// the EN 15026 wetting front while keeping adjacent ratios bounded; values in
/// [1.3, 1.6] behave similarly. Because the cap equals the legacy ratio, the
/// solved-ratio grid is a strict refinement of the legacy grid: identical
/// wherever the solve saturates at `r_max(factor)` (low/moderate n, where the
/// legacy grid did not underflow), and only finer below the cap at high n
/// (where the legacy fixed-ratio grid underflowed exponentially).
const GRID_GROWTH_RATIO_MAX: f64 = 1.5;

/// Most-aggressive target for the FIRST (end) cell, expressed as a fraction of
/// the uniform cell width `1/n`. `factor = 0` maps to fraction `1.0` (uniform);
/// `factor -> 1` maps to this floor (tightest near-surface/interface
/// clustering). The first cell is then `frac/n` of the layer thickness, so it
/// shrinks LINEARLY in `1/n` as the cell count grows — there is no exponential
/// end-cell underflow.
///
/// Calibration: reproduces the EN 15026 benchmark's existing first-cell size at
/// `n = 60`, `factor = 0.7`. The old fixed-ratio grid had total
/// `Σ = 46440 ⇒ frac = 60/46440 ≈ 1.29e-3`; matching that means
/// `FLOOR^0.7 ≈ 1.29e-3 ⇒ FLOOR ≈ 7.5e-5`.
const FIRST_CELL_FRACTION_FLOOR: f64 = 7.5e-5;

/// `Σ_{j=0}^{n-1} r^min(j, n-1-j)` — the unnormalized total layer length for a
/// two-sided symmetric geometric grading with adjacent-cell ratio `r`.
fn geometric_total(n: usize, r: f64) -> f64 {
    (0..n).map(|j| r.powi(j.min(n - 1 - j) as i32)).sum()
}

/// Largest `r` in `(1, r_max]` with
/// `geometric_total(n, r) ~= target_total` (the total is monotone increasing in
/// `r`). If even `r = r_max` can't reach `target_total` (too few cells to cluster
/// that hard), return `r_max` (clustering is accuracy-capped — correct, benign).
fn solve_growth_ratio(n: usize, target_total: f64, r_max: f64) -> f64 {
    let hi = r_max;
    if geometric_total(n, hi) <= target_total {
        return hi;
    }
    let (mut lo, mut hi) = (1.0_f64, r_max);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if geometric_total(n, mid) < target_total {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Normalized boundary position `p(i)` in `[0,1]` for `i` in `0..=n`, two-sided
/// symmetric geometric clustering.
///
/// `factor` in `[0,1]`: `0` = uniform; `->1` = stronger clustering at BOTH ends.
/// Because `build_x_column` applies this per material layer, clustering at both
/// ends of every layer fines the exterior/interior wall surfaces AND every
/// internal layer interface.
///
/// Parameterization (target-fraction -> solved-ratio): rather than fixing the
/// adjacent-cell growth ratio `r` from `factor` directly, we TARGET a first-cell
/// SIZE as a fraction of the uniform cell `1/n` — `factor = 0` -> fraction `1.0`
/// (uniform), `factor -> 1` -> [`FIRST_CELL_FRACTION_FLOOR`] — and then SOLVE for
/// the geometric ratio `r` that yields it. The solved `r` is CAPPED at the
/// legacy fixed-ratio formula `r_max(factor) = 1 + factor·([`GRID_GROWTH_RATIO_MAX`] - 1)`,
/// which makes this scheme a strict refinement of the legacy grid: identical
/// wherever the solve saturates at `r_max(factor)` (the legacy ratio — low/moderate
/// `n`, where the legacy grid did not underflow), and only finer below the cap at
/// high `n`. The first cell is therefore `frac/n` of the layer, which shrinks
/// LINEARLY in `1/n` as the cell count grows: NO exponential end-cell underflow
/// (the old fixed-`r` scheme made the end cell `~1/r^(n/2)`, underflowing to
/// sub-nanometre at high `n`). The solved `r -> 1` as `n` grows.
///
/// Properties: `p(0)=0`, `p(n)=1`, strictly increasing, and symmetric
/// (`p(i) = 1 - p(n-i)`). `O(n)` per call — the grid is built once with small
/// `n`, so this is not a hot path.
fn geometric_norm_pos(i: usize, n: usize, factor: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if i == 0 {
        return 0.0;
    }
    if i >= n {
        return 1.0;
    }
    if factor <= 1e-6 {
        return i as f64 / n as f64; // uniform
    }

    // Target first-cell fraction of the uniform cell (1/n): factor 0 -> 1.0
    // (uniform), factor -> 1 -> FIRST_CELL_FRACTION_FLOOR. Solve for the growth
    // ratio that yields it, so the first cell is `frac/n` (linear in 1/n — no
    // exponential underflow as n grows).
    let frac = FIRST_CELL_FRACTION_FLOOR.powf(factor.clamp(0.0, 1.0));
    let target_total = n as f64 / frac; // first cell (normalized) = 1/target_total = frac/n
    // Cap the solved ratio at the old fixed-ratio formula r_max(factor), so this grid equals the legacy grid wherever the legacy grid did not underflow (low/moderate n); it only refines below the cap at high n (where the legacy grid underflowed exponentially).
    let r_max = 1.0 + factor.clamp(0.0, 1.0) * (GRID_GROWTH_RATIO_MAX - 1.0);
    let r = solve_growth_ratio(n, target_total, r_max);

    // Relative size of cell j (j in 0..n): smallest at the two ends (j=0 and
    // j=n-1), largest in the middle — symmetric about the layer centre.
    let size = |j: usize| -> f64 { r.powi(j.min(n - 1 - j) as i32) };

    let total: f64 = (0..n).map(size).sum();
    let partial: f64 = (0..i).map(size).sum();
    partial / total
}

/// Build ONE x-column of cells (exterior boundary stack -> material layers ->
/// interior boundary stack), advancing `base_pt.x` as it goes. `col_y`/`col_z`
/// are the cross-section extents (y,z) used for EVERY cell in this column
/// (boundary and material). Returns the column cells in x-order.
pub(super) fn build_x_column(
    settings: &HamtSettings,
    construction: &Construction,
    base_pt: &mut Pt3D,
    col_y: Length<f64>,
    col_z: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<Vec<Cell>> {
    let boundary_cell_thickness = settings.base_cell_size.x();
    let boundary_cell_size = Size3D::new(boundary_cell_thickness, col_y, col_z);

    let get_size = |thickness: Length<f64>| Pt3D(thickness, col_y, col_z);
    let get_origin = |bp: Pt3D, size: Size3D| bp + size / 2.0;

    let mut cells: Vec<Cell> = vec![];

    // --- Exterior Boundary Cells (4) ---
    // All four share the same base_pt; no x-advance between them.
    let ext_con_cell = Cell::new(
        CellType::Boundary(Boundary::Convection, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_con_cell);

    let ext_rad_cell = Cell::new(
        CellType::Boundary(Boundary::Radiation, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_rad_cell);

    let ext_sky_cell = Cell::new(
        CellType::Boundary(Boundary::Sky, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_sky_cell);

    let ext_grn_cell = Cell::new(
        CellType::Boundary(Boundary::Ground, BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_grn_cell);

    // --- Exterior Virtual Surface Cell ---
    base_pt.0 += boundary_cell_thickness;
    let ext_virtual_cell = Cell::new(
        CellType::Virtual(BoundaryLocation::Exterior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(ext_virtual_cell);
    base_pt.0 += boundary_cell_thickness;

    // --- Material Cells ---
    let dist_factor = settings.div_distribution_factor;
    if dist_factor > 1.0 || dist_factor < 0.0 {
        return Err(crate::error::Error::InvalidInput(
            "DivDistributionFactor".to_string(),
            dist_factor.to_string(),
            crate::error::InvalidInputReason::OutOfRange,
        )
        .into());
    }

    for mat_arc in construction.layers.iter() {
        let mat_thickness_m = mat_arc.thickness;
        let div_len_m = settings.div_length;

        // 1. Number of divisions for this layer
        let divs = mat_arc.num_div.unwrap_or(
            ((mat_thickness_m / div_len_m).value().ceil() as i32).max(settings.n_div) as usize,
        );

        // Diagnostic: derived first-cell size and solved growth ratio for this
        // layer (target-fraction -> solved-ratio parameterization).
        if dist_factor > 1e-6 && divs > 1 {
            let frac = FIRST_CELL_FRACTION_FLOOR.powf(dist_factor.clamp(0.0, 1.0));
            let r_max = 1.0 + dist_factor.clamp(0.0, 1.0) * (GRID_GROWTH_RATIO_MAX - 1.0);
            let r = solve_growth_ratio(divs, divs as f64 / frac, r_max);
            let first_cell_m = (mat_thickness_m * geometric_norm_pos(1, divs, dist_factor))
                .value_in(meter); // strip: tracing log field
            tracing::debug!(
                material = %mat_arc.name,
                divs,
                factor = dist_factor,
                solved_ratio = r,
                first_cell_m,
                "geometric grid layer"
            );
        }

        // The position of the i-th boundary (0..=divs) within this layer is given
        // by `geometric_norm_pos` (two-sided symmetric geometric clustering),
        // proportional to layer thickness (range 0.0 to 1.0). The solver is fully
        // implicit (backward-Euler / block-Thomas / ADI) and therefore
        // unconditionally stable, so there is no minimum-cell-size constraint:
        // every requested division is honoured regardless of how small the
        // resulting first cell is.

        // 2. Generate Cells
        let mut prev_pos_m = 0.0 * meter; // Start at 0 relative to material layer

        for did in 1..=divs {
            // Calculate current boundary position
            let norm_pos = geometric_norm_pos(did, divs, dist_factor);
            let curr_pos_m = mat_thickness_m * norm_pos;

            // Length is difference between current boundary and previous boundary
            let length = curr_pos_m - prev_pos_m;
            prev_pos_m = curr_pos_m;

            let size = get_size(length);

            let cell = Cell::new(
                CellType::Material(mat_arc.clone()),
                get_origin(*base_pt, size),
                size,
            );

            cells.push(cell);

            // Update base pointer for the NEXT cell
            base_pt.0 += length;
        }
    }

    // --- Interior Virtual Surface Cell ---
    let int_virtual_cell = Cell::new(
        CellType::Virtual(BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_virtual_cell);

    base_pt.0 += boundary_cell_size.0;

    // --- Interior Boundary Cells ---
    // Interior Convection and Radiation cells share the same geometric position
    // (same base_pt, no x-advance between them) — mirroring the exterior side which
    // stacks Convection/Radiation/Sky/Ground at the same base_pt. Both are reachable
    // from the interior Virtual cell because `neighbor_indices` for Virtual(Interior)
    // collects ALL cells to its right, so the Virtual acts as a hub and the 1-D
    // neighbor walk doesn't need Radiation to be at a distinct x slot.
    let int_con_cell = Cell::new(
        CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_con_cell);

    let int_rad_cell = Cell::new(
        CellType::Boundary(Boundary::Radiation, BoundaryLocation::Interior),
        get_origin(*base_pt, boundary_cell_size),
        boundary_cell_size,
    );
    cells.push(int_rad_cell);

    // calc dependent properties
    cells
        .iter_mut()
        .for_each(|c| c.calc_dependent_state(init_bc, settings));

    Ok(cells)
}

/// Build a (nx, ny, 1) cell lattice: `ny` x-columns stacked along y, each with
/// real cross-section (col_y = y_extent/ny, col_z). Column j sits at y-origin
/// j*col_y. Returns the ArcArray3 ready for Surface assembly.
///
/// The 1-D path is the `ny = 1` case: one column at y-origin 0 with
/// `col_y = col_z = sqrt(area)` — byte-identical to the legacy `(N,1,1)` build.
///
/// `build_x_column` advances only `base_pt.x`; the y/z components of `base_pt`
/// are honoured by `get_origin` (`bp + size/2.0`) for every cell origin, so
/// seeding `base_pt.y = j*col_y` places column j correctly without any change
/// to `build_x_column`. Subdivision is x-only, so every column has the same nx.
pub(super) fn build_grid_2d(
    settings: &HamtSettings,
    construction: &Construction,
    surface_area: Area<f64>,
    ny: usize,
    y_extent: Length<f64>,
    init_bc: &BoundaryCondition,
) -> Result<ndarray::ArcArray3<Cell>> {
    let boundary_cell_thickness = settings.base_cell_size.x();
    let col_y = y_extent / ny as f64;
    let col_z = surface_area.sqrt(); // 2-D out-of-plane extent (z); unchanged convention
    let mut columns: Vec<Vec<Cell>> = Vec::with_capacity(ny);
    for j in 0..ny {
        // Each column starts at the same x offset as the 1-D path (two exterior
        // boundary layers worth of negative x), and y-origin j*col_y.
        let mut base_pt = Pt3D::new(-boundary_cell_thickness * 2.0, col_y * j as f64, 0.0 * meter);
        let col = build_x_column(settings, construction, &mut base_pt, col_y, col_z, init_bc)?;
        columns.push(col);
    }
    let nx = columns[0].len();
    // All columns must have identical nx (subdivision is x-only).
    assert!(
        columns.iter().all(|c| c.len() == nx),
        "non-uniform column length"
    );
    // Flatten C-order for shape (nx, ny, 1): flat[i*ny + j] = columns[j][i] = cell[[i,j,0]].
    let mut flat: Vec<Cell> = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for col in columns.iter() {
            flat.push(col[i].clone());
        }
    }
    Ok(ndarray::ArcArray3::from_shape_vec((nx, ny, 1), flat)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::BoundaryLocation;
    use crate::config::{FACE_Y_MIN, FaceBc};
    use crate::material::{Material, MaterialLibrary, SurfaceVaporCoef};
    use crate::physics::{Curve, LogY};
    use crate::uom::units::{
        dimensionless, j_per_kg_k, kg_per_m3, m2_per_s, square_meter, w_per_m_k,
    };
    use crate::uom::{
        DIM_DIMENSIONLESS, DIM_LIQUID_TRANSPORT_COEFF, DIM_THERMAL_CONDUCTIVITY,
        DIM_WATER_CONTENT, Quantity,
    };

    /// Build a minimal single-layer construction with simple monotone curves,
    /// suitable for exercising the grid topology (no solve is performed).
    fn build_test_construction() -> Arc<Construction> {
        let w_sat = 146.0 * kg_per_m3;
        let n_pts = 10usize;
        let si_x: Vec<_> = (0..=n_pts)
            .map(|k| (k as f64 / n_pts as f64) * dimensionless)
            .collect();
        let si_y: Vec<_> = si_x.iter().map(|&rh| w_sat * rh.value().max(0.0)).collect();
        let sorption_isotherm: Curve<DIM_DIMENSIONLESS, DIM_WATER_CONTENT> =
            Curve::new(si_x.clone(), si_y);

        let mu_y: Vec<_> = si_x.iter().map(|_| 10.0 * dimensionless).collect();
        let vapor_diffusion_resistance: Curve<DIM_DIMENSIONLESS, DIM_DIMENSIONLESS> =
            Curve::new(si_x.clone(), mu_y);

        let wc_x: Vec<_> = (0..=n_pts).map(|k| (k as f64 / n_pts as f64) * w_sat).collect();
        let k_y: Vec<_> = wc_x.iter().map(|_| 0.5 * w_per_m_k).collect();
        let thermal_conductivity: Curve<DIM_WATER_CONTENT, DIM_THERMAL_CONDUCTIVITY> =
            Curve::new(wc_x.clone(), k_y);

        let dw_y: Vec<_> = wc_x.iter().map(|_| 1.0e-10 * m2_per_s).collect();
        let liquid_tc: Curve<DIM_WATER_CONTENT, DIM_LIQUID_TRANSPORT_COEFF, LogY> =
            Curve::<_, _, LogY>::try_new(wc_x.clone(), dw_y).expect("valid Dw curve");

        let mat = Material {
            name: "GridTestMat".to_string(),
            thickness: 0.1 * meter,
            density: 200.0 * kg_per_m3,
            spec_heat: 1000.0 * j_per_kg_k,
            porosity: Quantity(0.146),
            initial_water_content: Quantity(0.5 * 0.146),
            sorption_isotherm,
            suction: liquid_tc.clone(),
            redistribution: liquid_tc,
            vapor_diffusion_resistance,
            thermal_conductivity,
            sorption_enthalpy: None,
            initial_temp: crate::boundary::celsius_to_kelvin(20.0),
            num_div: Some(3),
            div_distribution_factor: 0.5,
        };

        let lib = MaterialLibrary::new();
        lib.add_material(mat).expect("add material");
        let surface_coeff = SurfaceVaporCoef {
            is_external_const: false,
            external_value: 1e-12,
            is_internal_const: false,
            internal_value: 1e-8,
        };
        lib.create_construction(
            "GridTestConstruction".to_string(),
            vec!["GridTestMat".to_string()],
            surface_coeff,
        )
        .expect("create construction");
        lib.get_construction("GridTestConstruction").unwrap()
    }

    #[test]
    fn grid_2d_shape_and_sheets() {
        let settings = HamtSettings::default();
        let bc = BoundaryCondition::default();
        let construction = build_test_construction();
        let ny = 4usize;
        let y_extent = 2.0 * meter;

        let surface = Surface::from_construction_2d(
            &settings,
            construction,
            "GridTest".to_string(),
            1.0 * square_meter,
            ny,
            y_extent,
            &bc,
        )
        .expect("build 2-D surface");

        // Shape is (nx, ny, 1) with a genuinely subdivided x-stack.
        let dim = surface.cells.raw_dim();
        let nx = dim[0];
        assert!(nx > 1, "expected nx > 1, got {nx}");
        assert_eq!([dim[0], dim[1], dim[2]], [nx, ny, 1]);

        // Every Material cell spans col_y = y_extent / ny = 0.5 m in y.
        let expected_col_y = (y_extent / ny as f64).value_in(meter); // strip: scalar tolerance compare
        for cell in surface.cells.iter() {
            if matches!(cell.cell_type, CellType::Material(_)) {
                let got = cell.size.y().value_in(meter); // strip: scalar tolerance compare
                assert!(
                    (got - expected_col_y).abs() < 1e-9,
                    "material cell y-size {got} != {expected_col_y}"
                );
            }
        }

        // Per-row boundary sheets: one interior Convection + one interior Virtual
        // per y-row (no global Virtual cell).
        let int_conv = CellType::Boundary(Boundary::Convection, BoundaryLocation::Interior);
        let int_virtual = CellType::Virtual(BoundaryLocation::Interior);
        let count = |ct: &CellType| surface.cells.iter().filter(|c| &c.cell_type == ct).count();
        assert_eq!(count(&int_conv), ny, "interior Convection sheet count");
        assert_eq!(count(&int_virtual), ny, "interior Virtual sheet count");

        // Side (y-min) face stays Adiabatic.
        assert_eq!(surface.face_bcs[FACE_Y_MIN], FaceBc::Adiabatic);
    }

    // ---- geometric_norm_pos distribution tests ----

    #[test]
    fn geometric_norm_pos_endpoints() {
        for &n in &[1usize, 2, 3, 5, 8, 60, 480] {
            for &f in &[0.0, 0.3, 0.5, 0.7, 0.9, 1.0] {
                assert_eq!(geometric_norm_pos(0, n, f), 0.0, "p(0) for n={n}, f={f}");
                assert_eq!(geometric_norm_pos(n, n, f), 1.0, "p(n) for n={n}, f={f}");
                // Out-of-range i clamps to 1.0.
                assert_eq!(geometric_norm_pos(n + 1, n, f), 1.0);
            }
        }
        // Degenerate n == 0.
        assert_eq!(geometric_norm_pos(0, 0, 0.5), 0.0);
    }

    #[test]
    fn geometric_norm_pos_monotonic_increasing() {
        for &n in &[2usize, 3, 5, 8, 17, 60] {
            for &f in &[0.0, 0.4, 0.7, 0.9, 1.0] {
                let mut prev = geometric_norm_pos(0, n, f);
                for i in 1..=n {
                    let cur = geometric_norm_pos(i, n, f);
                    assert!(
                        cur > prev,
                        "not strictly increasing at i={i}, n={n}, f={f}: {prev} -> {cur}"
                    );
                    prev = cur;
                }
            }
        }
    }

    #[test]
    fn geometric_norm_pos_factor_zero_is_uniform() {
        for &n in &[1usize, 2, 5, 13, 60] {
            for i in 0..=n {
                let expected = i as f64 / n as f64;
                assert!(
                    (geometric_norm_pos(i, n, 0.0) - expected).abs() < 1e-15,
                    "factor=0 not uniform at i={i}, n={n}"
                );
            }
        }
    }

    #[test]
    fn geometric_norm_pos_symmetric() {
        for &n in &[2usize, 3, 5, 8, 17, 60] {
            for &f in &[0.0, 0.3, 0.5, 0.7, 0.9, 1.0] {
                for i in 0..=n {
                    let p = geometric_norm_pos(i, n, f);
                    let mirror = 1.0 - geometric_norm_pos(n - i, n, f);
                    assert!(
                        (p - mirror).abs() < 1e-12,
                        "not symmetric at i={i}, n={n}, f={f}: {p} vs {mirror}"
                    );
                }
            }
        }
    }

    #[test]
    fn geometric_norm_pos_clusters_at_ends() {
        for &n in &[5usize, 8, 17, 60, 120] {
            for &f in &[0.3, 0.5, 0.7, 0.9, 1.0] {
                let first = geometric_norm_pos(1, n, f) - geometric_norm_pos(0, n, f);
                // Middle cell: pick the cell straddling the layer centre.
                let mid = n / 2;
                let middle = geometric_norm_pos(mid + 1, n, f) - geometric_norm_pos(mid, n, f);
                let uniform = 1.0 / n as f64;
                assert!(
                    first < middle,
                    "first cell not finer than middle at n={n}, f={f}: {first} vs {middle}"
                );
                assert!(
                    first < uniform,
                    "first cell not finer than uniform at n={n}, f={f}: {first} vs {uniform}"
                );
            }
        }
    }

    /// The first (end) cell must shrink ~LINEARLY in `1/n`, not exponentially:
    /// the target-fraction -> solved-ratio parameterization makes the normalized
    /// first cell `fc(n) = frac/n` with `frac` independent of `n` (in the
    /// uncapped regime). The OLD fixed-ratio grid made `fc(n) ~ 1/r^(n/2)`, which
    /// underflowed to ~1e-30 by n=480.
    #[test]
    fn geometric_norm_pos_no_exponential_underflow() {
        let factor = 0.7;
        let fc = |n: usize| geometric_norm_pos(1, n, factor); // normalized first-cell width

        let ns = [60usize, 120, 240, 480];
        let fcs: Vec<f64> = ns.iter().map(|&n| fc(n)).collect();
        let prods: Vec<f64> = ns.iter().zip(&fcs).map(|(&n, &f)| n as f64 * f).collect();

        // (a) n*fc(n) ~ constant (= frac): each of n=120,240,480 within 5% of n=60.
        let ref_prod = prods[0];
        for (k, &p) in prods.iter().enumerate().skip(1) {
            let rel = (p - ref_prod).abs() / ref_prod;
            assert!(
                rel < 0.05,
                "n*fc not constant: n={}, n*fc={p} vs ref(n=60)={ref_prod} (rel {rel})",
                ns[k]
            );
        }

        // (b) fc(480) must NOT underflow (old fixed-ratio grid would give ~1e-30).
        assert!(
            fcs[3] > 1e-7,
            "fc(480)={} underflowed (expected linear ~frac/480)",
            fcs[3]
        );

        // (c) FLOOR calibration reproduces the EN-15026 benchmark operating point:
        //     fc(60) ~ old 2.15µm-equivalent normalized first cell (2.15e-5) within ~15%.
        let expected_fc60 = 2.15e-5;
        let rel60 = (fcs[0] - expected_fc60).abs() / expected_fc60;
        assert!(
            rel60 < 0.15,
            "fc(60)={} not within 15% of calibration target {expected_fc60} (rel {rel60})",
            fcs[0]
        );
    }

    /// MULTI-LAYER INTERFACE refinement: build a 2-layer construction and assert
    /// that material cells adjacent to the internal layer interface (and to the
    /// outer surfaces) are finer in x than the cells in the middle of a layer.
    #[test]
    fn geometric_two_layer_interface_refinement() {
        use crate::material::{MaterialLibrary, SurfaceVaporCoef};

        // Reuse the single-layer test material twice to form a 2-layer stack.
        // `build_test_construction` registers "GridTestMat" in its own library;
        // build a fresh library here with two copies so we control divisions.
        let base = build_test_construction();
        let mat0 = base.layers[0].as_ref().clone();
        let mut mat_a = mat0.clone();
        mat_a.name = "LayerA".to_string();
        mat_a.thickness = 0.1 * meter;
        mat_a.num_div = Some(9);
        mat_a.div_distribution_factor = 0.9;
        let mut mat_b = mat0;
        mat_b.name = "LayerB".to_string();
        mat_b.thickness = 0.1 * meter;
        mat_b.num_div = Some(9);
        mat_b.div_distribution_factor = 0.9;

        let lib = MaterialLibrary::new();
        lib.add_material(mat_a).expect("add LayerA");
        lib.add_material(mat_b).expect("add LayerB");
        let surface_coeff = SurfaceVaporCoef {
            is_external_const: false,
            external_value: 1e-12,
            is_internal_const: false,
            internal_value: 1e-8,
        };
        lib.create_construction(
            "TwoLayer".to_string(),
            vec!["LayerA".to_string(), "LayerB".to_string()],
            surface_coeff,
        )
        .expect("create two-layer construction");
        let construction = lib.get_construction("TwoLayer").unwrap();

        let mut settings = HamtSettings::default();
        settings.div_distribution_factor = 0.9;
        let bc = BoundaryCondition::default();

        let surface = Surface::from_construction(
            &settings,
            construction,
            "TwoLayerSurface".to_string(),
            1.0 * square_meter,
            &bc,
        )
        .expect("build surface");

        // Collect the x-sizes of material cells in order (1-D: ny=nz=1).
        let mat_sizes: Vec<f64> = surface
            .cells
            .iter()
            .filter(|c| matches!(c.cell_type, CellType::Material(_)))
            .map(|c| c.size.x().value_in(meter)) // strip: scalar comparison
            .collect();

        assert_eq!(mat_sizes.len(), 18, "expected 9 + 9 material cells");

        // Layer A spans indices 0..9, Layer B spans 9..18 (each 9 cells).
        // With two-sided clustering, the smallest cells are at the four faces:
        //   - exterior surface (cell 0),
        //   - interface, layer-A side (cell 8, last of A),
        //   - interface, layer-B side (cell 9, first of B),
        //   - interior surface (cell 17).
        // The largest cells are in the middle of each layer (cell 4 and cell 13).
        let mid_a = mat_sizes[4];
        let mid_b = mat_sizes[13];

        // Cells adjacent to the internal interface (A side = idx 8, B side = idx 9).
        assert!(
            mat_sizes[8] < mid_a,
            "interface (A-side) cell {} not finer than layer-A middle {}",
            mat_sizes[8],
            mid_a
        );
        assert!(
            mat_sizes[9] < mid_b,
            "interface (B-side) cell {} not finer than layer-B middle {}",
            mat_sizes[9],
            mid_b
        );
        // Outer surface cells are also fine.
        assert!(
            mat_sizes[0] < mid_a,
            "exterior surface cell not finer than layer-A middle"
        );
        assert!(
            mat_sizes[17] < mid_b,
            "interior surface cell not finer than layer-B middle"
        );
    }
}
