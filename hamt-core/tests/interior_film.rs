#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

//! Type 395 output 8 ("HeatFlux_in") is the TOTAL interior film heat flux:
//! convective film to the room air + longwave film to the mean radiant temperature.
//!
//! In a 14-input deck (separate interior radiation: input 12 `HCONVI` to the AIR node,
//! input 13 `TMR` with `h_rad = 4·σ·ε·T_m³`) the interior surface node hangs on TWO
//! parallel films — the interior `Convection` cell (HCONVI, TAIR) and the interior
//! `Radiation` cell (h_rad, TMR) — and the solver applies both. `Surface::heat_flux`
//! used to sum only the `Convection` cell, so output 8 (and the `HeatFlux_in` columns of
//! `Type395_*.out`) reported a fraction `h_c/(h_c + h_rad)` of what the wall actually
//! received from side 1: 1.37 of 2.18 W/m² through the DIN wall (HCONVI 8.65,
//! h_rad 5.14 W/(m²·K)), and even the WRONG SIGN when the radiant environment is colder
//! than the wall while the air is warmer.
//!
//! What enters the wall from side 1 is `output 8 + qWG1`: `qWG1` (input index 6, absorbed
//! shortwave / internal gains) is deposited in the interior surface material cell and is
//! NOT part of the film flux.
//!
//! The sign convention is unchanged: environment → surface positive.
//!
//! Wall of the steady-state tests: 0.1 m of the EN 15026 material (λ = 1.5 + 15.8e-3·w
//! ≈ 2.18 W/(m·K) at 50 % RH), exterior film 10 W/(m²·K) convective, vapour-tight
//! (β ≈ 0), so only heat moves and λ stays at its 50 %-RH value. The room has
//! `T_air ≠ T_MR`, so the two interior films drive different fluxes:
//!
//! ```text
//!   h   = h_c + h_rad,      T_eq = (h_c·T_air + h_rad·T_MR) / h
//!   q   = (T_eq − T_out) / (1/h + R_wall + 1/h_ext)          (env → wall at side 1)
//!   q_c = h_c·(T_air − T_s),   q_r = h_rad·(T_MR − T_s),   T_s = T_eq − q/h
//! ```
//!
//! The closed form takes `R_wall = Σ dx/λ` from the solver's own cells and `h_rad` from
//! the converged surface temperature (the solver linearises at `T_m = (T_s,mat + T_MR)/2`).

mod common;

use anyhow::Result;
use common::{CONSTRUCTION_NAME, create_library};

use hamt_core::boundary::{BoundaryCondition, kelvin_to_celsius};
use hamt_core::cell::{Boundary, BoundaryLocation, CellType};
use hamt_core::hamt_manager::{AdaptiveState, HamtManager, HamtSettings};
use hamt_core::physics::{EPSILON, SIGMA};
use hamt_core::uom::units::{
    hours, kj_per_hr, meter, seconds, square_meter, w_per_m_k, w_per_m2, w_per_m2_k,
};

const THICKNESS_M: f64 = 0.1;
/// Steady-state closed form vs solver: measured ≤ 1e-5 (see the asserts), bound 1e-4.
const CLOSED_FORM_TOL: f64 = 1e-4;
/// Steady-state energy balance of the reported fluxes: measured ≤ 1e-6, bound 1e-5.
const BALANCE_TOL: f64 = 1e-5;
/// Hourly macro steps; the wall's thermal time constant is ≈ 11 h, so 600 h is steady.
const STEADY_HOURS: usize = 600;
/// 1 W (W/m², W/(m²·K)) in the deck's kJ/h units.
const KJ_PER_H_PER_W: f64 = 3.6;

/// The room and the exterior, in the deck's terms.
#[derive(Clone, Copy)]
struct Room {
    /// TAIR (input 11) [°C]: the convective interior film is paired with this.
    t_air_c: f64,
    /// TMR (input 13) [°C]: the longwave interior film is paired with this.
    t_mr_c: f64,
    /// TB2 (input 1) [°C].
    t_out_c: f64,
    /// HCONVI (input 12), the PURE convective interior coefficient [W/(m²·K)].
    h_conv_w: f64,
    /// hconv2 (input 5), the exterior convective coefficient [W/(m²·K)].
    h_ext_w: f64,
    /// qWG1 (input index 6) [W]: interior surface gain, NOT part of the film flux.
    q_wg1_w: f64,
}

impl Room {
    /// Winter room with cold surfaces around: the air is 5 K warmer than the radiant
    /// environment, HCONVI = 8.65 W/(m²·K) as in the production deck (h_rad ≈ 5.1).
    fn winter() -> Self {
        Room {
            t_air_c: 22.0,
            t_mr_c: 17.0,
            t_out_c: 0.0,
            h_conv_w: 8.65,
            h_ext_w: 10.0,
            q_wg1_w: 0.0,
        }
    }
}

/// 14-input Type 395 vector of a deck with separate interior radiation, vapour-tight.
fn inputs14(r: &Room) -> [f64; 14] {
    [
        25.0,                        // 0  TSTAR — NOT used by the films of a 14-input deck
        r.t_out_c,                   // 1  TB2
        50.0,                        // 2  RH1 [%]
        50.0,                        // 3  RH2 [%]
        30.0,                        // 4  hconv1, combined — NOT used when input 12 is present
        r.h_ext_w * KJ_PER_H_PER_W,  // 5  hconv2
        r.q_wg1_w * KJ_PER_H_PER_W,  // 6  qWG1
        0.0,                         // 7  qWG2
        1e-20,                       // 8  beta1
        1e-20,                       // 9  beta2
        0.0,                         // 10 user-defined heat
        r.t_air_c,                   // 11 TAIR
        r.h_conv_w * KJ_PER_H_PER_W, // 12 HCONVI
        r.t_mr_c,                    // 13 TMR
    ]
}

/// Legacy 12-input vector: ONE combined interior film `h` to TSTAR, no TMR / HCONVI.
fn inputs12_legacy(t_in_c: f64, t_out_c: f64, h_w: f64, h_ext_w: f64) -> [f64; 12] {
    [
        t_in_c,                   // 0  TSTAR
        t_out_c,                  // 1  TB2
        50.0,                     // 2  RH1 [%]
        50.0,                     // 3  RH2 [%]
        h_w * KJ_PER_H_PER_W,     // 4  hconv1 (combined)
        h_ext_w * KJ_PER_H_PER_W, // 5  hconv2
        0.0,                      // 6  qWG1
        0.0,                      // 7  qWG2
        1e-20,                    // 8  beta1
        1e-20,                    // 9  beta2
        0.0,                      // 10 user-defined heat
        t_in_c,                   // 11 TAIR
    ]
}

fn settings() -> HamtSettings {
    let mut s = HamtSettings {
        timestep: 1.0 * hours,
        n_div: 1, // the material's num_div wins
        ..Default::default()
    };
    s.min_relaxation_factor = 0.1; // as the DLL
    s
}

fn manager_1d(init_temp_c: f64, area_m2: f64, first_inputs: &[f64]) -> Result<HamtManager> {
    let library = create_library(init_temp_c, 0.5, vec![THICKNESS_M], vec![Some(20)]);
    let construction = library.get_construction(CONSTRUCTION_NAME)?;
    HamtManager::new(
        Some(settings()),
        Some(BoundaryCondition::from_trnsys_inputs(first_inputs)),
        construction,
        "interior_film_1d".to_string(),
        area_m2 * square_meter,
    )
}

/// Advance `n_hours` hourly macro steps with the production driver. The accumulator of
/// `adaptive` holds the LAST macro step afterwards (it is reset by every solve).
fn run_hours(
    mgr: &mut HamtManager,
    adaptive: &mut AdaptiveState<()>,
    input: &[f64],
    n_hours: usize,
) -> Result<()> {
    for h in 0..n_hours {
        mgr.boundary_condition = BoundaryCondition::from_trnsys_inputs(input);
        adaptive.current_time = (h as f64) * hours;
        mgr.solve_timestep_adaptive(adaptive, None)?;
        mgr.tick_timestep()?;
    }
    Ok(())
}

/// One boundary film as the solver holds it: (kind, h [W/(m²·K)], environment T [°C]).
type Film = (Boundary, f64, f64);

/// Column (0,0) of one face: the surface (Virtual) node temperature [°C] and its films.
fn face(mgr: &HamtManager, loc: BoundaryLocation) -> (f64, Vec<Film>) {
    let mut t_node = None;
    let mut films = Vec::new();
    for ((_, j, k), cell) in mgr.surface_curr.cells.indexed_iter() {
        if j != 0 || k != 0 {
            continue;
        }
        match &cell.cell_type {
            CellType::Virtual(l) if *l == loc => t_node = Some(kelvin_to_celsius(cell.cell_temp)),
            CellType::Boundary(kind, l) if *l == loc => films.push((
                kind.clone(),
                // strip: test-side scalar bookkeeping
                cell.htc.value_in(w_per_m2_k),
                kelvin_to_celsius(cell.cell_temp),
            )),
            _ => {}
        }
    }
    (t_node.expect("surface (Virtual) node"), films)
}

/// Σ over the ACTIVE films of `h·(T_env − T_surface)` [W/m²], environment → wall: what the
/// solver applies to the surface node. Films with `h = 0` (inert placeholders) drop out.
fn applied_film_flux(t_node: f64, films: &[Film]) -> f64 {
    films.iter().map(|(_, h, t)| h * (t - t_node)).sum()
}

/// The convective film alone [W/m²] — what output 8 reported before the fix.
fn convective_film_flux(t_node: f64, films: &[Film]) -> f64 {
    films
        .iter()
        .filter(|(kind, _, _)| *kind == Boundary::Convection)
        .map(|(_, h, t)| h * (t - t_node))
        .sum()
}

fn active_films(films: &[Film]) -> usize {
    films.iter().filter(|(_, h, _)| *h > 0.0).count()
}

/// Everything a steady-state test reads from the solver.
struct Steady {
    /// `report().q_surf_int` / `q_surf_ext` (end-point fluxes) [W/m²].
    q_int: f64,
    q_ext: f64,
    /// Time-averaged fluxes of the last macro step (the values behind Type 395 outputs 8 / 9).
    acc_q_int: f64,
    acc_q_ext: f64,
    acc_total_time_s: f64,
    /// Output 8 / 9 of the DLL while the accumulator holds a solve: `(avg · area)` in kJ/h.
    output_8_kj_h: f64,
    output_9_kj_h: f64,
    /// Output 8 of the DLL before any solve (accumulator empty): `report().q_surf_int · area`.
    fallback_8_kj_h: f64,
    /// Surface (Virtual) node temperatures [°C] and films of column (0,0).
    t_si: f64,
    t_so: f64,
    films_int: Vec<Film>,
    films_ext: Vec<Film>,
    /// `Σ dx/λ` over the material cells with the solver's own λ(w) [m²K/W].
    r_wall: f64,
    /// Interior-most material cell [°C] (the proxy the solver linearises `h_rad` at).
    t_s_mat: f64,
    /// Surface area [m²].
    area: f64,
}

fn steady_state(mut mgr: HamtManager, input: &[f64], n_hours: usize) -> Result<Steady> {
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    run_hours(&mut mgr, &mut adaptive, input, n_hours)?;
    snapshot(&mgr, &adaptive)
}

fn snapshot(mgr: &HamtManager, adaptive: &AdaptiveState<()>) -> Result<Steady> {
    let rep = mgr.surface_curr.report(&mgr.boundary_condition)?;
    let acc = &adaptive.flux_accumulator;
    let (t_si, films_int) = face(mgr, BoundaryLocation::Interior);
    let (t_so, films_ext) = face(mgr, BoundaryLocation::Exterior);

    let mut r_wall = 0.0;
    for ((_, j, k), cell) in mgr.surface_curr.cells.indexed_iter() {
        if j == 0 && k == 0 && matches!(cell.cell_type, CellType::Material(_)) {
            // strip: test-side scalar bookkeeping, R = Σ dx/λ [m²K/W]
            r_wall += cell.size.x().value_in(meter) / cell.w_thermal_c.value_in(w_per_m_k);
        }
    }
    let i_mat = mgr
        .surface_curr
        .interior_surface_material_idx
        .expect("interior surface material cell");
    let t_s_mat = kelvin_to_celsius(mgr.surface_curr.cells[[i_mat, 0, 0]].cell_temp);

    Ok(Steady {
        q_int: rep.q_surf_int.value_in(w_per_m2),
        q_ext: rep.q_surf_ext.value_in(w_per_m2),
        acc_q_int: acc.avg_q_int().value_in(w_per_m2),
        acc_q_ext: acc.avg_q_ext().value_in(w_per_m2),
        acc_total_time_s: acc.total_time.value_in(seconds),
        output_8_kj_h: (acc.avg_q_int() * mgr.surface_curr.area).value_in(kj_per_hr),
        output_9_kj_h: (acc.avg_q_ext() * mgr.surface_curr.area).value_in(kj_per_hr),
        fallback_8_kj_h: (rep.q_surf_int * mgr.surface_curr.area).value_in(kj_per_hr),
        t_si,
        t_so,
        films_int,
        films_ext,
        r_wall,
        t_s_mat,
        area: mgr.surface_curr.area.value_in(square_meter),
    })
}

/// `h_rad = 4·σ·ε·T_m³` at `T_m = (T_s,mat + T_MR)/2` — the solver's linearisation.
fn h_rad_expected(t_s_mat_c: f64, t_mr_c: f64) -> f64 {
    let t_m_k = 0.5 * (t_s_mat_c + t_mr_c) + 273.15;
    4.0 * SIGMA * EPSILON * t_m_k.powi(3)
}

/// The h_rad the solver applied: the interior Radiation cell's htc [W/(m²·K)].
fn h_rad_applied(films_int: &[Film]) -> f64 {
    films_int
        .iter()
        .filter(|(kind, _, _)| *kind == Boundary::Radiation)
        .map(|(_, h, _)| *h)
        .sum()
}

/// Closed form of the steady state with two parallel interior films (see the module doc).
struct ClosedForm {
    /// Flux entering the wall at side 1 [W/m²] (= the total interior film flux, `qWG1 = 0`).
    q: f64,
    /// Surface (Virtual) node temperature [°C].
    t_s: f64,
    q_conv: f64,
    q_rad: f64,
}

fn closed_form(room: &Room, h_rad: f64, r_wall: f64) -> ClosedForm {
    let h = room.h_conv_w + h_rad;
    let t_eq = (room.h_conv_w * room.t_air_c + h_rad * room.t_mr_c) / h;
    let q = (t_eq - room.t_out_c) / (1.0 / h + r_wall + 1.0 / room.h_ext_w);
    let t_s = t_eq - q / h;
    ClosedForm {
        q,
        t_s,
        q_conv: room.h_conv_w * (room.t_air_c - t_s),
        q_rad: h_rad * (room.t_mr_c - t_s),
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

// ---------------------------------------------------------------------------
// (a) steady state: total interior flux = conv + rad = parallel-film U, and −q_ext
// ---------------------------------------------------------------------------

/// Headline regression. Through the wall the flux entering at side 1 is
/// `(T_eq − T_out)/(1/(h_c + h_rad) + R_wall + 1/h_ext)` and it is what `report()`
/// must state for the interior (`q_surf_int`), equal to the sum of the two films the
/// solver applies and to minus the exterior flux.
#[test]
fn steady_total_interior_flux_is_conv_plus_rad_and_matches_parallel_film_u() -> Result<()> {
    let room = Room::winter();
    let input = inputs14(&room);
    let s = steady_state(manager_1d(10.0, 1.0, &input)?, &input, STEADY_HOURS)?;

    let h_rad = h_rad_applied(&s.films_int);
    let cf = closed_form(&room, h_rad, s.r_wall);
    let q_applied = applied_film_flux(s.t_si, &s.films_int);
    let q_conv = convective_film_flux(s.t_si, &s.films_int);
    let q_rad = q_applied - q_conv;
    println!(
        "  steady: T_air {} / T_MR {} / T_out {} C, h_c {} + h_rad {:.4} W/m2K, R_wall {:.5} m2K/W\n\
         \x20   reported q_int = {:.5} W/m2 (closed form {:.5}, dev {:.1e}); films: conv {:.5} (closed form {:.5}) \
         + rad {:.5} (closed form {:.5}); reported q_ext = {:.5} W/m2 (|q_ext|/q_int − 1 = {:.1e})\n\
         \x20   T_si = {:.4} C (closed form {:.4})",
        room.t_air_c,
        room.t_mr_c,
        room.t_out_c,
        room.h_conv_w,
        h_rad,
        s.r_wall,
        s.q_int,
        cf.q,
        rel(s.q_int, cf.q),
        q_conv,
        cf.q_conv,
        q_rad,
        cf.q_rad,
        s.q_ext,
        rel(-s.q_ext, s.q_int),
        s.t_si,
        cf.t_s
    );

    // Structure: exactly two active interior films (the Radiation cell is ACTIVE in a
    // 14-input deck) and its coefficient is the solver's linearisation.
    assert_eq!(
        active_films(&s.films_int),
        2,
        "interior films of a 14-input deck"
    );
    assert!(
        rel(h_rad, h_rad_expected(s.t_s_mat, room.t_mr_c)) < 1e-6,
        "interior Radiation cell h = {h_rad:.6} W/m2K is not 4·σ·ε·T_m³ = {:.6}",
        h_rad_expected(s.t_s_mat, room.t_mr_c)
    );

    // The two films carry different fluxes because T_air ≠ T_MR (the discriminating part).
    assert!(
        q_rad > 1.0 && q_rad < q_conv,
        "radiative film flux {q_rad:.4} W/m2 should be a sizeable fraction of the convective {q_conv:.4}"
    );

    // 1. The interior flux is the flux of BOTH films the solver applies to the node.
    assert!(
        rel(s.q_int, q_applied) < 1e-9,
        "reported interior flux {:.6} W/m2 is not the sum of the applied films {:.6} W/m2 \
         (convective film alone: {:.6} W/m2) — output 8 must report conv + rad",
        s.q_int,
        q_applied,
        q_conv
    );
    // 2. …and is the parallel-film U flux (measured deviation ≈ 8e-6: the closed form takes
    //    R_wall from the final λ(w) while the wall's slow vapour redistribution, with its
    //    latent heat, is still settling; 1e-4 leaves an order of magnitude).
    assert!(
        rel(s.q_int, cf.q) < CLOSED_FORM_TOL,
        "reported interior flux {:.5} W/m2 differs by {:.3} % from the parallel-film closed form {:.5} W/m2",
        s.q_int,
        100.0 * rel(s.q_int, cf.q),
        cf.q
    );
    // …each film carries its own share: h_c·(T_air − T_s) and h_rad·(T_MR − T_s).
    assert!(
        rel(q_conv, cf.q_conv) < CLOSED_FORM_TOL && rel(q_rad, cf.q_rad) < CLOSED_FORM_TOL,
        "film fluxes conv {q_conv:.5} / rad {q_rad:.5} W/m2 differ from the closed form \
         {:.5} / {:.5} W/m2",
        cf.q_conv,
        cf.q_rad
    );
    // 3. Energy conservation of the REPORTED fluxes at steady state, no side gains
    //    (measured 9e-7).
    assert!(
        rel(-s.q_ext, s.q_int) < BALANCE_TOL,
        "reported exterior flux {:.5} W/m2 is not minus the interior flux {:.5} W/m2 at steady state",
        s.q_ext,
        s.q_int
    );
    Ok(())
}

/// The interior surface node is massless: what the two films deliver to it is what the wall's
/// first material cell receives by conduction, in a transient too. With a room whose air is
/// as warm as the wall and whose radiant environment is cold, the wall LOSES heat to the radiant
/// environment: the net flux into the wall is negative — the convective film alone (air warmer
/// than the surface) reported the opposite sign.
#[test]
fn cold_radiant_environment_gives_a_negative_total_equal_to_conduction_into_the_wall() -> Result<()>
{
    let room = Room {
        t_air_c: 20.0,
        t_mr_c: 10.0,
        t_out_c: 20.0, // exterior in equilibrium with the wall: no exterior flux to speak of
        h_conv_w: 8.65,
        h_ext_w: 10.0,
        q_wg1_w: 0.0,
    };
    let input = inputs14(&room);
    let area = 1.0;
    let mut mgr = manager_1d(20.0, area, &input)?;
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    run_hours(&mut mgr, &mut adaptive, &input, 1)?;
    let s = snapshot(&mgr, &adaptive)?;

    // Conduction from the first interior material cell into the surface node:
    // flux INTO the wall = G·(T_node − T_cell), G = λ/(dx/2).
    let i_mat = mgr.surface_curr.interior_surface_material_idx.unwrap();
    let c1 = &mgr.surface_curr.cells[[i_mat, 0, 0]];
    let lambda = c1.w_thermal_c.value_in(w_per_m_k); // strip: test-side scalar bookkeeping
    let half_dx = 0.5 * c1.size.x().value_in(meter);
    let q_conduction = lambda / half_dx * (s.t_si - s.t_s_mat);

    let q_applied = applied_film_flux(s.t_si, &s.films_int);
    let q_conv = convective_film_flux(s.t_si, &s.films_int);
    println!(
        "  cold-radiant step (T_air 20, T_MR 10, wall 20 C), 1 h: T_si = {:.4} C, films: conv {:.4} + rad {:.4} = {:.4} W/m2; \
         conduction into the wall {:.4} W/m2; reported q_int {:.4} W/m2 (dev from conduction {:.1e}), time-averaged {:.4} W/m2",
        s.t_si,
        q_conv,
        q_applied - q_conv,
        q_applied,
        q_conduction,
        s.q_int,
        rel(s.q_int, q_conduction),
        s.acc_q_int
    );

    assert!(
        q_conv > 0.0,
        "the warm air heats the surface node (sanity of the set-up)"
    );
    assert!(
        s.q_int < 0.0,
        "net interior flux must be NEGATIVE (the wall loses heat to the cold radiant environment), \
         got {:.4} W/m2 (the convective film alone is {:+.4} W/m2)",
        s.q_int,
        q_conv
    );
    assert!(
        rel(s.q_int, q_conduction) < 1e-4,
        "reported interior flux {:.6} W/m2 is not the conduction into the first wall cell {:.6} W/m2",
        s.q_int,
        q_conduction
    );
    // The time-average behind output 8 is taken over substeps of the same decaying transient.
    assert!(
        s.acc_q_int < 0.0,
        "time-averaged interior flux (output 8) must be negative too, got {:.4} W/m2",
        s.acc_q_int
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// (b) the value that reaches output 8 (accumulator path)
// ---------------------------------------------------------------------------

/// Type 395 output 8 = `(acc.avg_q_int() · area)` in kJ/h while the accumulator holds a solve
/// (`total_time > 0`), `report().q_surf_int · area` before. Both must be conv + rad; at a
/// steady state they are the same number, and output 8 + output 9 = 0.
#[test]
fn accumulator_path_behind_output_8_is_conv_plus_rad() -> Result<()> {
    let room = Room::winter();
    let input = inputs14(&room);
    let area = 2.5; // ≠ 1 so a missing / doubled area factor shows up
    let s = steady_state(manager_1d(10.0, area, &input)?, &input, STEADY_HOURS)?;

    assert!(
        s.acc_total_time_s > 0.0,
        "the accumulator must hold the last solve (the output-8 path, not the fallback)"
    );

    // What the DLL writes to output 8 / 9 [kJ/h], and the value of its fallback path.
    let (output_8, output_9, fallback_kj_h) = (s.output_8_kj_h, s.output_9_kj_h, s.fallback_8_kj_h);
    // The same quantity from the films the solver applied (independent of `heat_flux`).
    let films_kj_h = applied_film_flux(s.t_si, &s.films_int) * area * KJ_PER_H_PER_W;
    let conv_only_kj_h = convective_film_flux(s.t_si, &s.films_int) * area * KJ_PER_H_PER_W;
    println!(
        "  output 8 = {output_8:.4} kJ/h (films {films_kj_h:.4}, convective film alone {conv_only_kj_h:.4}, \
         fallback {fallback_kj_h:.4}); output 9 = {output_9:.4} kJ/h"
    );

    assert!(
        rel(output_8, films_kj_h) < 1e-9,
        "output 8 = {output_8:.4} kJ/h is not the total interior film flux {films_kj_h:.4} kJ/h \
         (the convective film alone is {conv_only_kj_h:.4} kJ/h)"
    );
    assert!(
        rel(fallback_kj_h, films_kj_h) < 1e-9,
        "fallback value {fallback_kj_h:.4} kJ/h is not the total interior film flux {films_kj_h:.4} kJ/h"
    );
    assert!(
        rel(-output_9, output_8) < BALANCE_TOL,
        "output 8 {output_8:.4} kJ/h and output 9 {output_9:.4} kJ/h do not balance at steady state"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// (e) qWG1 is not part of the film flux
// ---------------------------------------------------------------------------

/// `qWG1` is deposited in the interior surface material cell: the heat entering the wall from
/// side 1 is `output 8 + qWG1`, and the whole wall balances: `q_int + qWG1/A + q_ext = 0`.
#[test]
fn side_1_gain_qwg1_is_not_part_of_the_film_flux() -> Result<()> {
    const Q_WG1: f64 = 15.0; // W on 1 m²
    let room = Room {
        q_wg1_w: Q_WG1,
        ..Room::winter()
    };
    let input = inputs14(&room);
    let s = steady_state(manager_1d(10.0, 1.0, &input)?, &input, STEADY_HOURS)?;

    let q_applied = applied_film_flux(s.t_si, &s.films_int);
    let q_conv = convective_film_flux(s.t_si, &s.films_int);
    let balance = s.q_int + Q_WG1 / s.area + s.q_ext;
    println!(
        "  qWG1 = {Q_WG1} W/m2: reported q_int = {:.5} W/m2 (films {:.5}, convective alone {:.5}), \
         q_ext = {:.5} W/m2, q_int + qWG1 + q_ext = {:.6} W/m2",
        s.q_int, q_applied, q_conv, s.q_ext, balance
    );
    assert!(
        rel(s.q_int, q_applied) < 1e-9,
        "reported interior flux {:.6} W/m2 must be the film flux {:.6} W/m2 (qWG1 excluded, \
         radiation film included; convective film alone {:.6} W/m2)",
        s.q_int,
        q_applied,
        q_conv
    );
    assert!(
        balance.abs() < BALANCE_TOL * s.q_int.abs(),
        "heat entering the wall from side 1 (output 8 + qWG1) must leave at side 2: \
         q_int + qWG1 + q_ext = {balance:.6} W/m2"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// (f) 2-D lattice (per-column boundary sheets, global T_MR / h_rad)
// ---------------------------------------------------------------------------

#[test]
fn two_d_wall_total_interior_flux_matches_the_parallel_film_u() -> Result<()> {
    let room = Room::winter();
    let input = inputs14(&room);
    let library = create_library(10.0, 0.5, vec![THICKNESS_M], vec![Some(10)]);
    let construction = library.get_construction(CONSTRUCTION_NAME)?;
    let mut mgr = HamtManager::new_2d(
        Some(settings()),
        Some(BoundaryCondition::from_trnsys_inputs(&input)),
        construction,
        "interior_film_2d".to_string(),
        1.0 * square_meter,
        2,
        1.0 * meter,
    )?;
    let mut adaptive: AdaptiveState<()> = AdaptiveState::default();
    run_hours(&mut mgr, &mut adaptive, &input, STEADY_HOURS)?;
    let s = snapshot(&mgr, &adaptive)?;

    let h_rad = h_rad_applied(&s.films_int);
    let cf = closed_form(&room, h_rad, s.r_wall);
    let q_conv = convective_film_flux(s.t_si, &s.films_int);
    println!(
        "  2-D (2 columns): reported q_int = {:.5} W/m2 (closed form {:.5}, dev {:.1e}; convective film alone {:.5}), q_ext = {:.5} W/m2 (|q_ext|/q_int − 1 = {:.1e})",
        s.q_int,
        cf.q,
        rel(s.q_int, cf.q),
        q_conv,
        s.q_ext,
        rel(-s.q_ext, s.q_int)
    );
    assert!(
        rel(s.q_int, cf.q) < CLOSED_FORM_TOL,
        "2-D reported interior flux {:.5} W/m2 differs by {:.3} % from the parallel-film closed form {:.5} W/m2",
        s.q_int,
        100.0 * rel(s.q_int, cf.q),
        cf.q
    );
    assert!(
        rel(-s.q_ext, s.q_int) < BALANCE_TOL,
        "2-D reported exterior flux {:.5} W/m2 is not minus the interior flux {:.5} W/m2",
        s.q_ext,
        s.q_int
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// (c) legacy 12-input mode and (d) the exterior side are UNCHANGED
// ---------------------------------------------------------------------------

/// Legacy decks: ONE combined interior film to TSTAR (`hconv1`), the interior Radiation cell is
/// inert (`h = 0`) — the total interior flux IS that single film, exactly as before.
#[test]
fn legacy_12_input_interior_flux_is_the_single_combined_film() -> Result<()> {
    const H_IN: f64 = 10.0;
    const H_OUT: f64 = 10.0;
    let input = inputs12_legacy(20.0, 0.0, H_IN, H_OUT);
    let area = 2.5;
    let s = steady_state(manager_1d(10.0, area, &input)?, &input, STEADY_HOURS)?;

    let q_closed = (20.0 - 0.0) / (1.0 / H_IN + s.r_wall + 1.0 / H_OUT);
    let q_film = applied_film_flux(s.t_si, &s.films_int);
    println!(
        "  legacy 12-input: q_int = {:.5} W/m2 (closed form {:.5}, single film {:.5}), accumulator {:.5} W/m2, \
         q_ext = {:.5} W/m2; active interior films = {}",
        s.q_int,
        q_closed,
        q_film,
        s.acc_q_int,
        s.q_ext,
        active_films(&s.films_int)
    );
    assert_eq!(
        active_films(&s.films_int),
        1,
        "legacy decks have exactly one active interior film (the Radiation cell is inert)"
    );
    assert_eq!(
        h_rad_applied(&s.films_int),
        0.0,
        "interior Radiation cell must be inert"
    );
    assert!(
        rel(s.q_int, q_film) < 1e-12,
        "legacy interior flux {:.9} W/m2 is not the single combined film {:.9} W/m2",
        s.q_int,
        q_film
    );
    assert!(
        rel(s.acc_q_int, q_film) < 1e-9,
        "legacy accumulator flux {:.9} W/m2 is not the single combined film {:.9} W/m2",
        s.acc_q_int,
        q_film
    );
    assert!(
        rel(s.q_int, q_closed) < CLOSED_FORM_TOL,
        "legacy interior flux {:.5} W/m2 differs from the single-film closed form {:.5} W/m2",
        s.q_int,
        q_closed
    );
    Ok(())
}

/// Side 2 has exactly one (convective) film in every deck type; its reported flux is that film,
/// also in a 14-input deck, and the inert placeholders contribute nothing.
#[test]
fn exterior_flux_is_the_single_convective_film_in_a_14_input_deck() -> Result<()> {
    let room = Room::winter();
    let input = inputs14(&room);
    let area = 2.5;
    let s = steady_state(manager_1d(10.0, area, &input)?, &input, STEADY_HOURS)?;

    let q_film_ext = applied_film_flux(s.t_so, &s.films_ext);
    let expected = room.h_ext_w * (room.t_out_c - s.t_so);
    println!(
        "  exterior: reported q_ext = {:.6} W/m2 (film h·(T_out − T_so) = {:.6}), accumulator {:.6}, active films = {}",
        s.q_ext,
        expected,
        s.acc_q_ext,
        active_films(&s.films_ext)
    );
    assert_eq!(
        active_films(&s.films_ext),
        1,
        "exactly one active exterior film (Convection); Radiation / Sky / Ground are inert"
    );
    assert!(
        rel(s.q_ext, expected) < 1e-12,
        "reported exterior flux {:.9} W/m2 is not h_ext·(T_out − T_so) = {:.9} W/m2",
        s.q_ext,
        expected
    );
    assert!(
        rel(s.q_ext, q_film_ext) < 1e-12,
        "reported exterior flux {:.9} W/m2 is not the applied exterior film {:.9} W/m2",
        s.q_ext,
        q_film_ext
    );
    assert!(
        rel(s.acc_q_ext, s.q_ext) < 1e-9,
        "accumulated exterior flux {:.9} W/m2 differs from the end-point value {:.9} W/m2 at steady state",
        s.acc_q_ext,
        s.q_ext
    );
    Ok(())
}
