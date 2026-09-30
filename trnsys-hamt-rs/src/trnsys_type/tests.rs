//! Output mapping of the Type (`TrnSysType::update_output`), without the TRNSYS kernel.
//!
//! The wall is solved with the same calls `iterate` makes (boundary condition from the input
//! vector, `solve_timestep_adaptive` on the state's adaptive state, `tick_timestep`); only
//! `update_output(.., write_file = false)` — which never touches the TRNSYS C API — maps the
//! solver state to the outputs, and that is what is under test: Type 395 output 8
//! ("HeatFlux_in") is the TOTAL interior film heat flux, convective film to the air PLUS the
//! longwave film to TMR in a 14-input deck, in kJ/h for the whole surface.
//!
//! Expected values are built from the boundary cells' own films (`h·(T_env − T_surface)`), not
//! from `Surface::heat_flux`, so the tests check the hand-over rather than restate it. Output 8
//! is the time average over the macro step of the end-point flux of each substep, so in a
//! transient it lies between the flux at the start and at the end of the step.

use super::*;
use hamt_core::boundary::BoundaryCondition;
use hamt_core::cell::{Boundary, BoundaryLocation, CellType};
use hamt_core::uom::units::{hours, square_meter};

/// Production wall (DIN GKP with foil, 1×25 mm gypsum), shared with the hamt-core tests.
const WALL_IDF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../hamt-core/fixtures/gkp_withfoil_1x25.idf"
);
const AREA_M2: f64 = 2.5;
/// 1 W in kJ/h.
const KJ_PER_H_PER_W: f64 = 3.6;

/// 14-input deck: air 22 °C, radiant environment 20 °C, HCONVI = 8.65 W/(m²·K)
/// (31.14 kJ/(h·m²·K)), exterior 0 °C. The surface sits at ≈ 21 °C: the convective film heats
/// it (≈ +7 W/m²), the longwave film cools it (≈ −5 W/m²), the net is the ≈ 2 W/m² the wall takes.
fn inputs_14() -> [f64; 14] {
    [
        25.0,     // 0  TSTAR (not used by the films of a 14-input deck)
        0.0,      // 1  TB2
        50.0,     // 2  RH1 [%]
        70.0,     // 3  RH2 [%]
        63.475,   // 4  hconv1, combined (not used when input 12 is present)
        64.0,     // 5  hconv2 [kJ/(h m2 K)]
        0.0,      // 6  qWG1
        0.0,      // 7  qWG2
        6.057e-8, // 8  beta1
        1.2e-8,   // 9  beta2
        0.0,      // 10 user-defined heat
        22.0,     // 11 TAIR
        31.14,    // 12 HCONVI [kJ/(h m2 K)]
        20.0,     // 13 TMR
    ]
}

/// Legacy 12-input deck: ONE combined interior film (8.65 W/(m²·K) to TSTAR = 22 °C).
fn inputs_12() -> [f64; 12] {
    [
        22.0, 0.0, 50.0, 70.0, 31.14, 64.0, 0.0, 0.0, 6.057e-8, 1.2e-8, 0.0, 22.0,
    ]
}

fn state_with_wall(first_inputs: &[f64]) -> TrnSysState {
    let library = load_hamt_by_path(WALL_IDF).expect("fixture IDF");
    let construction = library
        .get_construction_read_lock()
        .expect("construction lock")
        .values()
        .next()
        .cloned()
        .expect("construction");
    let settings = HamtSettings {
        timestep: 1.0 * hours,
        n_div: 10,
        ..Default::default()
    };
    let manager = HamtManager::new(
        Some(settings),
        Some(BoundaryCondition::from_trnsys_inputs(first_inputs)),
        construction,
        "output_mapping_test".to_string(),
        AREA_M2 * square_meter,
    )
    .expect("manager");
    let mut state = TrnSysState::new();
    state.hamt_manager = Some(manager);
    state
}

/// What `iterate` + `end_of_timestep` do to the wall for one hourly macro step.
fn macro_step(state: &mut TrnSysState, input: &[f64], hour: usize) {
    let manager = state.hamt_manager.as_mut().expect("manager");
    manager.boundary_condition = BoundaryCondition::from_trnsys_inputs(input);
    state.adaptive_state.current_time = (hour as f64) * hours;
    manager
        .solve_timestep_adaptive(&mut state.adaptive_state, None)
        .expect("solve");
    manager.tick_timestep().expect("tick");
}

/// Σ over the ACTIVE films of one face of `h·(T_env − T_surface)` [W/m²] (column 0,0) — what the
/// solver applies to the surface node; and the convective film alone.
fn film_fluxes(state: &TrnSysState, loc: BoundaryLocation) -> (f64, f64) {
    let manager = state.hamt_manager.as_ref().expect("manager");
    let mut t_node = None;
    let mut films: Vec<(Boundary, f64, f64)> = Vec::new(); // (kind, h [W/m2K], T_env [K])
    for ((_, j, k), cell) in manager.surface_curr.cells.indexed_iter() {
        if j != 0 || k != 0 {
            continue;
        }
        match &cell.cell_type {
            CellType::Virtual(l) if *l == loc => {
                t_node = Some(cell.cell_temp.value_in(hamt_core::uom::units::kelvin))
            }
            CellType::Boundary(kind, l) if *l == loc => films.push((
                kind.clone(),
                cell.htc.value_in(w_per_m2_k),
                cell.cell_temp.value_in(hamt_core::uom::units::kelvin),
            )),
            _ => {}
        }
    }
    let t_node = t_node.expect("surface node");
    let total: f64 = films.iter().map(|(_, h, t)| h * (t - t_node)).sum();
    let conv: f64 = films
        .iter()
        .filter(|(kind, _, _)| *kind == Boundary::Convection)
        .map(|(_, h, t)| h * (t - t_node))
        .sum();
    (total, conv)
}

fn output_8_w_per_m2(state: &TrnSysState) -> f64 {
    state.outputs[7].value / (AREA_M2 * KJ_PER_H_PER_W)
}

fn output_9_w_per_m2(state: &TrnSysState) -> f64 {
    state.outputs[8].value / (AREA_M2 * KJ_PER_H_PER_W)
}

fn between(x: f64, a: f64, b: f64, eps_rel: f64) -> bool {
    let (lo, hi) = (a.min(b), a.max(b));
    let eps = eps_rel * hi.abs().max(lo.abs());
    x >= lo - eps && x <= hi + eps
}

/// 14-input deck: output 8 is conv + rad. Start-of-step and end-of-step film fluxes bracket the
/// time average that the accumulator hands over; the convective film alone (what output 8 used
/// to be) is several times larger here and far outside the bracket. The fallback (accumulator
/// empty) is the end-of-step total exactly.
#[test]
fn output_8_is_the_total_interior_film_flux_in_a_14_input_deck() {
    let input = inputs_14();
    let mut state = state_with_wall(&input);
    let ty = TrnSysType::new();

    // Spin up (a slow transient of the insulated wall), then look at ONE more macro step.
    for h in 0..47 {
        macro_step(&mut state, &input, h);
    }
    let (start_total, start_conv) = film_fluxes(&state, BoundaryLocation::Interior);
    macro_step(&mut state, &input, 47);
    let (end_total, end_conv) = film_fluxes(&state, BoundaryLocation::Interior);

    ty.update_output(&mut state, false).expect("update_output");
    let out8 = output_8_w_per_m2(&state);
    println!(
        "  output 8 = {:.6} W/m2 = {:.4} kJ/h for {AREA_M2} m2; films start {:.6} / end {:.6} W/m2 \
         (convective film alone: {:.6} / {:.6})",
        out8, state.outputs[7].value, start_total, end_total, start_conv, end_conv
    );
    assert!(
        state
            .adaptive_state
            .flux_accumulator
            .total_time
            .value_in(hamt_core::uom::units::seconds)
            > 0.0,
        "the accumulator must hold the solve (output-8 path, not the fallback)"
    );
    assert!(
        (end_total - end_conv).abs() > 0.2 * end_total.abs() && end_total > 0.0,
        "set-up: the longwave film must be a sizeable part ({end_total:.4} vs {end_conv:.4} W/m2)"
    );
    assert!(
        between(out8, start_total, end_total, 1e-9),
        "output 8 = {out8:.6} W/m2 is not the total interior film flux (start {start_total:.6}, \
         end {end_total:.6} W/m2; convective film alone {end_conv:.6} W/m2)"
    );

    // Fallback path: accumulator empty -> end-point report, the total film flux of the end state.
    state.adaptive_state.flux_accumulator.reset();
    ty.update_output(&mut state, false).expect("update_output");
    let fallback = output_8_w_per_m2(&state);
    assert!(
        (fallback - end_total).abs() <= 1e-9 * end_total.abs(),
        "fallback output 8 = {fallback:.9} W/m2 is not the end-state total film flux {end_total:.9} W/m2"
    );
}

/// Output 9 (exterior) is the single convective exterior film in every deck: unchanged.
#[test]
fn output_9_is_the_single_exterior_film() {
    let input = inputs_14();
    let mut state = state_with_wall(&input);
    let ty = TrnSysType::new();
    for h in 0..47 {
        macro_step(&mut state, &input, h);
    }
    let (start, _) = film_fluxes(&state, BoundaryLocation::Exterior);
    macro_step(&mut state, &input, 47);
    let (end, end_conv) = film_fluxes(&state, BoundaryLocation::Exterior);
    ty.update_output(&mut state, false).expect("update_output");
    let out9 = output_9_w_per_m2(&state);
    assert_eq!(
        end, end_conv,
        "exterior: the convective film is the only active film"
    );
    assert!(
        between(out9, start, end, 1e-9),
        "output 9 = {out9:.6} W/m2 is not the exterior film flux (start {start:.6}, end {end:.6} W/m2)"
    );
}

/// Legacy 12-input deck: output 8 is the single combined film, as before.
#[test]
fn output_8_in_a_legacy_12_input_deck_is_the_single_combined_film() {
    let input = inputs_12();
    let mut state = state_with_wall(&input);
    let ty = TrnSysType::new();
    for h in 0..47 {
        macro_step(&mut state, &input, h);
    }
    let (start, _) = film_fluxes(&state, BoundaryLocation::Interior);
    macro_step(&mut state, &input, 47);
    let (end, end_conv) = film_fluxes(&state, BoundaryLocation::Interior);
    ty.update_output(&mut state, false).expect("update_output");
    let out8 = output_8_w_per_m2(&state);
    assert_eq!(
        end, end_conv,
        "legacy deck: the single combined film is the only active one"
    );
    assert!(
        between(out8, start, end, 1e-9),
        "output 8 = {out8:.6} W/m2 is not the combined film flux (start {start:.6}, end {end:.6} W/m2)"
    );
}
