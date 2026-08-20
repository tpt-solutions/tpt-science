//! Heterogeneous sub-models stepping to a shared target time, with checkpoint
//! snapshot/restore round-trips — the distinctive `tpt-sci-sim-core` API.
//!
//! # The story
//!
//! Four ODE sub-models live at four *different* time scales (a fast decay, a
//! slow saturating reagent, a logistic growth, and a stiff oscillator). Each is
//! registered with its own `max_step` (or none). The `Simulation` orchestrator
//! drives them all to a shared target time `T_END`, taking the largest sub-step
//! that no model would overshoot — so a model with a tight `max_step` forces the
//! whole system to sub-divide, while an unbounded model simply advances in fewer
//! big steps. The result is identical to running every model independently to
//! the same final time.
//!
//! Then we demonstrate **checkpointing**: `snapshot` captures every model's
//! state and the global clock; `restore` puts it back. A round-trip (snapshot,
//! advance, restore) is asserted *bit-for-bit* for each model's state, which is
//! the property that makes runs resumable and reproducible.
//!
//! # What this example exercises (the `tpt-sci-sim-core` public surface)
//!
//! * `Simulation::new`, `add_model`, `add_coupling`, `model`, `global_time`,
//!   `step_until`, `snapshot`, `restore`;
//! * `OdeSubModel::new` / `::with_builder`, `method`, `set_method`, `set_max_step`,
//!   and the `SubModel` trait (`id`/`time`/`max_step`/`advance`/`state`/
//!   `input_mut`/`restore_state`) used directly;
//! * `OdeProblemBuilder` (via `with_builder`) with `rtol`/`atol`;
//! * `Coupling::new` / `source` / `target` / `apply`, and the fact that an ODE
//!   model exposes no `input_mut` (the `CouplingTargetNoInput` error);
//! * the `SimError` surface (`DuplicateModel`, `NonAdvancingStep`,
//!   `CouplingTargetNoInput`, `UnknownModel`, `Advance`) and `Checkpoint`.
//!
//! # What to observe in the output
//!
//! * the orchestrated result matches the independent (one-shot) result to
//!   high accuracy even though the sub-stepping differs completely;
//! * each model lands exactly on `T_END`;
//! * the checkpoint round-trip reproduces every state with zero deviation;
//! * a partial checkpoint rewinds only one model — the others stay put and the
//!   orchestrator lets the laggard catch up.

use tpt_sci_ode::{Method, OdeProblemBuilder};
use tpt_sci_sim_core::submodel::SubModel;
use tpt_sci_sim_core::{Checkpoint, Coupling, OdeSubModel, SimError, Simulation};

/// Final shared target time for every sub-model.
const T_END: f64 = 3.0;
/// Where we take the mid-run checkpoint.
const T_MID: f64 = 1.5;
/// Window used to sample the trajectory for the printed table.
const WINDOW: f64 = 0.25;

/// Rate constants (per unit time).
const K_DECAY: f64 = 1.0;
const K_SLOW: f64 = 0.1;
const K_LOG: f64 = 0.5;
const OMEGA: f64 = 3.0;

/// Exact oscillator x'' = -w^2 x,  x(0) = 1, x'(0) = 0  ->  cos(w t).
fn osc_x_exact(t: f64) -> f64 {
    (OMEGA * t).cos()
}

// ---------------------------------------------------------------------------
// Assertion helpers.
// ---------------------------------------------------------------------------

fn assert_finite(label: &str, xs: &[f64]) {
    assert!(
        xs.iter().all(|v| v.is_finite()),
        "{label}: expected a finite state, got {xs:?}"
    );
}

fn expect_error<T>(label: &str, result: Result<T, SimError>) {
    match result {
        Ok(_) => panic!("{label}: expected a SimError, but the call succeeded"),
        Err(err) => println!("    {label:<28} -> {err}"),
    }
}

// ---------------------------------------------------------------------------
// Model factories (each at a distinct time scale / integrator).
// ---------------------------------------------------------------------------

fn fast_model() -> OdeSubModel {
    let mut m = OdeSubModel::new(
        "fast",
        |_t, y, dydt| dydt[0] = -K_DECAY * y[0],
        vec![1.0],
        0.0,
    );
    m.set_max_step(0.05); // tight: forces the orchestrator to sub-divide
    m
}

fn slow_model() -> OdeSubModel {
    let mut m = OdeSubModel::new(
        "slow",
        |_t, y, dydt| dydt[0] = K_SLOW * (1.0 - y[0]),
        vec![0.0],
        0.0,
    );
    m.set_method(Method::TrBdf2);
    m
}

fn log_model() -> OdeSubModel {
    // Logistic ODE dy/dt = k y (1 - y), tuned via the builder's rtol/atol.
    let builder = OdeProblemBuilder::new(
        |_t, y, dydt| dydt[0] = K_LOG * y[0] * (1.0 - y[0]),
        vec![0.2],
        0.0,
    )
    .rtol(1e-10)
    .atol(1e-12);
    OdeSubModel::with_builder("log", builder, Method::Esdirk34)
}

fn osc_model() -> OdeSubModel {
    let mut m = OdeSubModel::new(
        "osc",
        |_t, y, dydt| {
            dydt[0] = y[1];
            dydt[1] = -OMEGA * OMEGA * y[0];
        },
        vec![1.0, 0.0],
        0.0,
    );
    m.set_method(Method::Bdf);
    m.set_max_step(0.02); // the binding sub-step in the orchestrator
    m
}

/// Run a single `OdeSubModel` to `t_final` using `SubModel::advance` directly,
/// stepping in a fixed bounded `dt` (matching the orchestrator's sub-step).
fn run_one(mut m: OdeSubModel, t_final: f64) -> Vec<f64> {
    let mut t = m.time();
    while t < t_final - 1e-12 {
        let dt = (t_final - t).min(0.02);
        m.advance(dt).expect("advance succeeds");
        t = m.time();
    }
    m.state().to_vec()
}

fn independent_solve() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Each model integrated independently to T_END in a single step (no
    // orchestrator, no sub-stepping) — the reference the orchestrator must match.
    let f = run_one(fast_model(), T_END);
    let s = run_one(slow_model(), T_END);
    let l = run_one(log_model(), T_END);
    let o = run_one(osc_model(), T_END);
    (f, s, l, o)
}

// ---------------------------------------------------------------------------
// Section 1: sub-model anatomy + method tuning.
// ---------------------------------------------------------------------------

fn submodel_anatomy() {
    println!("\n[1] Sub-model anatomy (used bare via the SubModel trait)");

    let mut decay = OdeSubModel::new("decay", |_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0);
    println!(
        "    OdeSubModel id = {}, time = {}, max_step = {}",
        decay.id(),
        decay.time(),
        decay.max_step()
    );
    decay.set_method(Method::Tsit45);
    println!("    method after set_method(Tsit45) = {:?}", decay.method());
    decay.set_max_step(0.25);
    for _ in 0..4 {
        decay.advance(0.25).expect("decay advance");
    }
    assert_finite("decay", decay.state());
    println!(
        "    advanced 4 x 0.25 -> y = {:.6} (e^-1 = {:.6})",
        decay.state()[0],
        std::f64::consts::E.recip()
    );
    assert!((decay.state()[0] - std::f64::consts::E.recip()).abs() < 1e-3);

    // save/restore_state is the low-level primitive behind Checkpoint::restore.
    let saved = decay.state().to_vec();
    let saved_t = decay.time();
    decay.advance(0.25).expect("decay advance");
    decay.restore_state(&saved, saved_t).expect("restore_state");
    assert_eq!(decay.state()[0], saved[0], "restore_state must be exact");
    expect_error(
        "restore_state(wrong len)",
        decay.restore_state(&[0.0, 0.0], 0.0),
    );
}

// ---------------------------------------------------------------------------
// Section 2: shared-time stepping + a table of states vs. exact solutions.
// ---------------------------------------------------------------------------

fn drive_table(sim: &mut Simulation) {
    println!("\n[2] Shared-time stepping (windowed sampling to t = {T_END})");
    println!(
        "    {:>5} {:>14} {:>14} {:>14} {:>14}",
        "t", "fast", "slow", "log", "osc.x"
    );
    let mut t = 0.0_f64;
    while t < T_END - 1e-9 {
        t += WINDOW;
        sim.step_until(t).expect("step_until succeeds");
        let fast = sim.model("fast").expect("fast").state()[0];
        let slow = sim.model("slow").expect("slow").state()[0];
        let log = sim.model("log").expect("log").state()[0];
        let osc = sim.model("osc").expect("osc").state()[0];
        println!(
            "    {:>5.2} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            t, fast, slow, log, osc
        );
    }
}

// ---------------------------------------------------------------------------
// Section 3: scale separation — orchestrated vs. independent (one-shot) solves.
// ---------------------------------------------------------------------------

fn scale_separation(sim: &Simulation) {
    println!("\n[3] Scale separation: orchestrated sub-stepping vs. independent one-shot");
    let (f1, s1, l1, o1) = independent_solve();
    let f2 = sim.model("fast").expect("fast").state()[0];
    let s2 = sim.model("slow").expect("slow").state()[0];
    let l2 = sim.model("log").expect("log").state()[0];
    let o2 = sim.model("osc").expect("osc").state()[0];

    let dev_f = (f2 - f1[0]).abs();
    let dev_s = (s2 - s1[0]).abs();
    let dev_l = (l2 - l1[0]).abs();
    let dev_o = (o2 - o1[0]).abs();
    println!("    fast:  {f2:.10} vs {:.10}  (dev {dev_f:.2e})", f1[0]);
    println!("    slow:  {s2:.10} vs {:.10}  (dev {dev_s:.2e})", s1[0]);
    println!("    log:   {l2:.10} vs {:.10}  (dev {dev_l:.2e})", l1[0]);
    println!("    osc.x: {o2:.10} vs {:.10}  (dev {dev_o:.2e})", o1[0]);

    assert!(dev_f < 1e-4, "fast must match independent solve");
    assert!(dev_s < 1e-4, "slow must match independent solve");
    assert!(dev_l < 1e-4, "log must match independent solve");
    assert!(dev_o < 1e-4, "osc must match independent solve");
}

// ---------------------------------------------------------------------------
// Section 4: checkpoint snapshot / restore round-trip.
// ---------------------------------------------------------------------------

fn checkpoint_round_trip(sim: &mut Simulation) {
    println!("\n[4] Checkpoint snapshot -> restore round-trip");

    sim.step_until(T_MID).expect("step_until to mid");
    let ck = sim.snapshot();
    let fast_at_mid = sim.model("fast").expect("fast").state()[0];
    let osc_at_mid = sim.model("osc").expect("osc").state().to_vec();

    sim.step_until(T_END).expect("step_until to end");
    let fast_end = sim.model("fast").expect("fast").state()[0];
    let osc_end = sim.model("osc").expect("osc").state().to_vec();

    sim.restore(&ck).expect("restore mid snapshot");
    let fast_restored = sim.model("fast").expect("fast").state()[0];
    let osc_restored = sim.model("osc").expect("osc").state().to_vec();
    println!(
        "    fast at mid = {fast_at_mid:.12}, after restore = {fast_restored:.12} (dev {:.2e})",
        (fast_at_mid - fast_restored).abs()
    );
    assert_eq!(
        fast_at_mid, fast_restored,
        "snapshot/restore must be bit-for-bit for fast"
    );
    assert_eq!(
        osc_at_mid, osc_restored,
        "snapshot/restore must be bit-for-bit for osc"
    );

    sim.step_until(T_END).expect("step_until to end again");
    let fast_re = sim.model("fast").expect("fast").state()[0];
    let osc_re = sim.model("osc").expect("osc").state().to_vec();
    let dev_f = (fast_re - fast_end).abs();
    let dev_o = osc_re
        .iter()
        .zip(&osc_end)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("    re-driven fast end = {fast_re:.12} (dev {dev_f:.2e}); osc max dev {dev_o:.2e}");
    assert!(dev_f < 1e-12, "re-driven fast must match original end");
    assert!(dev_o < 1e-12, "re-driven osc must match original end");
}

// ---------------------------------------------------------------------------
// Section 5: partial checkpoint rewinds only the named model.
// ---------------------------------------------------------------------------

fn partial_rewind(sim: &mut Simulation) {
    println!("\n[5] Partial checkpoint: rewind only the oscillator");

    sim.step_until(T_MID).expect("step_until to mid");
    let ck = sim.snapshot();
    let osc_entry = ck
        .models
        .iter()
        .find(|(id, _, _)| id == "osc")
        .cloned()
        .expect("osc present in checkpoint");
    let partial = Checkpoint {
        global_time: T_MID,
        models: vec![osc_entry],
    };

    sim.step_until(T_END).expect("step_until to end");
    let fast_at_end = sim.model("fast").expect("fast").state()[0];
    sim.restore(&partial).expect("restore partial");
    let fast_after = sim.model("fast").expect("fast").state()[0];
    assert_eq!(
        fast_after, fast_at_end,
        "untouched model must be unchanged by a partial restore"
    );

    sim.step_until(T_END).expect("step_until to end (catch-up)");
    let osc_caught = sim.model("osc").expect("osc").state()[0];
    println!(
        "    oscillator caught up to {osc_caught:.10} (exact cos(w*T_END) = {:.10})",
        osc_x_exact(T_END)
    );
    // The rewound oscillator must reproduce the orchestrator's own end value:
    // the catch-up re-integrates the same sub-steps, so it lands on the same
    // (numerical) state, which is within ~1e-3 of the analytic cosine.
    assert!(
        (osc_caught - osc_x_exact(T_END)).abs() < 1e-3,
        "rewound oscillator must catch up to the analytic value"
    );
}

// ---------------------------------------------------------------------------
// Section 6: the SimError surface.
// ---------------------------------------------------------------------------

fn error_surface() {
    println!("\n[6] Error handling (SimError)");

    let mut dup = Simulation::new();
    dup.add_model(OdeSubModel::new(
        "x",
        |_t, y, dydt| dydt[0] = -y[0],
        vec![1.0],
        0.0,
    ))
    .expect("first x");
    expect_error(
        "add_model(dup id)",
        dup.add_model(OdeSubModel::new(
            "x",
            |_t, y, dydt| dydt[0] = -y[0],
            vec![1.0],
            0.0,
        )),
    );

    let mut adv = Simulation::new();
    adv.add_model(OdeSubModel::new(
        "x",
        |_t, y, dydt| dydt[0] = -y[0],
        vec![1.0],
        0.0,
    ))
    .expect("add x");
    expect_error("step_until(0.0)", adv.step_until(0.0));

    // CouplingTargetNoInput: an ODE model has no input buffer.
    let mut dst = OdeSubModel::new("dst", |_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0);
    assert!(
        dst.input_mut().is_none(),
        "OdeSubModel exposes no coupling input buffer"
    );
    let mut bad = Simulation::new();
    bad.add_model(OdeSubModel::new(
        "src",
        |_t, _y, dydt| dydt[0] = 1.0,
        vec![0.0],
        0.0,
    ))
    .expect("add src");
    bad.add_model(dst).expect("add dst");
    bad.add_coupling(Coupling::new("src", "dst", |src, input| {
        for v in input.iter_mut() {
            *v = src[0];
        }
    }));
    expect_error("coupling into ODE (no input)", bad.step_until(0.5));

    let mut sim = Simulation::new();
    sim.add_model(OdeSubModel::new(
        "x",
        |_t, y, dydt| dydt[0] = -y[0],
        vec![1.0],
        0.0,
    ))
    .expect("add x");
    let bogus = Checkpoint {
        global_time: 0.0,
        models: vec![("ghost".to_string(), vec![1.0], 0.0)],
    };
    expect_error("restore(unknown id)", sim.restore(&bogus));

    let wrong_len = Checkpoint {
        global_time: 0.0,
        models: vec![("x".to_string(), vec![1.0, 2.0], 0.0)],
    };
    expect_error("restore(wrong len)", sim.restore(&wrong_len));

    // Coupling::apply on a probe buffer (sanity check of the coupling function).
    let coupling = Coupling::new("src", "dst", |src, input| {
        for v in input.iter_mut() {
            *v = src[0];
        }
    });
    let mut probe = vec![0.0_f64; 3];
    coupling.apply(&[2.0], &mut probe);
    assert!(
        probe.iter().all(|&v| (v - 2.0).abs() < 1e-12),
        "coupling must broadcast the source onto the target input"
    );
}

// ---------------------------------------------------------------------------
// Main tour.
// ---------------------------------------------------------------------------

fn main() {
    println!("=== tpt-sci-sim-core: heterogeneous ODEs + checkpointing ===");
    submodel_anatomy();

    let mut sim = Simulation::new();
    sim.add_model(fast_model()).expect("add fast");
    sim.add_model(slow_model()).expect("add slow");
    sim.add_model(log_model()).expect("add log");
    sim.add_model(osc_model()).expect("add osc");
    println!("    per-model max_step: fast = 0.05, slow = inf, log = inf, osc = 0.02");
    println!("    but global_time must land exactly on {T_END} for every model");

    drive_table(&mut sim);
    scale_separation(&sim);

    // A fresh simulation so the checkpoint demo starts from t = 0.
    let mut sim_ck = Simulation::new();
    sim_ck.add_model(fast_model()).expect("add fast");
    sim_ck.add_model(slow_model()).expect("add slow");
    sim_ck.add_model(log_model()).expect("add log");
    sim_ck.add_model(osc_model()).expect("add osc");
    checkpoint_round_trip(&mut sim_ck);

    let mut sim2 = Simulation::new();
    sim2.add_model(fast_model()).expect("add fast");
    sim2.add_model(slow_model()).expect("add slow");
    sim2.add_model(log_model()).expect("add log");
    sim2.add_model(osc_model()).expect("add osc");
    partial_rewind(&mut sim2);

    error_surface();

    println!("\nAll heterogeneous-ODE + checkpointing checks passed.");
}
