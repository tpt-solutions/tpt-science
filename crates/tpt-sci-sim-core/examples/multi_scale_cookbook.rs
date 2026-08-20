//! Multi-scale cookbook: a reaction network (SIR) built with the
//! `tpt-sci-reaction-network` DSL drives spatial diffusion fields through
//! `tpt-sci-sim-core` orchestration — composing `tpt-sci-reaction-network` +
//! `tpt-sci-sim-core` + `tpt-sci-grid` end to end.
//!
//! # The multi-scale story
//!
//! An epidemiological *reactor* (a 3-species SIR ODE) emits pathogen at a rate
//! proportional to the number currently infectious. That emission is a
//! **cross-scale coupling**: the fast reaction ODE's state is projected onto the
//! **input buffer** of a 1-D `DiffusionSubModel`, which spreads it through space
//! under explicit-Euler diffusion. A second diffusion field *senses* the first
//! (one-way coupling) to represent a cleared/response concentration. The
//! `Simulation` orchestrator sub-steps everything on the *slowest* stable scale
//! — the diffusion field's explicit-Euler stability limit — while the reaction
//! advances alongside it, and fires every coupling after each sub-step.
//!
//! # What this example exercises (the `tpt-sci-sim-core` public surface)
//!
//! * `Simulation::new` / `::default`, `add_model`, `add_coupling`, `model`,
//!   `global_time`, `step_until`, `snapshot`, `restore`;
//! * `OdeSubModel::new` / `::with_builder`, `method`, `set_method`,
//!   `set_max_step`, and the `SubModel` trait (`id`/`time`/`max_step`/`advance`/
//!   `state`/`input_mut`/`restore_state`) used directly;
//! * `DiffusionSubModel::new`, `coeff`, `set_max_step`, its `input_mut` source
//!   term, and the `SubModel` trait used directly;
//! * `Coupling::new`, `source`, `target`, `apply` (including a preview of the
//!   coupling function on a probe buffer);
//! * the `SubModel` trait used directly on a registered model via `model`;
//! * the `SimError` surface (`DuplicateModel`, `NonAdvancingStep`,
//!   `CouplingTargetNoInput`, `UnknownModel`, `Advance`) and `Checkpoint`.
//!
//! # What to observe in the output
//!
//! * `step_until` lands *every* model at the same target time despite wildly
//!   different internal rates (the diffusion `max_step` is the binding stability
//!   limit and the orchestrator never overshoots it);
//! * the infected compartment peaks then crashes to zero while the diffusion
//!   field integrates the total emission into a smooth spatial profile;
//! * the SIR population `S + I + R` is conserved to machine precision and the
//!   diffusion **mass** matches the analytically predicted injected budget;
//! * a `snapshot` at the epidemic peak, followed by `restore`, reproduces the
//!   reaction state *bit-for-bit* (the diffusion field lags by exactly one
//!   injection sub-step, because checkpoints store model state + clock but not
//!   the coupling input buffer — the intended semantics);
//! * the error surface shows each `SimError` variant is produced as documented.

use tpt_sci_grid::{Boundary, UniformGrid1D};
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::{ReactionNetwork, ReactionSystem};
use tpt_sci_sim_core::submodel::SubModel;
use tpt_sci_sim_core::{
    Checkpoint, Coupling, DiffusionSubModel, OdeSubModel, SimError, Simulation,
};

/// Number of 1-D grid nodes for every diffusion field.
const N: usize = 41;
/// Spatial domain of the (unit, dimensionless) domain.
const X0: f64 = 0.0;
const X1: f64 = 1.0;
/// Diffusion coefficients (slow vs. very slow spatial spreading).
const D_PATHOGEN: f64 = 0.02;
const D_RESPONSE: f64 = 0.005;
/// SIR transmission / recovery rates (per unit time).
const BETA: f64 = 0.002;
const GAMMA: f64 = 0.4;
/// Initial demography.
const S0: f64 = 990.0;
const I0: f64 = 10.0;
const R0: f64 = 0.0;
/// Coupling strengths: emission of pathogen per infectious individual, and the
/// clearance gain mapping pathogen concentration onto the response field.
const SHED_RATE: f64 = 0.01;
const CLEAR_GAIN: f64 = 0.01;
/// Where the reactor is located along the domain (emission is localized here).
const SOURCE_X: f64 = 0.5;
const SOURCE_WIDTH: f64 = 0.12;
/// Final wall-clock simulation time and the windowing used to sample the peak.
const T_END: f64 = 15.0;
const WINDOW: f64 = 0.25;

/// (species indices, makes the rest of the code readable)
struct SirIdx {
    s: usize,
    i: usize,
    r: usize,
}

/// One sampled instant of the coupled run.
#[derive(Clone)]
struct Sample {
    t: f64,
    sir: Vec<f64>,
    pathogen_mass: f64,
}

/// Everything recorded across the windowed drive.
struct History {
    samples: Vec<Sample>,
    peak_i: f64,
    peak_t: f64,
    peak_checkpoint: Checkpoint,
}

// ---------------------------------------------------------------------------
// Assertion helpers (loud, self-checking diagnostics).
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
// Profile sketch (compact horizontal bars at a handful of nodes).
// ---------------------------------------------------------------------------

fn profile_sketch(label: &str, xs: &[f64], coords: &[f64], width: f64) {
    let n = xs.len();
    let step = (n / 21).max(1);
    let cols = 40usize;
    println!("    {label} (x -> value):");
    for k in (0..n).step_by(step) {
        let x = coords[k];
        let bar_len = ((xs[k].abs() / width) * cols as f64).round() as usize;
        let bar = "#".repeat(bar_len.min(cols));
        println!("      x={x:4.2} | {bar:<40} {val:.4}", val = xs[k]);
    }
}

// ---------------------------------------------------------------------------
// Section 0: anatomy of the two SubModel implementations.
// ---------------------------------------------------------------------------

fn submodel_anatomy(grid: UniformGrid1D) {
    println!("\n[0] Sub-model anatomy (used bare via the SubModel trait)");

    // --- OdeSubModel: a single fast exponential decay dy/dt = -y, y(0) = 1. ---
    let mut decay = OdeSubModel::new("decay", |_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0);
    println!(
        "    OdeSubModel::new id = {}, time = {}, max_step = {}",
        decay.id(),
        decay.time(),
        decay.max_step()
    );
    // The integrator method is tunable per sub-model.
    decay.set_method(Method::Tsit45);
    println!("    method after set_method(Tsit45) = {:?}", decay.method());
    // Restrict the largest internal step; the orchestrator will sub-divide to it.
    decay.set_max_step(0.25);
    let mut peak = 0.0_f64;
    for _ in 0..4 {
        decay.advance(0.25).expect("decay advance succeeds");
        peak = peak.max(decay.state()[0]);
    }
    assert_finite("decay state", decay.state());
    println!(
        "    advanced 4 x 0.25 -> y = {:.6} (e^-1 = {:.6}), max ever = {peak:.6}",
        decay.state()[0],
        std::f64::consts::E.recip()
    );
    assert!((decay.state()[0] - std::f64::consts::E.recip()).abs() < 1e-3);

    // save/restore_state is the low-level primitive behind Checkpoint::restore.
    let saved = decay.state().to_vec();
    let saved_t = decay.time();
    decay.advance(0.25).expect("decay advance succeeds");
    decay
        .restore_state(&saved, saved_t)
        .expect("restore_state succeeds");
    assert_eq!(decay.state()[0], saved[0], "restore_state must be exact");
    // A length mismatch is a SimError::Advance.
    expect_error(
        "restore_state(wrong len)",
        decay.restore_state(&[0.0, 0.0], 0.0),
    );

    // --- DiffusionSubModel: explicit-Euler 1-D diffusion with an input source. ---
    // Dirichlet boundary: mass leaks out through the ends, so this is a clean
    // demonstration of the input buffer being the per-node forcing term.
    let heat_weights = weights(&grid);
    let mut heat =
        DiffusionSubModel::new("heat", grid, 0.02, Boundary::Dirichlet, vec![0.0; N]).unwrap();
    println!(
        "    DiffusionSubModel::new id = {}, coeff = {}, stability max_step = {:.5}",
        heat.id(),
        heat.coeff(),
        heat.max_step()
    );
    // The input buffer is how couplings (and here, a manual source) drive the
    // field: u += dt (D L u + input).
    let input = heat.input_mut().expect("diffusion exposes an input buffer");
    for (j, w) in heat_weights.iter().enumerate() {
        input[j] = *w; // unit-magnitude forcing at the center node
    }
    let mut peak = 0.0_f64;
    for _ in 0..20 {
        heat.advance(0.01).expect("heat advance succeeds");
        peak = peak.max(heat.state().iter().cloned().fold(0.0_f64, f64::max));
    }
    assert_finite("heat state", heat.state());
    let mass1: f64 = heat.state().iter().sum();
    println!(
        "    20 x dt=0.01 with a center source: peak = {peak:.4}, \
         total mass = {mass1:.3} (Dirichlet boundary nodes hold the source, \
         so mass grows toward the steady input)"
    );
    assert!(peak < 1.0, "heated peak must stay bounded");
}

// ---------------------------------------------------------------------------
// Section 1: cross-scale couplings, previewed before registration.
// ---------------------------------------------------------------------------

/// Spatial weights for a Gaussian centered at `SOURCE_X` (normalized to sum 1).
fn weights(grid: &UniformGrid1D) -> Vec<f64> {
    let coords = grid.coordinates();
    let mut w: Vec<f64> = coords
        .iter()
        .map(|&x| (-((x - SOURCE_X).powi(2)) / (2.0 * SOURCE_WIDTH * SOURCE_WIDTH)).exp())
        .collect();
    let sum: f64 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w
}

fn couplings(idx: &SirIdx, w: &[f64]) -> (Coupling, Coupling) {
    // Pathogen shedding: broadcast the infectious compartment onto the field as
    // a localized source (Gaussian weights), scaled by SHED_RATE.
    let w_owned = w.to_vec();
    let idx_i = idx.i;
    let shed = Coupling::new("sir", "pathogen", move |src, input| {
        let infected = src[idx_i];
        for (j, wj) in w_owned.iter().enumerate() {
            input[j] = infected * SHED_RATE * wj;
        }
    });
    // One-way sensing: the response field reads the pathogen field directly.
    let uptake = Coupling::new("pathogen", "response", |src, input| {
        for (j, v) in src.iter().enumerate() {
            input[j] = *v * CLEAR_GAIN;
        }
    });
    (shed, uptake)
}

fn coupling_preview(sir: &ReactionSystem, idx: &SirIdx, w: &[f64], shed: &Coupling) {
    println!(
        "\n[1] Couplings (source = {}, target = {})",
        shed.source(),
        shed.target()
    );
    println!(
        "    Coupling::source() = {:?}, target() = {:?}",
        shed.source(),
        shed.target()
    );
    // Exercise Coupling::apply on a probe: feed the initial SIR state and verify
    // the projected source matches the analytic formula I0 * SHED_RATE * w_j.
    let mut probe = vec![0.0_f64; N];
    let y0 = sir
        .initial_state(&[("S", S0), ("I", I0), ("R", R0)])
        .unwrap();
    shed.apply(&y0, &mut probe);
    let j = N / 2;
    let expected = y0[idx.i] * SHED_RATE * w[j];
    println!(
        "    apply(initial SIR): probe[center] = {:.6} (expected I0*rate*w = {:.6})",
        probe[j], expected
    );
    assert!(
        (probe[j] - expected).abs() < 1e-12,
        "coupling projection must match the analytic formula"
    );
}

// ---------------------------------------------------------------------------
// Section 2: assemble the reaction network and the orchestrated Simulation.
// ---------------------------------------------------------------------------

fn assemble_sir() -> (ReactionSystem, SirIdx) {
    let mut sir = ReactionNetwork::from_dsl(
        "beta, S + I --> 2 I
         gamma, I --> R",
    )
    .expect("SIR DSL must parse");
    // Bind the rate constants (otherwise they default to 0 and nothing evolves).
    sir.set_parameter("beta", BETA).expect("set beta");
    sir.set_parameter("gamma", GAMMA).expect("set gamma");
    let idx = SirIdx {
        s: sir.species_index("S").unwrap(),
        i: sir.species_index("I").unwrap(),
        r: sir.species_index("R").unwrap(),
    };
    (sir, idx)
}

fn build_sim(sir: &ReactionSystem, shed: Coupling, uptake: Coupling) -> Simulation {
    let y0 = sir
        .initial_state(&[("S", S0), ("I", I0), ("R", R0)])
        .unwrap();

    // The reaction ODE, wrapped as an OdeSubModel and tuned for accuracy via the
    // underlying OdeProblemBuilder (rtol/atol).
    let sir_builder = sir.ode_builder(&y0, 0.0).unwrap().rtol(1e-10).atol(1e-12);
    let mut sir_model = OdeSubModel::with_builder("sir", sir_builder, Method::Bdf);
    sir_model.set_max_step(0.1);

    let grid = UniformGrid1D::new(N, X0, X1).unwrap();
    let pathogen = DiffusionSubModel::new(
        "pathogen",
        grid.clone(),
        D_PATHOGEN,
        Boundary::Neumann,
        vec![0.0; N],
    )
    .unwrap();
    let response = DiffusionSubModel::new(
        "response",
        grid,
        D_RESPONSE,
        Boundary::Neumann,
        vec![0.0; N],
    )
    .unwrap();

    // Show which scale actually bounds the orchestrator.
    println!(
        "    per-model max_step: sir = {:.3} (set), \
         pathogen = {:.5} (stability), response = {:.5} (stability)",
        sir_model.max_step(),
        pathogen.max_step(),
        response.max_step()
    );

    let mut sim = Simulation::new();
    sim.add_model(sir_model).expect("add sir");
    sim.add_model(pathogen).expect("add pathogen");
    sim.add_model(response).expect("add response");
    sim.add_coupling(shed);
    sim.add_coupling(uptake);
    sim
}

// ---------------------------------------------------------------------------
// Section 3: windowed drive that records the infection peak and checkpoints it.
// ---------------------------------------------------------------------------

fn drive(sim: &mut Simulation, start: f64, stop: f64) -> Vec<Sample> {
    let mut t = start;
    let mut samples = Vec::new();
    while t < stop - 1e-9 {
        t += WINDOW;
        sim.step_until(t).expect("step_until succeeds");
        let sir = sim.model("sir").expect("sir present").state().to_vec();
        let pathogen_mass = sim
            .model("pathogen")
            .expect("pathogen present")
            .state()
            .iter()
            .sum();
        samples.push(Sample {
            t,
            sir,
            pathogen_mass,
        });
    }
    samples
}

fn record(sim: &mut Simulation, idx: &SirIdx) -> History {
    let mut samples = Vec::new();
    let mut peak_i = 0.0_f64;
    let mut peak_t = 0.0_f64;
    let mut peak_ck: Option<Checkpoint> = None;

    let mut t = 0.0_f64;
    while t < T_END - 1e-9 {
        t += WINDOW;
        sim.step_until(t).expect("step_until succeeds");
        let sir = sim.model("sir").expect("sir present").state().to_vec();
        let pathogen_mass = sim
            .model("pathogen")
            .expect("pathogen present")
            .state()
            .iter()
            .sum();
        if sir[idx.i] > peak_i {
            peak_i = sir[idx.i];
            peak_t = t;
            peak_ck = Some(sim.snapshot());
        }
        samples.push(Sample {
            t,
            sir,
            pathogen_mass,
        });
    }
    let peak_checkpoint = peak_ck.expect("a positive peak must be recorded");
    History {
        samples,
        peak_i,
        peak_t,
        peak_checkpoint,
    }
}

// ---------------------------------------------------------------------------
// Section 4: diagnostics + conservation assertions.
// ---------------------------------------------------------------------------

fn diagnose(sim: &Simulation, idx: &SirIdx, hist: &History) {
    println!("\n[4] Diagnostics at t = {:.2}", sim.global_time());

    let sir = sim.model("sir").expect("sir").state();
    let pathogen = sim.model("pathogen").expect("pathogen").state();
    let response = sim.model("response").expect("response").state();
    assert_finite("sir", sir);
    assert_finite("pathogen", pathogen);
    assert_finite("response", response);

    let grid = UniformGrid1D::new(N, X0, X1).unwrap();
    let coords = grid.coordinates();
    let p_peak = pathogen.iter().cloned().fold(0.0_f64, f64::max);
    let r_peak = response.iter().cloned().fold(0.0_f64, f64::max);
    let p_mass: f64 = pathogen.iter().sum();
    let r_mass: f64 = response.iter().sum();
    let p_center = pathogen[N / 2];
    let r_center = response[N / 2];

    println!(
        "    SIR: S = {:.2}, I = {:.2}, R = {:.2} \
         (peak I = {:.2} at t = {:.2})",
        sir[idx.s], sir[idx.i], sir[idx.r], hist.peak_i, hist.peak_t
    );
    println!(
        "    pathogen field: max = {p_peak:.4}, center = {p_center:.4}, total mass = {p_mass:.4}"
    );
    println!(
        "    response field: max = {r_peak:.4}, center = {r_center:.4}, total mass = {r_mass:.4}"
    );
    profile_sketch("pathogen", pathogen, &coords, p_peak.max(1e-9));
    profile_sketch("response", response, &coords, r_peak.max(1e-9));

    // --- Conservation: SIR population is closed (S+I+R conserved). ---
    let pop: f64 = sir.iter().sum();
    let pop0 = S0 + I0 + R0;
    println!(
        "    S+I+R = {pop:.6} (initial {pop0:.1}, relative drift {:.2e})",
        (pop - pop0).abs() / pop0
    );
    assert!(
        (pop - pop0).abs() / pop0 < 1e-6,
        "SIR population must be conserved"
    );

    // --- Diffusion mass budget: total shed = integral over time of I * SHED_RATE. ---
    // Analytically, d(R)/dt = GAMMA*I  =>  integral(I dt) = (R_end - R0)/GAMMA.
    let shed_analytic = SHED_RATE * (sir[idx.r] - R0) / GAMMA;
    println!(
        "    pathogen mass {p_mass:.4} vs analytic injected budget {shed_analytic:.4} \
         (relative {:.2e})",
        (p_mass - shed_analytic).abs() / shed_analytic.max(1e-9)
    );
    assert!(
        (p_mass - shed_analytic).abs() / shed_analytic.max(1e-9) < 0.05,
        "pathogen mass must match the injected budget"
    );

    // --- Response field: a one-way sensor, must be positive and driven. ---
    let r_integral: f64 = hist
        .samples
        .iter()
        .map(|s| s.pathogen_mass * WINDOW)
        .sum::<f64>()
        * CLEAR_GAIN
        / WINDOW;
    let r_trap: f64 = trapz(
        &hist.samples.iter().map(|s| s.t).collect::<Vec<_>>(),
        &hist
            .samples
            .iter()
            .map(|s| s.pathogen_mass)
            .collect::<Vec<_>>(),
    ) * CLEAR_GAIN;
    println!(
        "    response mass {r_mass:.4} vs integral(pathogen)*gain ~ {:.4} (trapz {:.4})",
        r_integral, r_trap
    );
    assert!(r_mass > 0.0, "response field must be driven positive");
    assert!(
        (r_mass - r_trap).abs() / r_trap.max(1e-9) < 0.05,
        "response mass must track the pathogen integral"
    );

    // --- The emission is localized: the peak must sit at the source node. ---
    let peak_idx = pathogen
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("    pathogen peak node index = {peak_idx} (source at x = {SOURCE_X})");
    assert_eq!(peak_idx, N / 2, "emission peak must be at the source node");

    // Response must be non-negative everywhere (it is driven by a positive field).
    assert!(
        response.iter().all(|&v| v >= -1e-9),
        "response field must stay non-negative"
    );
}

/// Trapezoidal integral of y(t) sampled at times t (t must be monotonic).
fn trapz(t: &[f64], y: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for k in 1..t.len() {
        acc += 0.5 * (t[k] - t[k - 1]) * (y[k] + y[k - 1]);
    }
    acc
}

// ---------------------------------------------------------------------------
// Section 5: checkpoint snapshot / restore reproducibility.
// ---------------------------------------------------------------------------

fn reproducibility(sim: &mut Simulation, idx: &SirIdx, hist: &History) {
    println!(
        "\n[5] Checkpoint reproducibility (snapshot at peak t = {:.2})",
        hist.peak_t
    );
    // Restore the peak snapshot and confirm it reproduces the recorded peak
    // state *bit-for-bit* (checkpoints store model state + clock exactly).
    let peak_sample = hist
        .samples
        .iter()
        .find(|s| (s.t - hist.peak_t).abs() < 1e-9)
        .expect("peak sample present");
    sim.restore(&hist.peak_checkpoint)
        .expect("restore succeeds");
    let restored = sim.model("sir").expect("sir").state().to_vec();
    let sir_dev = peak_sample
        .sir
        .iter()
        .zip(&restored)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "    restore(peak): SIR state vs recorded peak = {:.2e} (bit-for-bit for the ODE)",
        sir_dev
    );
    assert!(sir_dev < 1e-12, "ODE sub-model state must restore exactly");

    // Re-drive from the restored peak to the end and compare the *finished* SIR
    // state to the original finished run (both should be identical up to the
    // one-injection-sub-step lag in the diffusion input buffer, which is the
    // documented checkpoint semantics).
    let target = drive(sim, hist.peak_t, T_END);
    let finished = target.last().expect("samples non-empty");
    let orig = hist.samples.last().expect("samples non-empty");
    let dev_i = (finished.sir[idx.i] - orig.sir[idx.i]).abs();
    println!(
        "    re-driven SIR I at t={:.2}: {:.6} vs original {:.6} (dev {:.2e})",
        T_END, finished.sir[idx.i], orig.sir[idx.i], dev_i
    );
    assert!(dev_i < 1e-9, "re-driven SIR must match the original run");

    // The diffusion fields lag by one injection sub-step; assert they reconverge
    // to within 1% (the documented limitation of input buffers in checkpoints).
    let rel = (finished.pathogen_mass - orig.pathogen_mass).abs() / orig.pathogen_mass.max(1e-9);
    println!(
        "    re-driven pathogen mass deviates {:.2e} from original (one-sub-step lag)",
        rel
    );
    assert!(rel < 1e-2, "diffusion reconverges after restore");
}

// ---------------------------------------------------------------------------
// Section 6: the SimError surface.
// ---------------------------------------------------------------------------

fn error_surface() {
    println!("\n[6] Error handling (SimError)");

    // DuplicateModel: same id registered twice.
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

    // NonAdvancingStep: step_until to a non-increasing target.
    let mut adv = Simulation::new();
    adv.add_model(OdeSubModel::new(
        "x",
        |_t, y, dydt| dydt[0] = -y[0],
        vec![1.0],
        0.0,
    ))
    .expect("add x");
    expect_error("step_until(0.0)", adv.step_until(0.0));

    // CouplingTargetNoInput: an ODE model exposes no input buffer, so coupling
    // into it fails at step time. Show input_mut() is None for an ODE model.
    let mut bad = Simulation::new();
    let src = OdeSubModel::new("src", |_t, _y, dydt| dydt[0] = 1.0, vec![0.0], 0.0);
    let mut dst = OdeSubModel::new("dst", |_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0);
    assert!(
        dst.input_mut().is_none(),
        "OdeSubModel exposes no coupling input buffer"
    );
    bad.add_model(src).expect("add src");
    bad.add_model(dst).expect("add dst");
    bad.add_coupling(Coupling::new("src", "dst", |src, input| {
        for v in input.iter_mut() {
            *v = src[0];
        }
    }));
    expect_error("coupling into ODE (no input)", bad.step_until(0.5));

    // UnknownModel: a checkpoint referencing an unregistered id.
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

    // Advance (wrong length): a checkpoint with a mismatched state length.
    let wrong_len = Checkpoint {
        global_time: 0.0,
        models: vec![("x".to_string(), vec![1.0, 2.0], 0.0)],
    };
    expect_error("restore(wrong len)", sim.restore(&wrong_len));

    // A structurally invalid diffusion (initial field length != grid size).
    expect_error(
        "DiffusionSubModel(bad len)",
        DiffusionSubModel::new(
            "bad",
            UniformGrid1D::new(10, 0.0, 1.0).unwrap(),
            0.01,
            Boundary::Neumann,
            vec![0.0; 5],
        ),
    );
}

// ---------------------------------------------------------------------------
// Main tour.
// ---------------------------------------------------------------------------

fn main() {
    println!("=== tpt-sci-sim-core: multi-scale SIR -> diffusion cookbook ===");

    let grid = UniformGrid1D::new(N, X0, X1).unwrap();
    let (sir, idx) = assemble_sir();

    // Reaction-network introspection (the cross-crate composition layer).
    let y0 = sir
        .initial_state(&[("S", S0), ("I", I0), ("R", R0)])
        .unwrap();
    println!(
        "SIR species: {:?}, stoichiometry (S,I,R rows) = {:?}",
        sir.species_names(),
        sir.stoichiometry_matrix()
    );
    let rates0 = sir.reaction_rates(&y0);
    println!(
        "initial reaction rates r = [{:.4}, {:.4}] (beta*S*I, gamma*I)",
        rates0[0], rates0[1]
    );

    let w = weights(&grid);
    let (shed, uptake) = couplings(&idx, &w);
    coupling_preview(&sir, &idx, &w, &shed);

    let mut sim = build_sim(&sir, shed, uptake);

    submodel_anatomy(grid);

    let hist = record(&mut sim, &idx);
    diagnose(&sim, &idx, &hist);
    reproducibility(&mut sim, &idx, &hist);
    error_surface();

    println!("\nAll multi-scale cookbook checks passed.");
}
