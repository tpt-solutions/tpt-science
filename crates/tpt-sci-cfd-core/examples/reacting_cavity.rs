//! Coupled scenario: a reacting species in a lid-driven cavity.
//!
//! The flow solver (`Step`) spins up a recirculating cavity flow. A reactant is
//! injected at the bottom-left corner and transported by that flow. It is
//! consumed by a first-order reaction whose rate constant follows the Arrhenius
//! law from `tpt-sci-kinetics`.
//!
//! The coupling is one-way: the velocity field from the flow solver drives the
//! species transport, and the reaction source acts on the species. The checks
//! are physical. The lid must drive a recirculation, and the hotter reactor must
//! consume more reactant than the cold one, because the rate constant is larger.
//!
//! Run with: `cargo run --release --example reacting_cavity -p tpt-sci-cfd-core`

use tpt_sci_cfd_core::{Boundary, CollocatedGrid, ScalarTransport, Step};
use tpt_sci_kinetics::ArrheniusRate;

const N: usize = 32;
const NU: f64 = 0.01;
const FLOW_DT: f64 = 0.002;
const FLOW_STEPS: usize = 600; // t = 1.2
const INJECTION: f64 = 1.0; // amount per unit time, in the injection patch
const DIFFUSIVITY: f64 = 0.005;

/// Rate constant at temperature `t` for the reaction `A → B`.
fn rate_at(t: f64) -> f64 {
    // A = 1e3 s⁻¹, Eₐ = 20 kJ/mol.
    ArrheniusRate::new(1.0e3, 20_000.0)
        .expect("valid Arrhenius parameters")
        .rate_constant(t)
}

/// Run the flow once and return the final velocity field (the lid is moving, so
/// the flow is the same for both temperatures).
fn flow() -> (CollocatedGrid, Step) {
    let grid = CollocatedGrid::new(N, N, 1.0, 1.0).expect("valid grid");
    let mut step = Step::new(grid.clone(), NU, FLOW_DT, 1.0);
    step.set_boundary(Boundary::Top, 1.0);
    for _ in 0..FLOW_STEPS {
        assert!(step.advance(), "the flow solver must keep running");
    }
    (grid, step)
}

/// Transport and react the species for the given temperature, and return the
/// total amount remaining in the cavity.
fn react(grid: &CollocatedGrid, u: &[f64], v: &[f64], temperature: f64) -> f64 {
    let transport = ScalarTransport::new(grid.clone(), DIFFUSIVITY).expect("valid diffusivity");
    let k = rate_at(temperature);
    // Inject in the bottom-left corner, and consume everywhere at rate k.
    let injected = |cell: usize| -> f64 {
        let (i, j) = (cell % N, cell / N);
        if i < 4 && j < 4 { INJECTION } else { 0.0 }
    };
    let mut c = vec![0.0; N * N];
    // Sub-step the scalar inside each flow step so both stay within their stability limits.
    let dt_max = transport
        .max_stable_dt(u, v)
        .min(0.5 * k.recip().max(1e-12));
    let substeps = (FLOW_DT / dt_max).ceil().max(1.0) as usize;
    let dt = FLOW_DT / substeps as f64;
    for _ in 0..FLOW_STEPS * substeps {
        transport
            .step(&mut c, u, v, dt, &|cell, ci| injected(cell) - k * ci)
            .expect("a stable sub-step");
    }
    transport.total(&c)
}

fn main() {
    // Step 1 — the flow. The moving lid drives a recirculating cavity flow.
    let (grid, flow) = flow();
    let max_speed = flow
        .u
        .iter()
        .zip(&flow.v)
        .map(|(a, b)| (a * a + b * b).sqrt())
        .fold(0.0_f64, f64::max);
    println!(
        "step 1: lid-driven cavity, {N}x{N} cells, t = {:.1}",
        FLOW_STEPS as f64 * FLOW_DT
    );
    println!("        maximum speed in the cavity: {max_speed:.3}");
    assert!(max_speed > 0.2, "the lid must drive a recirculation");

    // Step 2 — the reactant at two temperatures, with the same flow.
    let (t_cold, t_hot) = (300.0, 600.0);
    let total_cold = react(&grid, &flow.u, &flow.v, t_cold);
    let total_hot = react(&grid, &flow.u, &flow.v, t_hot);
    let injected_total = INJECTION * 16.0 * FLOW_STEPS as f64 * FLOW_DT;
    println!("step 2: injected amount over the run: {injected_total:.2}");
    println!(
        "        remaining at {t_cold} K: {total_cold:.3} (k = {:.2} /s)",
        rate_at(t_cold)
    );
    println!(
        "        remaining at {t_hot} K: {total_hot:.3} (k = {:.2} /s)",
        rate_at(t_hot)
    );

    // Step 3 — the hot reactor consumes more reactant.
    assert!(
        total_hot < total_cold,
        "the faster reaction at {t_hot} K must leave less reactant"
    );
    println!("step 3: the hotter reactor leaves less reactant, as the Arrhenius law requires");
}
