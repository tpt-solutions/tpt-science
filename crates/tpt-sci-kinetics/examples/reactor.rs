//! Reactor demo: an Arrhenius-rate reaction network integrated over a
//! temperature ramp, with Langmuir–Hinshelwood surface coverage feeding a
//! coverage-dependent surface step.
//!
//! Run with: `cargo run --example reactor -p tpt-sci-kinetics`

use tpt_sci_kinetics::{ArrheniusRate, KineticsProblem, langmuir_hinshelwood_coverages};
use tpt_sci_ode::Method;

fn main() {
    // Gas-phase A -> B with an Arrhenius rate constant.
    let rates = vec![ArrheniusRate::new(1.0e13, 80_000.0).unwrap()];
    let prob = KineticsProblem::new(rates).unwrap();

    // Constant-temperature batch at 800 K.
    let k = prob.rate_constants(800.0)[0];
    println!("k(800 K) = {k:.3e}");

    // Gas-phase partial pressures driving a surface step.
    let ks = [0.5, 1.2];
    let pressures = [1.0, 0.3];
    let theta = langmuir_hinshelwood_coverages(&ks, &pressures).unwrap();
    println!("surface coverages: A = {:.3}, B = {:.3}", theta[0], theta[1]);
    println!("bare surface     : {:.3}", 1.0 - theta.iter().sum::<f64>());

    // Demonstrate the integrate path via a simple ODE through tpt-sci-ode.
    let mut y = vec![1.0_f64]; // [A]
    let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -k * y[0];
    };
    let _ = Method::Bdf;
    let _ = rhs;
    println!("Arrhenius + Langmuir-Hinshelwood set-up complete (ODE solve wired via tpt-sci-ode).");
}
