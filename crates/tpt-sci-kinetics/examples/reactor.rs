//! Reactor demo: an Arrhenius-rate gas-phase decay `A -> B` integrated over a
//! temperature ramp, with a Langmuir–Hinshelwood surface coverage feeding a
//! coverage-dependent surface step.
//!
//! Run with: `cargo run --example reactor -p tpt-sci-kinetics`

use tpt_sci_kinetics::{ArrheniusRate, KineticsProblem, langmuir_hinshelwood_coverages};
use tpt_sci_ode::{Method, OdeProblem};

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
    println!(
        "surface coverages: A = {:.3}, B = {:.3}",
        theta[0], theta[1]
    );
    println!("bare surface     : {:.3}", 1.0 - theta.iter().sum::<f64>());

    // Integrate the batch decay A -> B at 800 K through tpt-sci-ode.
    // k is large (~6e7 1/s), so the decay timescale is ~1/k; integrate over a
    // short window that spans a few e-foldings.
    let t_end = 5.0e-8;
    let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -k * y[0];
    };
    let oprob = OdeProblem::new(rhs, vec![1.0_f64], 0.0).unwrap();
    let y = oprob.solve(Method::Tsit45, t_end).unwrap();
    println!(
        "A remaining after {:.1e} s = {:.4} (vs e^-k·t = {:.4})",
        t_end,
        y[0],
        (-k * t_end).exp()
    );
}
