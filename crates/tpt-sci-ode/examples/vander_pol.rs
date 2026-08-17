//! Van der Pol oscillator — a classic limit-cycle ODE solved with `tpt-sci-ode`.
//!
//! Run with: `cargo run --example vander_pol -p tpt-sci-ode`

use tpt_sci_ode::{Method, OdeProblem};

fn main() {
    // Van der Pol: x'' - mu*(1 - x^2)*x' + x = 0, written as a first-order system.
    let mu = 1.0;
    let prob = OdeProblem::new(
        move |_t, y, dydt| {
            let (x, v) = (y[0], y[1]);
            dydt[0] = v;
            dydt[1] = mu * (1.0 - x * x) * v - x;
        },
        vec![2.0, 0.0],
        0.0,
    )
    .unwrap();

    let y = prob.solve(Method::Bdf, 12.0).unwrap();
    println!("Van der Pol at t = 12: x = {:.4}, v = {:.4}", y[0], y[1]);

    // The oscillator settles onto a stable limit cycle of amplitude ~2.
    assert!(y[0].abs() < 3.0);
    assert!(y[1].abs() < 3.0);
}
