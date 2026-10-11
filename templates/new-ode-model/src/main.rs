//! An ODE model: exponential decay `y' = -k·y`, solved and checked against the
//! closed form. Replace the right-hand side with your own model.

use tpt_sci::ode::{Method, OdeProblemBuilder};

const K: f64 = 0.5; // decay rate
const T_END: f64 = 10.0;

fn main() {
    let problem = OdeProblemBuilder::new(
        |_t, y: &[f64], dy: &mut [f64]| dy[0] = -K * y[0],
        vec![1.0],
        0.0,
    )
    .rtol(1e-8)
    .atol(1e-10)
    .build()
    .expect("valid problem");

    let times: Vec<f64> = (1..=10).map(|i| i as f64 * T_END / 10.0).collect();
    let traj = problem
        .solve_trajectory(Method::Bdf, &times, &["y"])
        .expect("integration succeeds");

    // Write CSV to stdout: `t,y` header and one row per output time.
    print!("{}", traj.to_csv());

    let worst = traj
        .t
        .iter()
        .zip(&traj.y)
        .map(|(t, y)| (y[0] - (-K * t).exp()).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("max error against the closed form: {worst:.2e}");
    assert!(worst < 1e-6, "solution drifted from the closed form");
}
