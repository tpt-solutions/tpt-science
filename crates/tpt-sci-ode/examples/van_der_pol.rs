//! Van der Pol oscillator integrated with `tpt-sci-ode`, showing both a single
//! point solve and a dense trajectory evaluation.
use tpt_sci_ode::{Method, OdeProblem};

fn main() {
    // dy0/dt = y1,  dy1/dt = mu*(1 - y0^2)*y1 - y0   (mu = 1)
    let prob = OdeProblem::new(
        |_t, y, dydt| {
            let mu = 1.0;
            dydt[0] = y[1];
            dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        },
        vec![2.0, 0.0],
        0.0,
    )
    .unwrap();

    let y = prob.solve(Method::Tsit45, 20.0).unwrap();
    println!("Van der Pol at t=20: y0 = {:.4}, y1 = {:.4}", y[0], y[1]);

    let times = [1.0, 5.0, 10.0, 15.0, 20.0];
    let dense = prob.solve_dense(Method::Tsit45, &times).unwrap();
    println!("Amplitude along the trajectory:");
    for (t, s) in times.into_iter().zip(&dense) {
        println!("  t = {t:5.1}  y0 = {:7.4}", s[0]);
    }
}
