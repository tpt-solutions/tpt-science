//! Multi-scale orchestration: two ODE sub-models stepping at their own (different)
//! maximum) rates, driven together to a shared target time by `tpt-sci-sim-core`.
use tpt_sci_sim_core::{OdeSubModel, Simulation};

fn main() {
    // Fast: dx/dt = -x  (a quickly-decaying reagent).
    let fast = OdeSubModel::new(
        "fast",
        |_t, y, dydt| dydt[0] = -y[0],
        vec![1.0],
        0.0,
    );
    // Slow: dy/dt = 0.1 * (1 - y)  (a slowly saturating quantity).
    let slow = OdeSubModel::new(
        "slow",
        |_t, y, dydt| dydt[0] = 0.1 * (1.0 - y[0]),
        vec![0.0],
        0.0,
    );

    let mut sim = Simulation::new();
    sim.add_model(fast).unwrap();
    sim.add_model(slow).unwrap();

    // Each model advances on its own internal time scale; the simulation takes the
    // largest sub-step that keeps every model from overshooting the target.
    let t_end = 3.0;
    sim.step_until(t_end).unwrap();

    let fast_val = sim.model("fast").unwrap().state()[0];
    let slow_val = sim.model("slow").unwrap().state()[0];
    println!(
        "At t={t_end}: fast = {fast_val:.4} (e^-3 = {:.4}), slow = {slow_val:.4} (1 - e^-0.3 = {:.4})",
        std::f64::consts::E.powi(-3),
        1.0 - std::f64::consts::E.powf(-0.3)
    );
}
