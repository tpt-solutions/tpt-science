//! Multi-scale cookbook: a reaction network (SIR) feeding a spatial diffusion
//! field through `tpt-sci-sim-core` orchestration, composing
//! `tpt-sci-reaction-network` + `tpt-sci-sim-core` + `tpt-sci-grid`.
//!
//! The SIR model is built with the reaction-network DSL and integrated directly;
//! the multi-scale run then drives a 1-D `DiffusionSubModel` field from an ODE
//! sub-model (standing in for the reaction output), advanced on its own finer
//! time scale by the `Simulation`.
use tpt_sci_grid::{Boundary, UniformGrid1D};
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::ReactionNetwork;
use tpt_sci_sim_core::{Coupling, DiffusionSubModel, OdeSubModel, Simulation};

fn main() {
    // --- Reaction side: SIR built with the reaction-network DSL. ---
    let mut sir = ReactionNetwork::from_dsl(
        "beta, S + I --> 2 I
         gamma, I --> R",
    )
    .unwrap();
    sir.set_parameter("beta", 0.002).unwrap();
    sir.set_parameter("gamma", 0.4).unwrap();
    let y0 = sir.initial_state(&[("S", 990.0), ("I", 10.0), ("R", 0.0)]).unwrap();
    let prob = sir.to_ode_problem(&y0, 0.0).unwrap();
    let y = prob.solve(Method::Bdf, 150.0).unwrap();
    let i_idx = sir.species_index("I").unwrap();
    println!("Reaction-only SIR at t=150: I = {:.2} infected", y[i_idx]);

    // --- Multi-scale side: a source ODE drives a diffusion field. ---
    let grid = UniformGrid1D::new(51, 0.0, 1.0).unwrap();
    let diffusion = DiffusionSubModel::new(
        "field",
        grid,
        0.02,
        Boundary::Dirichlet,
        vec![0.0; 51],
    )
    .unwrap();

    let source = OdeSubModel::new(
        "source",
        |_t, y, dydt| dydt[0] = -0.5 * y[0],
        vec![1.0],
        0.0,
    );

    let mut sim = Simulation::new();
    sim.add_model(diffusion).unwrap();
    sim.add_model(source).unwrap();
    sim.add_coupling(Coupling::new("source", "field", |out, input| {
        for v in input.iter_mut() {
            *v = out[0];
        }
    }));

    sim.step_until(5.0).unwrap();
    let field = sim.model("field").unwrap().state();
    let peak = field.iter().cloned().fold(0.0_f64, f64::max);
    let l2: f64 = field.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!(
        "Diffusion field after coupling: peak = {peak:.4}, L2 norm = {l2:.4}"
    );
}
