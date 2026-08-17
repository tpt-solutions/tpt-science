//! Multi-scale cookbook: a reaction network (SIR) built with the
//! `tpt-sci-reaction-network` DSL drives a spatial diffusion field through
//! `tpt-sci-sim-core` orchestration — composing `tpt-sci-reaction-network` +
//! `tpt-sci-sim-core` + `tpt-sci-grid` end to end.
//!
//! The SIR ODE is wrapped directly as an `OdeSubModel`; its infected-compartment
//! state is coupled onto the input buffer of a 1-D `DiffusionSubModel` (the
//! canonical cross-scale pattern: a fast reaction feeding a slow spatial field).
//! The orchestrator sub-steps on the diffusion field's stability-limited time
//! scale while the reaction sub-model advances alongside it.
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
    let i_idx = sir.species_index("I").unwrap();

    // --- Multi-scale side: the SIR system drives a diffusion field. ---
    let grid = UniformGrid1D::new(41, 0.0, 1.0).unwrap();
    let diffusion = DiffusionSubModel::new(
        "field",
        grid,
        0.02,
        Boundary::Dirichlet,
        vec![0.0; 41],
    )
    .unwrap();

    // Wrap the reaction network's mass-action RHS as an ODE sub-model. tpt-sci-sim-core
    // integrates it on the SIR's own time scale and exposes its state to couplings.
    let sir_model = OdeSubModel::with_builder(
        "sir",
        sir.ode_builder(&y0, 0.0).unwrap(),
        Method::Bdf,
    );

    let coupling_strength = 0.01;
    let mut sim = Simulation::new();
    sim.add_model(diffusion).unwrap();
    sim.add_model(sir_model).unwrap();
    sim.add_coupling(Coupling::new("sir", "field", move |out, input| {
        // Broadcast the infected compartment onto every diffusion node as a source.
        let infected = out[i_idx];
        for v in input.iter_mut() {
            *v = infected * coupling_strength;
        }
    }));

    sim.step_until(20.0).unwrap();

    let sir_state = sim.model("sir").unwrap().state();
    let infected = sir_state[i_idx];
    let field = sim.model("field").unwrap().state();
    let peak = field.iter().cloned().fold(0.0_f64, f64::max);
    let l2: f64 = field.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!(
        "Coupled multi-scale run at t=20: I = {infected:.2} infected, \
         diffusion field peak = {peak:.4}, L2 norm = {l2:.4}"
    );
}
