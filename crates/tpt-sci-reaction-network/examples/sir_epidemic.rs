//! Full SIR epidemic run from the Catalyst.jl-style species/rate DSL.
//!
//! Run with: `cargo run --example sir_epidemic -p tpt-sci-reaction-network`

use tpt_sci_ode::Method;
use tpt_sci_reaction_network::ReactionNetwork;

fn main() {
    // S + I -> 2 I  (transmission),  I -> R  (recovery)
    let mut model = ReactionNetwork::from_dsl(
        "beta, S + I --> 2 I
         gamma, I --> R",
    )
    .unwrap();
    model.set_parameter("beta", 0.0021).unwrap();
    model.set_parameter("gamma", 0.45).unwrap();

    let y0 = model
        .initial_state(&[("S", 999.0), ("I", 1.0), ("R", 0.0)])
        .unwrap();
    let prob = model.to_ode_problem(&y0, 0.0).unwrap();

    // Integrate 100 days.
    let y = prob.solve(Method::Bdf, 100.0).unwrap();

    let s = model.species_index("S").unwrap();
    let i = model.species_index("I").unwrap();
    let r = model.species_index("R").unwrap();

    println!("SIR after 100 days (population N = 1000):");
    println!("  Susceptible  S = {:.1}", y[s]);
    println!("  Infected     I = {:.1}", y[i]);
    println!("  Recovered    R = {:.1}", y[r]);
    println!("  conserved S+I+R = {:.1}", y[s] + y[i] + y[r]);

    assert!((y[s] + y[i] + y[r] - 1000.0).abs() < 1e-3);
}
