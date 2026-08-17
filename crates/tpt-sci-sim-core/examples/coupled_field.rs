//! Multi-scale orchestration: a fast ODE source drives a slow 1-D diffusion
//! field through a cross-scale coupling, plus a checkpoint/restore round-trip.
//!
//! Run with: `cargo run --example coupled_field -p tpt-sci-sim-core`

use tpt_sci_grid::{Boundary, UniformGrid1D};
use tpt_sci_sim_core::{Coupling, DiffusionSubModel, OdeSubModel, Simulation};

fn main() {
    // Source ODE: y' = 1 (a ramp) feeding a diffusion field via coupling.
    let mut source = OdeSubModel::new("src", |_t, _y, dydt| dydt[0] = 1.0, vec![0.0], 0.0);
    source.set_max_step(0.1);
    let grid = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
    let field = DiffusionSubModel::new("field", grid, 1e-3, Boundary::Neumann, vec![0.0; 21]).unwrap();

    let mut sim = Simulation::new();
    sim.add_model(source).unwrap();
    sim.add_model(field).unwrap();
    // Broadcast the source scalar onto every diffusion node.
    sim.add_coupling(Coupling::new("src", "field", |src, input| {
        for x in input.iter_mut() {
            *x = src[0];
        }
    }));

    sim.step_until(0.5).unwrap();
    let field_state = sim.model("field").unwrap().state();
    let total: f64 = field_state.iter().sum();
    println!(
        "field total after coupling = {total:.4} (source y = {:.3})",
        sim.model("src").unwrap().state()[0]
    );

    // Checkpoint & restore demonstration.
    let checkpoint = sim.snapshot();
    let before = sim.model("field").unwrap().state()[10];
    sim.step_until(1.0).unwrap();
    let after = sim.model("field").unwrap().state()[10];
    sim.restore(&checkpoint).unwrap();
    let restored = sim.model("field").unwrap().state()[10];
    println!("field[10]: before_advance = {before:.4}, after_advance = {after:.4}, restored = {restored:.4}");
    assert!((before - restored).abs() < 1e-12);
}
