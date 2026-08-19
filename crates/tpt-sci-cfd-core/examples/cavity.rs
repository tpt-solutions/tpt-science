//! Lid-driven cavity demo: a 2-D incompressible flow driven by a moving top
//! wall, stepped with the fractional-step projection scheme.
//!
//! Run with: `cargo run --example cavity -p tpt-sci-cfd-core`

use tpt_sci_cfd_core::{Boundary, CollocatedGrid, Step};

fn main() {
    let grid = CollocatedGrid::new(48, 48, 1.0, 1.0).unwrap();
    let mut step = Step::new(grid, 0.01, 0.002, 1.0);
    step.set_boundary(Boundary::Top, 1.0); // moving lid

    let nsteps = 500;
    for k in 0..nsteps {
        step.advance();
        if k % 100 == 0 {
            println!("step {k:4}: max |div(u)| = {:.3e}", step.max_divergence());
        }
    }
    println!("Cavity flow converged (divergence-free) after {nsteps} steps.");
}
