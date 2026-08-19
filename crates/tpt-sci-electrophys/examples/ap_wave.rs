//! Electrophysiology demo: an action potential is triggered at one edge of a
//! 2-D tissue sheet and propagates via the monodomain diffusion operator.
//!
//! Run with: `cargo run --example propagation -p tpt-sci-electrophys`

use tpt_sci_electrophys::Tissue;

fn main() {
    let mut t = Tissue::new(32, 32, 0.8).unwrap();
    // Stimulate a vertical strip on the left edge.
    for j in 0..32 {
        t.stimulate(0, j, 40.0);
    }

    let n = 400;
    for k in 0..n {
        t.step(0.005);
        if k % 100 == 0 {
            println!("step {k}: max Vm = {:.2} mV", t.max_voltage());
        }
    }
    println!("Action potential propagated across the sheet.");
}
