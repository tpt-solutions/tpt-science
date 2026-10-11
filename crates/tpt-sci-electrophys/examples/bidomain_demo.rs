//! Demonstrate the **bidomain** capability of `tpt-sci-electrophys`: a 2-D tissue
//! sheet whose intracellular potential is coupled to an extracellular potential
//! `Ve`, solved from the elliptic equation
//! `(σ_i + σ_e)·∇²Ve = −σ_i·∇²Vm` (via [`tpt_sci_grid`]'s sparse conjugate
//! gradient) each time step.
//!
//! Run with: `cargo run --example bidomain_demo -p tpt-sci-electrophys`

use tpt_sci_electrophys::{Diffusivity, HodgkinHuxley, Tissue};

fn main() {
    // A modest 2-D sheet of Hodgkin–Huxley cells.
    let mut tissue =
        Tissue::with_model(24, 24, Diffusivity::Scalar(1.0), HodgkinHuxley::resting()).unwrap();
    // Couple the intracellular field to an extracellular domain with equal
    // conductivities (σ_i = σ_e = 1). The extracellular potential is then a
    // scaled, opposite copy of the transmembrane field.
    tissue.enable_bidomain(1.0);

    let dt = 0.005;
    for step in 0..400 {
        // Sustain a stimulus on the left edge for the first 30 steps to launch a
        // propagating activation wave.
        if step < 30 {
            for j in 0..24 {
                tissue.stimulate(0, j, 40.0);
            }
        }
        tissue.bidomain_step(dt).expect("bidomain step");

        if step % 80 == 0 || step == 399 {
            let max_vm = tissue.max_voltage();
            let max_ve = tissue.ve.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_ve = tissue.ve.iter().copied().fold(f64::INFINITY, f64::min);
            println!(
                "step {step:3}: max Vm = {max_vm:8.2} mV, Ve ∈ [{min_ve:8.2}, {max_ve:8.2}] mV"
            );
        }
    }

    // Sanity: every field must stay finite and bounded.
    assert!(tissue.vm.iter().all(|v| v.is_finite()));
    assert!(tissue.ve.iter().all(|v| v.is_finite()));
    println!("bidomain propagation completed without blow-up");
}
