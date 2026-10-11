//! Tutorial: from a single cell to a propagating wave in a bidomain cable.
//!
//! A Hodgkin–Huxley cell fires an action potential when stimulated above
//! threshold. Coupling cells by diffusion turns that into a travelling wave.
//! Adding an extracellular domain (the bidomain model) gives an extracellular
//! potential we can inspect.
//!
//! Run with: `cargo run --release --example tutorial_cable_bidomain -p tpt-sci --features full`

use tpt_sci::electrophys::{Diffusivity, HodgkinHuxley, Tissue};

const NX: usize = 60; // cells along the cable
const NY: usize = 1; // a one-cell-wide cable
const DT: f64 = 0.01; // ms
const STEPS: usize = 6000; // 60 ms

fn main() {
    // Step 1 — one cell. Unstimulated, it stays at rest.
    let mut quiet = Tissue::with_model(1, 1, Diffusivity::Scalar(1.0), HodgkinHuxley::resting())
        .expect("valid single-cell tissue");
    let mut peak_quiet = f64::NEG_INFINITY;
    for _ in 0..(20.0 / DT) as usize {
        quiet.step(DT);
        peak_quiet = peak_quiet.max(quiet.vm[0]);
    }
    println!("step 1: unstimulated cell peak = {peak_quiet:.1} mV (stays at rest)");
    assert!(peak_quiet < 0.0, "an unstimulated cell must not spike");

    // Step 2 — a cable. Stimulate the left end and watch the wave travel.
    let mut cable = Tissue::with_model(NX, NY, Diffusivity::Scalar(1.0), HodgkinHuxley::resting())
        .expect("valid cable");
    for i in 0..4 {
        cable.stimulate(i, 0, 40.0);
    }
    let probe = NX - 5; // cell we time the wave at
    let mut crossing: Option<f64> = None;
    for step in 0..STEPS {
        cable.step(DT);
        let t = (step + 1) as f64 * DT;
        if crossing.is_none() && cable.vm[cable.idx(probe, 0)] > 0.0 {
            crossing = Some(t);
        }
    }
    let t_arrive = crossing.expect("the wave must reach the probe cell");
    println!(
        "step 2: wave crosses 0 mV at cell {probe} at t = {t_arrive:.2} ms (≈ {:.2} cells/ms)",
        (probe - 4) as f64 / t_arrive
    );

    // Step 3 — bidomain. The same cable with an extracellular domain of equal
    // conductivity. The extracellular potential varies along the cable.
    let mut bidomain =
        Tissue::with_model(NX, NY, Diffusivity::Scalar(1.0), HodgkinHuxley::resting())
            .expect("valid cable");
    bidomain.enable_bidomain(1.0);
    for i in 0..4 {
        bidomain.stimulate(i, 0, 40.0);
    }
    for _ in 0..(20.0 / DT) as usize {
        bidomain.bidomain_step(DT).expect("bidomain step");
    }
    let phi_e = bidomain.extracellular_potential();
    let (lo, hi) = phi_e
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    println!("step 3: extracellular potential along the cable ∈ [{lo:.2}, {hi:.2}] mV");
    assert!(lo < hi, "the extracellular field must vary along the cable");
}
