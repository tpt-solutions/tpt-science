//! Bell-state and GHZ-state walkthrough with measurement statistics.
//!
//! Run with: `cargo run --example bell_ghz -p tpt-sci-quantum`

use tpt_math_prob_core::SplitMix64;
use tpt_sci_quantum::State;

fn main() {
    // --- Bell state |Phi+> = (|00> + |11>)/sqrt(2) ---
    let mut bell = State::new(2).unwrap();
    bell.h(0).unwrap();
    bell.cnot(0, 1).unwrap();

    let p = bell.probabilities();
    println!("Bell |Φ+> outcome probabilities:");
    println!("  |00> = {:.3}", p[0]);
    println!("  |11> = {:.3}", p[3]);
    assert!((p[0] - 0.5).abs() < 1e-9);
    assert!((p[3] - 0.5).abs() < 1e-9);

    // Sample the Bell state many times and confirm the 50/50 split.
    let mut rng = SplitMix64::seed_from_u64(42);
    let trials: usize = 20_000;
    let mut count_00 = 0usize;
    let mut count_11 = 0usize;
    for _ in 0..trials {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        match s.measure(&mut rng) {
            0 => count_00 += 1,
            3 => count_11 += 1,
            other => panic!("unexpected Bell outcome {other}"),
        }
    }
    let f00 = count_00 as f64 / trials as f64;
    let f11 = count_11 as f64 / trials as f64;
    println!("  sampled |00> = {f00:.3}, |11> = {f11:.3} over {trials} shots");
    assert!((f00 - 0.5).abs() < 0.05);
    assert!((f11 - 0.5).abs() < 0.05);

    // --- GHZ state |GHZ> = (|000> + |111>)/sqrt(2) ---
    let mut ghz = State::new(3).unwrap();
    ghz.h(0).unwrap();
    ghz.cnot(0, 1).unwrap();
    ghz.cnot(1, 2).unwrap();
    let pg = ghz.probabilities();
    println!("\nGHZ outcome probabilities:");
    println!("  |000> = {:.3}", pg[0]);
    println!("  |111> = {:.3}", pg[7]);
    assert!((pg[0] - 0.5).abs() < 1e-9);
    assert!((pg[7] - 0.5).abs() < 1e-9);

    // All-parity check: <Z⊗Z⊗Z> = 0 for a GHZ state.
    println!("  <Z⊗Z⊗Z> = {:.3} (expect 0)", ghz.expectation_z());
    assert!(ghz.expectation_z().abs() < 1e-9);

    println!("\nAll checks passed.");
}
