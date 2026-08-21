//! # grover_oracle.rs — Grover search & phase oracles in `tpt-sci-quantum`
//!
//! This example explores a genuinely different quantum primitive from the
//! Bell/GHZ tour: **Grover's search algorithm** and the **phase oracle** it
//! relies on. Whereas `bell_ghz.rs` demonstrates entanglement and measurement
//! statistics, this example shows *amplitude amplification* — marking one
//! computational-basis state and rotating the uniform superposition so that the
//! marked state dominates the Born distribution.
//!
//! ## Background
//!
//! * A **phase oracle** `U_w` flips the sign (applies `−1`) of the amplitude of
//!   a single marked basis state `|w⟩` and leaves every other state untouched.
//!   For an arbitrary `|w⟩` we synthesize it from `X` (bit-flips) and a
//!   controlled-`Z` (CZ): flip the qubits that are `0` in `|w⟩`, apply CZ, then
//!   unflip. This is the standard "marking" circuit.
//! * **Grover's diffusion operator** inverts the state about the mean:
//!   `D = H^{⊗n} · X^{⊗n} · CZ · X^{⊗n} · H^{⊗n}`.
//! * For `N = 2^n` items with one marked, the optimal number of Grover
//!   iterations is `≈ π/4·√N`; a single iteration on `n = 2` qubits (one marked
//!   out of four) amplifies the marked state to probability `1`.
//!
//! We verify two things with real `State`/`Circuit`/`Gate` calls: the oracle
//! negates *only* the marked amplitude, and the full Grover iterate collapses
//! the Born distribution onto the marked state (checked analytically and with
//! multi-shot sampling).
//!
//! Run with: `cargo run --example grover_oracle -p tpt-sci-quantum`

use tpt_math_prob_core::SplitMix64;
use tpt_sci_quantum::{Circuit, State, Z};

fn main() {
    println!("=== tpt-sci-quantum: Grover search & phase oracle ===\n");

    // ---------------------------------------------------------------------
    // 1. Phase oracle marking |01> (index 1 in little-endian qubit order):
    //    q0 = 1, q1 = 0. Synthesize it from X and a controlled-Z.
    //
    //    Only qubit q1 is 0 in the marked state, so we flip q1, apply CZ(0,1)
    //    (which flips the sign of |11>), and unflip q1. The net effect is to
    //    negate exactly the amplitude of |01>.
    // ---------------------------------------------------------------------
    let marked = 1usize; // |01> = (q1=0, q0=1)
    let mut oracle = Circuit::new(2);
    oracle.x(1).controlled(0, 1, Z).x(1);
    let u_oracle = oracle.unitary();

    // Apply the oracle to a uniform superposition H⊗H|00>.
    let mut sup = State::new(2).unwrap();
    sup.h(0).unwrap();
    sup.h(1).unwrap();
    let tagged = sup.apply_unitary(&u_oracle).unwrap();
    // A uniform 2-qubit superposition has amplitude 1/2 in every basis state.
    let amp = 0.5_f64;
    println!("Phase oracle on H⊗H|00> (all amplitudes start at +1/2):");
    for i in 0..4 {
        let re = tagged.amplitude(i).re;
        let sign = if i == marked { -1.0 } else { 1.0 };
        println!(
            "  |{:02b}> amplitude = {:+.4} (expect {:+.4})",
            i, re, sign * amp
        );
        assert!((re - sign * amp).abs() < 1e-9, "oracle sign on |{i:02b}>");
    }
    // Exactly one amplitude is negated.
    assert!((tagged.amplitude(marked).re + amp).abs() < 1e-9, "marked state flipped");
    println!("  -> only the marked state |{marked:02b}> was phase-flipped\n");

    // ---------------------------------------------------------------------
    // 2. Full Grover iteration for the same marked state on 2 qubits.
    //    H⊗H |00>  ->  oracle  ->  diffusion  ->  ~ |01>.
    // ---------------------------------------------------------------------
    let mut g = State::new(2).unwrap();
    g.h(0).unwrap();
    g.h(1).unwrap();

    // Oracle U_w: X(1); CZ(0,1); X(1).
    g.x(1).unwrap();
    g.apply_controlled(0, 1, &Z).unwrap();
    g.x(1).unwrap();

    // Diffusion operator D = H⊗H · X⊗X · CZ(0,1) · X⊗X · H⊗H.
    g.h(0).unwrap();
    g.h(1).unwrap();
    g.x(0).unwrap();
    g.x(1).unwrap();
    g.apply_controlled(0, 1, &Z).unwrap();
    g.x(0).unwrap();
    g.x(1).unwrap();
    g.h(0).unwrap();
    g.h(1).unwrap();

    let p = g.probabilities();
    println!("After one Grover iteration (n=2, one marked):");
    for (i, &pi) in p.iter().enumerate() {
        println!("  P(|{:02b}>) = {:.4}", i, pi);
    }
    println!("  norm = {:.6} (expect 1)", g.norm());
    assert!((g.norm() - 1.0).abs() < 1e-12, "post-Grover state normalized");
    // One iteration on 2 qubits sends the marked state to certainty.
    assert!(
        (p[marked] - 1.0).abs() < 1e-6,
        "marked state should be found with probability 1, got {}",
        p[marked]
    );
    for (i, &pi) in p.iter().enumerate() {
        if i != marked {
            assert!(pi < 1e-6, "non-marked state |{i:02b}> should be empty");
        }
    }
    println!(
        "  -> P(|{marked:02b}>) = {:.6}, measured distribution collapsed onto the marked state\n",
        p[marked]
    );

    // ---------------------------------------------------------------------
    // 3. Multi-shot sampling confirms the algorithm *identifies* the marked
    //    state: drawing from the (non-destructive) Born distribution returns
    //    |01> essentially every time.
    // ---------------------------------------------------------------------
    let mut rng = SplitMix64::seed_from_u64(0x9E3779B9);
    let trials = 10_000usize;
    let mut hits = 0usize;
    for _ in 0..trials {
        if g.measure(&mut rng) == marked {
            hits += 1;
        }
    }
    let frac = hits as f64 / trials as f64;
    println!(
        "Grover sampling over {trials} shots: P(|{marked:02b}>) = {frac:.4} (analytic = 1.000)"
    );
    assert!(frac > 0.9, "Grover failed to identify the marked state: frac={frac}");

    println!("\nAll Grover/oracle checks passed.");
}
