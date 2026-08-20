//! # Bell & GHZ state tour of `tpt-sci-quantum`
//!
//! This example is a guided walk through the public surface of the
//! `tpt-sci-quantum` crate: state construction, the in-place gate API, the
//! tensor/Kronecker `Circuit` API, non-destructive vs collapsing measurement,
//! multi-shot statistics, explicit unitary application, and error handling.
//!
//! ## Background
//!
//! * **Bell (EPR) state** `|Φ+⟩ = (|00⟩ + |11⟩)/√2`: two maximally-entangled
//!   qubits. Measuring both yields `00` or `11` with equal probability and
//!   perfect correlation.
//! * **GHZ state** `|GHZ⟩ = (|000⟩ + |111⟩)/√2`: three maximally-entangled
//!   qubits. Measuring all three yields all-zeros or all-ones with equal
//!   probability.
//!
//! Both are built from a Hadamard followed by a chain of CNOTs. Run with:
//! `cargo run --example bell_ghz -p tpt-sci-quantum`

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_math_prob_core::SplitMix64;
use tpt_sci_quantum::tensor::{embed_gate_2x2, kron};
use tpt_sci_quantum::{Circuit, H, State, StateError, X};

fn main() {
    println!("=== tpt-sci-quantum API tour: Bell & GHZ states ===\n");
    let inv = std::f64::consts::FRAC_1_SQRT_2;

    // 1) Bell state via the in-place `State` gate API.
    let mut bell = State::new(2).unwrap();
    bell.h(0).unwrap();
    bell.cnot(0, 1).unwrap();
    let p = bell.probabilities();
    println!("Bell |Φ+> = (|00> + |11>)/√2");
    println!("  P(|00>) = {:.3}, P(|11>) = {:.3}", p[0], p[3]);
    assert!((p[0] - 0.5).abs() < 1e-9);
    assert!((p[3] - 0.5).abs() < 1e-9);
    // All-qubit parity ⟨Z⊗Z⟩ = +1 for Bell (both outcomes have even parity).
    println!("  <Z⊗Z> = {:.3} (expect +1)", bell.expectation_z());
    assert!((bell.expectation_z() - 1.0).abs() < 1e-9);
    println!("  norm = {:.6} (expect 1)", bell.norm());
    assert!((bell.norm() - 1.0).abs() < 1e-12);

    // 2) The same Bell state via the tensor/Kronecker `Circuit` API.
    let mut circ = Circuit::new(2);
    circ.h(0).cnot(0, 1);
    let u = circ.unitary();
    let bell2 = State::zero(2).unwrap().apply_unitary(&u).unwrap();
    for i in 0..4 {
        assert!((bell2.amplitude(i) - bell.amplitude(i)).norm() < 1e-9);
    }
    println!(
        "\nCircuit::unitary + State::apply_unitary reproduce |Φ+> (verified on {} qubits)\n",
        circ.n_qubits()
    );

    // 3) GHZ state via the in-place API (3 qubits).
    let mut ghz = State::new(3).unwrap();
    ghz.h(0).unwrap();
    ghz.cnot(0, 1).unwrap();
    ghz.cnot(1, 2).unwrap();
    let pg = ghz.probabilities();
    println!("GHZ = (|000> + |111>)/√2");
    println!("  P(|000>) = {:.3}, P(|111>) = {:.3}", pg[0], pg[7]);
    assert!((pg[0] - 0.5).abs() < 1e-9);
    assert!((pg[7] - 0.5).abs() < 1e-9);
    // GHZ parity ⟨Z⊗Z⊗Z⟩ = 0 (even and odd outcomes cancel).
    println!("  <Z⊗Z⊗Z> = {:.3} (expect 0)", ghz.expectation_z());
    assert!(ghz.expectation_z().abs() < 1e-9);

    // 4) GHZ via `Circuit` with a NON-adjacent CNOT(0,2), exercising the
    //    SWAP decomposition inside `Circuit::controlled`.
    let mut c3 = Circuit::new(3);
    c3.h(0).cnot(0, 1).cnot(0, 2);
    let ghz2 = State::new(3).unwrap().apply_unitary(&c3.unitary()).unwrap();
    for i in 0..8 {
        assert!((ghz2.amplitude(i) - ghz.amplitude(i)).norm() < 1e-9);
    }
    println!("Circuit (incl. non-adjacent CNOT) reproduces GHZ (verified)\n");

    // 5) Explicit Kronecker construction H⊗I on two qubits.
    let h2 = embed_gate_2x2(&H);
    let i2 = DMatrix::from_fn(2, 2, |a, b| if a == b { 1.0 } else { 0.0 });
    let u_hi = kron(&h2, &i2); // 8x8 = 2^(2+1) for a 2-qubit state
    let hi_state = State::new(2).unwrap().apply_unitary(&u_hi).unwrap();
    // H on qubit 0 of |00>: |00> and |10> each amplitude 1/√2.
    assert!((hi_state.amplitude(0).re - inv).abs() < 1e-9);
    assert!((hi_state.amplitude(2).re - inv).abs() < 1e-9);
    println!("kron(embed_gate_2x2(H), I) via apply_unitary (H on q0, verified)");

    // 6) Non-destructive vs collapsing measurement semantics.
    let mut sup = State::new(2).unwrap();
    sup.h(0).unwrap();
    sup.cnot(0, 1).unwrap();
    let before = sup.probabilities();
    let mut rng = SplitMix64::seed_from_u64(7);
    let _sample = sup.measure(&mut rng);
    let after = sup.probabilities();
    for (a, b) in before.iter().zip(&after) {
        assert!((a - b).abs() < 1e-12, "measure() must not change the state");
    }
    println!("measure() is non-destructive (probabilities unchanged)");

    let mut rng2 = SplitMix64::seed_from_u64(7);
    let first = sup.measure_collapsing(&mut rng2);
    let second = sup.measure(&mut rng2);
    assert_eq!(
        first, second,
        "collapse makes a later measurement deterministic"
    );
    println!("measure_collapsing() projects the state (2nd measure = {second})");
    println!("  post-collapse norm = {:.6}\n", sup.norm());
    assert!((sup.norm() - 1.0).abs() < 1e-12);

    // 7) Multi-shot statistics for Bell (measure is non-destructive, so one
    //    state is sampled repeatedly).
    let mut rng3 = SplitMix64::seed_from_u64(12345);
    let trials = 20_000usize;
    let mut bell_state = State::new(2).unwrap();
    bell_state.h(0).unwrap();
    bell_state.cnot(0, 1).unwrap();
    let mut c00 = 0usize;
    let mut c11 = 0usize;
    for _ in 0..trials {
        match bell_state.measure(&mut rng3) {
            0 => c00 += 1,
            3 => c11 += 1,
            other => panic!("Bell yielded unexpected outcome {other}"),
        }
    }
    let f00 = c00 as f64 / trials as f64;
    let f11 = c11 as f64 / trials as f64;
    println!("Bell over {trials} shots: P(|00>) = {f00:.3}, P(|11>) = {f11:.3}");
    assert!((f00 - 0.5).abs() < 0.05, "f00 = {f00}");
    assert!((f11 - 0.5).abs() < 0.05, "f11 = {f11}");

    // 8) Multi-shot statistics for GHZ (only |000> and |111> are possible).
    let mut rng4 = SplitMix64::seed_from_u64(999);
    let mut ghz_state = State::new(3).unwrap();
    ghz_state.h(0).unwrap();
    ghz_state.cnot(0, 1).unwrap();
    ghz_state.cnot(1, 2).unwrap();
    let mut g000 = 0usize;
    let mut g111 = 0usize;
    for _ in 0..trials {
        match ghz_state.measure(&mut rng4) {
            0 => g000 += 1,
            7 => g111 += 1,
            other => panic!("GHZ yielded unexpected outcome {other}"),
        }
    }
    let f000 = g000 as f64 / trials as f64;
    let f111 = g111 as f64 / trials as f64;
    println!("GHZ  over {trials} shots: P(|000>) = {f000:.3}, P(|111>) = {f111:.3}");
    assert!((f000 - 0.5).abs() < 0.05, "f000 = {f000}");
    assert!((f111 - 0.5).abs() < 0.05, "f111 = {f111}");

    // 9) Explicit single- and controlled-gate application (lower-level API).
    let mut a = State::new(1).unwrap();
    a.apply_single(&H, 0).unwrap();
    assert!((a.amplitude(0).re - inv).abs() < 1e-9);
    let mut b = State::new(2).unwrap();
    b.x(0).unwrap();
    b.apply_controlled(0, 1, &X).unwrap(); // equals CNOT
    assert!((b.amplitude(3).re - 1.0).abs() < 1e-9);
    println!("\napply_single(H) and apply_controlled(X) match the gate helpers");

    // 10) normalize() is idempotent on an already-normalized state.
    let mut n = State::new(2).unwrap();
    n.h(0).unwrap();
    n.cnot(0, 1).unwrap();
    n.normalize();
    assert!((n.norm() - 1.0).abs() < 1e-12);
    println!("State::normalize() keeps an already-unit state at norm 1");

    // 11) Error surface: the `StateError` variants.
    assert_eq!(State::new(0), Err(StateError::TooManyQubits(0)));
    let mut e = State::new(2).unwrap();
    assert_eq!(e.h(5), Err(StateError::InvalidQubit { qubit: 5, n: 2 }));
    assert_eq!(e.cnot(1, 1), Err(StateError::SameQubits(1)));
    let wrong = DMatrix::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
    assert_eq!(
        State::new(2).unwrap().apply_unitary(&wrong),
        Err(StateError::UnitarySizeMismatch {
            expected: 8,
            got: 4
        })
    );
    println!("StateError exercised: TooManyQubits, InvalidQubit, SameQubits, UnitarySizeMismatch");

    println!("\nAll tours passed.");
}
