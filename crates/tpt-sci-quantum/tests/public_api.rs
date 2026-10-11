//! Public-API invariants for the state-vector simulator.

use tpt_math_prob_core::SplitMix64;
use tpt_sci_quantum::State;

#[test]
fn bell_state_has_equal_weight_on_correlated_outcomes() {
    let mut s = State::new(2).unwrap();
    s.h(0).unwrap();
    s.apply_cnot(0, 1).unwrap();
    let p = s.probabilities();
    assert!((p[0] - 0.5).abs() < 1e-12);
    assert!((p[3] - 0.5).abs() < 1e-12);
    assert!(p[1].abs() < 1e-12 && p[2].abs() < 1e-12);
    // expectation_z is the Z⊗…⊗Z parity observable: +1 on a Bell pair.
    assert!((s.expectation_z() - 1.0).abs() < 1e-12);
}

#[test]
fn unitary_gates_preserve_the_norm() {
    let mut s = State::new(3).unwrap();
    s.h(0).unwrap();
    s.t(1).unwrap();
    s.apply_cnot(0, 2).unwrap();
    s.s(2).unwrap();
    s.x(1).unwrap();
    assert!((s.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn collapsing_measurement_leaves_a_basis_state() {
    let mut s = State::new(2).unwrap();
    s.h(0).unwrap();
    s.apply_cnot(0, 1).unwrap();
    let mut rng = SplitMix64::seed_from_u64(7);
    let outcome = s.measure_collapsing(&mut rng);
    let p = s.probabilities();
    assert!(
        (p[outcome] - 1.0).abs() < 1e-12,
        "state not collapsed to |{outcome}>"
    );
    assert!((s.norm() - 1.0).abs() < 1e-12);
}
