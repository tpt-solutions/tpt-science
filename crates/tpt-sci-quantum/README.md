# tpt-sci-quantum

A from-scratch **qubit state-vector simulator** for the `tpt-science` pillar.

The simulator represents an `n`-qubit pure state as a length-`2^n` vector of
complex amplitudes (in the computational basis `|0...0>` … `|1...1>`). Gates
are applied in place by iterating over the relevant pairs of basis states, and
measurement samples a basis index from the Born rule distribution. Supports up
to **20 qubits**.

No external quantum-simulation crates are used (QuantRS2 is disqualified per
ADR 0007 — Apache-2.0-only). The linear algebra is just complex scalars and
explicit index arithmetic over the amplitude vector.

A separate `tensor` module (`Circuit`) assembles a full circuit's real-embedded
unitary via a Kronecker (tensor) product formulation and applies it in one
shot through `State::apply_unitary`; non-adjacent two-qubit gates are
SWAP-decomposed.

A `density` module (`DensityMatrix`) adds mixed-state simulation alongside the
pure-state path, for **noise modeling**: an `n`-qubit density matrix (`2^n ×
2^n` complex, trace-1), built from a pure `State` or an explicit probabilistic
mixture of pure states, with gate application via unitary conjugation
(`ρ ↦ UρU†`, reusing the `tensor` module's Kronecker-product unitary
assembly) and Kraus-channel noise application (`ρ ↦ Σ_k K_kρK_k†`), including
ready-made single-qubit bit-flip and depolarizing channels. Because storage is
`O(4^n)` rather than the pure-state path's `O(2^n)`, density-matrix simulation
is practical only for a much smaller qubit count than the 20-qubit pure-state
limit — a dozen or so qubits already means multiple gigabytes of dense `f64`
storage.

Depends on `tpt-math-linalg`, `tpt-math-prob-core`, `num-complex` (published).

## Example

```rust
use tpt_sci_quantum::State;

let mut state = State::new(2).unwrap();
state.h(0).unwrap();
state.cnot(0, 1).unwrap();

let p = state.probabilities();
assert!((p[0] - 0.5).abs() < 1e-9);
assert!((p[3] - 0.5).abs() < 1e-9);

// Destructive, Born-rule-weighted measurement that collapses the state.
use tpt_math_prob_core::SplitMix64;
let mut rng = SplitMix64::seed_from_u64(7);
let _outcome = state.measure_collapsing(&mut rng);
```

Other notable public API: `State::zero`, `State::norm` / `normalize`,
`State::expectation_z`, `State::apply_controlled`, the `Circuit` builder in
the `tensor` module, and in the `density` module `DensityMatrix::from_pure_state`
/ `from_mixture`, `apply_kraus`, and the `bit_flip_kraus` / `depolarizing_kraus`
noise-channel constructors.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
