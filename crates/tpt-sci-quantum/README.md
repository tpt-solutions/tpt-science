# tpt-sci-quantum

A from-scratch **qubit state-vector simulator** for the `tpt-science` pillar.

The simulator represents an `n`-qubit pure state as a length-`2^n` vector of
complex amplitudes (in the computational basis `|0...0>` … `|1...1>`). Gates
are applied in place by iterating over the relevant pairs of basis states, and
measurement samples a basis index from the Born rule distribution. Supports up
to **20 qubits**.

No external quantum-simulation crates are used (QuantRS2 is disqualified per
ADR 0007 — Apache-2.0-only). The linear algebra is just complex scalars and
explicit index arithmetic over the amplitude vector. Tensor-network support was
not implemented (out of the planned v1 scope).

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
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
