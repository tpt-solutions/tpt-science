# Changelog

All notable changes to `tpt-sci-quantum` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `State::measure_collapsing()` for post-measurement state collapse (Qiskit/Cirq
  semantics); `measure()` remains non-destructive for inspection.

### Changed

- `StateError` moved into its own `error.rs` module, matching the per-crate error
  convention; re-exported from the crate root (`UnitarySizeMismatch` also lives
  there).

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-quantum`.

### Added

- From-scratch qubit state-vector simulator (up to 20 qubits) on
  `tpt-math-linalg` / `tpt-math-prob-core`. QuantRS2 is disqualified per
  ADR 0007 (Apache-2.0-only).
- `State` with gate application (`h`, `cnot`, …), `Kronecker` circuit
  formulation, and Born-rule `measure` / `probabilities`. Non-adjacent two-qubit
  gates are SWAP-decomposed.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
