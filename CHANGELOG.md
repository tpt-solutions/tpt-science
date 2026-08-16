# Changelog

All notable changes to the `tpt-science` workspace crates are documented here.
This project adheres to [Semantic Versioning](https://semver.org); the initial
v0.1.0 covers the implementation of all eight Phase 1–3 crates.

## [Unreleased]

### Added

- `tpt-sci-reaction-network` — from-scratch Catalyst.jl-style species/rate/
  stoichiometry DSL and mass-action ODE builder, depending on `tpt-sci-ode`.
  Provides a programmatic [`ReactionNetwork`] builder and a textual
  (`kB, S + E --> SE`) DSL, a compiled [`ReactionSystem`] exposing the
  stoichiometry matrix and per-reaction rate vector, custom (non-mass-action)
  rate laws, and an `OdeProblem` bridge into `tpt-sci-ode`. The stochastic
  SSA backend, SDE/jump models, SBML I/O, network analysis, and conservation
  laws are explicitly out of v1 (the `rebop` crate is the intended wrap target
  for a future stochastic backend).

## [0.1.0] — 2026-08-16

Initial implementation of the `tpt-science` simulation/modeling substrate.

### Added

- `tpt-sci-ode` — ODE/DAE solving by wrapping `diffsol` (dual-licensed
  MIT OR Apache-2.0); depends on `tpt-math-numeric`.
- `tpt-sci-grid` — from-scratch structured finite-difference grids and
  Laplacian assembly; depends on `tpt-math-linalg`.
- `tpt-sci-sim-core` — multi-scale orchestration (adaptive sub-stepping,
  cross-scale coupling, checkpoint snapshot/restore) over `tpt-sci-ode` and
  `tpt-sci-grid`.
- `tpt-sci-ppl` — from-scratch NUTS Hamiltonian Monte Carlo backend and model
  DSL on `tpt-math-autodiff-rev` / `tpt-math-prob` (the `nuts-rs` wrap planned
  in `spec.txt` was dropped).
- `tpt-sci-image` — from-scratch 2-D parallel-beam CT (Radon transform, ram-lak
  filtered back-projection, naive back-projection) on `tpt-math-signal-fft`.
- `tpt-sci-physics-rigid` — from-scratch rigid-body sphere world with analytic
  collision resolution (rapier disqualified per ADR 0007).
- `tpt-sci-quantum` — from-scratch qubit state-vector simulator (≤20 qubits) on
  `tpt-math-linalg` / `tpt-math-prob-core` (QuantRS2 disqualified per ADR 0007).
- `tpt-sci-astro` — from-scratch two-body / Keplerian orbital-mechanics
  primitives on `tpt-math-linalg`.

### Documentation

- Per-crate READMEs for all eight crates.
- `no_std` audit recorded in the workspace README: pillar is std-only by design.

### Notes

- `tpt-sci-reaction-network` (Catalyst.jl-style DSL) was implemented after the
  dedicated ecosystem research pass (see the `[Unreleased]` section).
- Crates are `publish = false` and consumed as path/workspace dependencies; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
