# Changelog

All notable changes to `tpt-sci-cfd-core` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `CollocatedGrid` — uniform 2-D collocated grid with cell-centred velocity
  storage (`idx`, `len`).
- `Step` — fractional-step (Chorin) incompressible Navier–Stokes solver:
  explicit advection–diffusion (`momentum`) plus pressure-Poisson projection
  (`project`) enforcing `∇·u = 0`; `advance`, `max_divergence` quality metric.
- `turbulence::eddy_viscosity` — algebraic Smagorinsky eddy-viscosity estimate.
- `SimpleSolver` — SIMPLE/PISO-style implicit pressure-correction
  (`predict`/`correct`/`advance`): provisional explicit momentum, pressure-Poisson
  solve via the sparse conjugate-gradient solver, divergence-free correction
  (`simple.rs`).
- `KOmegaSst` — two-equation `k`-`ω` SST (Menter) turbulence closure, running
  alongside the algebraic eddy-viscosity model (`komega_sst.rs`).
- `UnstructuredMesh` — additive unstructured (triangular) finite-volume path:
  least-squares cell-gradient reconstruction and a diffusion + upwind-advection
  residual assembly, alongside the structured `CollocatedGrid` (`unstructured.rs`).

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-cfd-core` (spec2.txt expanded vision).

### Added

- From-scratch 2-D incompressible Navier–Stokes solver (no wrapped engine;
  `pravash` was audited and rejected, GPL-3.0-only) on a uniform collocated grid
  with fractional-step pressure projection. Foundation for `tpt-sci-hemodynamics`
  and `tpt-sci-ocean`.

### Scope (v1)

- 2-D, uniform collocated grid. Implemented (v1): SIMPLE/PISO implicit
  pressure-correction, `k`-`ω` SST turbulence, and unstructured triangular
  finite-volume assembly. Still out of scope: 3-D tetrahedral assembly,
  production-grade parallel solvers.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
