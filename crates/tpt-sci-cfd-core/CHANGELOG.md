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

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-cfd-core` (spec2.txt expanded vision).

### Added

- From-scratch 2-D incompressible Navier–Stokes solver (no wrapped engine;
  `pravash` was audited and rejected, GPL-3.0-only) on a uniform collocated grid
  with fractional-step pressure projection. Foundation for `tpt-sci-hemodynamics`
  and `tpt-sci-ocean`.

### Scope (v1)

- 2-D, uniform, explicit. Unstructured meshes, implicit / SIMPLE solvers, and a
  full coupled `k`-`ω` SST two-equation turbulence model are out of scope for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
