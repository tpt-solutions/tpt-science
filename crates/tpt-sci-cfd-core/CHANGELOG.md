# Changelog

All notable changes to `tpt-sci-cfd-core` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-cfd-core` to crates.io (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- From-scratch 2-D incompressible Navier–Stokes solver (no wrapped engine;
  `pravash` was audited and rejected, GPL-3.0-only) on a uniform collocated grid
  with fractional-step pressure projection. Foundation for `tpt-sci-hemodynamics`
  and `tpt-sci-ocean`.
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

### Fixed

- **Collocated-grid SIMPLE corner-cell divergence**: the pressure-Poisson matrix
  was assembled from combined per-cell flux columns, which added spurious
  `1/(dx·dy)` cross terms at edge/corner cells and broke the required block
  adjoint identity `A = FₓFₓᵀ + F_yF_yᵀ`. The assembly now accumulates each
  axis's outer product separately, making the projection exactly
  divergence-free (the previously `#[ignore]`d
  `pressure_correction_reduces_divergence` test now passes and a permanent
  adjoint-identity regression test guards the property).

### Scope (v1)

- 2-D, uniform collocated grid. Implemented (v1): SIMPLE/PISO implicit
  pressure-correction, `k`-`ω` SST turbulence, and unstructured triangular
  finite-volume assembly. Still out of scope: 3-D tetrahedral assembly,
  production-grade parallel solvers.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
