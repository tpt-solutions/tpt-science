# Changelog

All notable changes to `tpt-sci-ocean` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Criterion benchmark suite (enches/) covering the crate's core hot path.

- `ShallowWater` — 2-D shallow-water model: height `h` + depth-averaged `u`/`v`,
  continuity + momentum with gravity and Coriolis `f`, on a uniform grid
  (`new`, `perturb_center`, `step`, `max_speed`).
- `Ocean3D` — 3-D z-level ocean core: `nz` hydrostatic layers with density from a
  linear equation of state, prognostic temperature/salinity tracers, constant-coefficient
  vertical mixing, and a hydrostatic `step_3d` (`new`, `density`,
  `hydrostatic_pressure`, `mix_vertical`, `step_3d`).
- Non-hydrostatic pressure-correction projection: `Ocean3D::nonhydrostatic_correct`
  and `step_3d_nonhydrostatic` solve the 3-D pressure-Poisson equation with
  `tpt-sci-grid::sparse::conjugate_gradient` and project the provisional velocity
  to be divergence-free.
- Data assimilation module: `nudge` (relaxation toward sparse observations),
  `EnsembleKalmanFilter` (stochastic EnKF), and `Var3D` (3D-Var-lite analysis
  with a background-error covariance).
- `examples/ocean3d.rs` touring the 3-D core, non-hydrostatic projection, and
  assimilation schemes.
- `OceanError::DimensionMismatch` and `OceanError::LinAlg` variants.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-ocean` (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (enches/) covering the crate's core hot path.

- 2-D shallow-water / primitive-equation ocean circulation built on
  `tpt-sci-cfd-core`. `pravash` was audited and rejected (GPL-3.0-only), same as
  CFD.

### Scope (v1)

- 2-D shallow-water circulation (geostrophic balance, gravity waves), a 3-D z-level
  hydrostatic ocean core (density stratification, tracer transport, vertical mixing),
  an optional non-hydrostatic pressure-correction projection, and a nudging / EnKF /
  3D-Var data-assimilation module. A full 3-D primitive-equation ocean GCM with
  sigma/terrain-following coordinates and a complete assimilation system remain out
  of scope for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
