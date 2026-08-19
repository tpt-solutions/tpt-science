# Changelog

All notable changes to `tpt-sci-climate` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `EnergyBalanceModel` — 0-D global energy balance with CO₂ forcing
  `ΔF = 5.35·ln(C/C0)` (W/m²), explicit-Euler `step` and `equilibrium_temperature`
  (Newton iteration).
- `grey_radiative_transfer` — single-layer grey-atmosphere surface temperature.
- `ChemistryBox` — constant-production / first-order-loss tracer
  (`dC/dt = P − k·C`), with `steady_state`.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-climate` (spec2.txt expanded vision).

### Added

- Reduced-order climate modelling built on `tpt-sci-ode` and `tpt-math-linalg`:
  0-D EBM, simple grey radiative transfer, single-tracer atmospheric chemistry.

### Scope (v1)

- 0-D EBM, simple grey radiative transfer, single-tracer chemistry. GCMs, full
  radiative-transfer bands, and 3-D atmospheric chemistry are out of v1 scope.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
