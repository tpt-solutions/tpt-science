# Changelog

All notable changes to `tpt-sci-climate` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- `EnergyBalanceModel` — 0-D global energy balance with CO₂ forcing
  `ΔF = 5.35·ln(C/C0)` (W/m²), explicit-Euler `step` and `equilibrium_temperature`
  (Newton iteration).
- `grey_radiative_transfer` — single-layer grey-atmosphere surface temperature.
- `ChemistryBox` — constant-production / first-order-loss tracer
  (`dC/dt = P − k·C`), with `steady_state`.
- `MultiBandRadiativeTransfer` and `CorrelatedKRt` — multi-band / correlated-k
  longwave radiative transfer replacing the single grey band
  (`radiative_transfer.rs`).
- `Tracer3D` — 3-D advection–diffusion–reaction tracer on a `tpt-sci-grid` 3-D
  grid (`chemistry_3d.rs`).
- `AtmosphereGcm` — primitive-equation atmospheric GCM dynamical core
  (hydrostatic, optionally non-hydrostatic) coupled to the EBM
  (`gcm.rs`).

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-climate` (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- Reduced-order climate modelling built on `tpt-sci-ode` and `tpt-math-linalg`:
  0-D EBM, simple grey radiative transfer, single-tracer atmospheric chemistry.

### Scope (v1)

- 0-D EBM, multi-band/correlated-k longwave radiative transfer, single- and 3-D
  tracer atmospheric chemistry, and a hydrostatic (optionally non-hydrostatic)
  primitive-equation GCM dynamical core coupled to the EBM. Clouds, moist
  convection, and spectral-dynamics GCM cores remain out of scope.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
