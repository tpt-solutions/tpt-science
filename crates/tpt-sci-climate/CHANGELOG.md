# Changelog

All notable changes to `tpt-sci-climate` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.2.0] — 2026-10-11

### Changed

- **Breaking:** `grey_radiative_transfer` returns `Result<f64, ClimateError>`
  instead of `f64`.
- The energy-balance step is a linearised backward-Euler update, unconditionally
  stable for the radiative term.
- `SIGMA` is read from `tpt-sci-units`.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-climate` to crates.io (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- Reduced-order climate modelling built on `tpt-sci-ode` and `tpt-math-linalg`:
  0-D EBM, simple grey radiative transfer, single-tracer atmospheric chemistry.
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

### Scope (v1)

- 0-D EBM, multi-band/correlated-k longwave radiative transfer, single- and 3-D
  tracer atmospheric chemistry, and a hydrostatic (optionally non-hydrostatic)
  primitive-equation GCM dynamical core coupled to the EBM. Clouds, moist
  convection, and spectral-dynamics GCM cores remain out of scope.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
