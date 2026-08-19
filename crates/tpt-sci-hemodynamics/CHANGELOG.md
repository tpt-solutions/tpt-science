# Changelog

All notable changes to `tpt-sci-hemodynamics` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `Vessel` — 1-D compliant artery: cross-sectional area, flow, linear tube-law
  pressure `p = β·(√A − √A0)`, wave speed `c`.
- `tube_law_beta` — wall stiffness from Young's modulus + thickness.
- `womersley_velocity` — analytic pulsatile (Womersley) profile amplitude.
- `casson_viscosity` — shear-thinning (non-Newtonian) Casson correction.
- `Network` — method-of-lines 1-D area/flow advance (`rhs`, `step`).

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-hemodynamics` (spec2.txt expanded vision).

### Added

- 1-D compliant-vessel hemodynamics built on `tpt-sci-cfd-core` and `tpt-sci-ode`,
  using the 1-D augmented Navier–Stokes equations reduced to a vessel centerline.

### Scope (v1)

- 1-D reduced-order vascular flow. 3-D patient-specific, full 0-D/1-D/3-D coupling,
  and a real Womersley complex-Bessel solve are out of scope for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
