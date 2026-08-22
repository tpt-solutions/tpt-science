# Changelog

All notable changes to `tpt-sci-hemodynamics` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- `Vessel` — 1-D compliant artery: cross-sectional area, flow, linear tube-law
  pressure `p = β·(√A − √A0)`, wave speed `c`.
- `tube_law_beta` — wall stiffness from Young's modulus + thickness.
- `womersley_velocity` — analytic pulsatile (Womersley) profile amplitude.
- `casson_viscosity` — shear-thinning (non-Newtonian) Casson correction.
- `Network` — method-of-lines 1-D area/flow advance (`rhs`, `step`).
- Real Womersley complex-Bessel solution (`womersley.rs`): series-based
  `bessel_j0`/`bessel_j1`, `womersley_velocity_profile`, and
  `womersley_flow_rate_*`, replacing the approximate profile from v0.1.0.
- 0-D/1-D/3-D coupling interface (`coupling.rs`): `Windkessel` three-element
  model plus the `CfdCoupling` trait and `couple` driver connecting the 1-D
  network to a `tpt-sci-cfd-core` domain.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-hemodynamics` (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- 1-D compliant-vessel hemodynamics built on `tpt-sci-cfd-core` and `tpt-sci-ode`,
  using the 1-D augmented Navier–Stokes equations reduced to a vessel centerline.

### Scope (v1)

- 1-D reduced-order vascular flow with approximate (non-complex-Bessel)
  Womersley profiles and no coupling interface. Both limitations have since
  been lifted in `[Unreleased]`: see the real complex-Bessel Womersley solve
  (`src/womersley.rs`) and the 0-D/1-D/3-D coupling module (`src/coupling.rs`).

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
