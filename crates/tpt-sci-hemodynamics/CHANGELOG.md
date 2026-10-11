# Changelog

All notable changes to `tpt-sci-hemodynamics` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [Unreleased]

### Fixed

- `Network::rhs` had no spatial coupling, so `dA/dt = −Q` per cell drained the
  inlet area to its floor and every probe read the same deviation. It now uses
  continuity `∂A/∂t = −∂Q/∂x` and momentum `∂Q/∂t = −(A/ρ)∂p/∂x` with
  neighbour differences, an imposed inlet flow and a closed terminal.
- Friction is treated implicitly, which removes a late-time instability at 40
  cells.
- The `wave_reflection` example comment attributed its short-versus-long
  comparison to reflection. It is volume spreading over fewer cells.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-hemodynamics` to crates.io (spec2.txt expanded
vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- 1-D compliant-vessel hemodynamics built on `tpt-sci-cfd-core` and `tpt-sci-ode`,
  using the 1-D augmented Navier–Stokes equations reduced to a vessel centerline.
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

### Scope (v1)

- 1-D reduced-order vascular flow, with a real complex-Bessel Womersley solve
  (`src/womersley.rs`) and a 0-D/1-D/3-D coupling module (`src/coupling.rs`).

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
