# Changelog

All notable changes to `tpt-sci-astro` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [Unreleased]

### Added

- `environment` module: `atmospheric_density`, `in_earth_shadow` and
  `srp_acceleration`.
- `frames` module: `perifocal_to_eci`, `cross3`, the anomaly conversions
  `true_to_eccentric` and `eccentric_to_true`, and `solve_kepler`.
- Constants `SUN_MU`, `MOON_MU`, `MOON_DISTANCE_KM`, `ASTRONOMICAL_UNIT_KM` and
  `SOLAR_PRESSURE_1AU`.

### Changed

- Constants moved into a `constants` module (re-exported). `EARTH_MU` and
  `EARTH_RADIUS_EQ` are read from `tpt-sci-units`.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-astro` to crates.io.

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- From-scratch two-body / Keplerian orbital-mechanics primitives on
  `tpt-math-linalg`. `rapier` is not involved; `quantrs`/`quant-physics` not
  relevant.
- `OrbitalElements`: validated classical Keplerian elements, conversion to/from
  ECI Cartesian state vectors, and time propagation via Kepler's equation.
- Assumes an ideal point-mass central body (no perturbations in the base model).
- J2 perturbation model: `OrbitalElements::propagate_j2` and `j2_secular_rates`
  give first-order secular nodal-regression / apsidal-precession rates
  (`EARTH_J2` / `EARTH_RADIUS_EQ` supplied).
- Improved `solve_kepler` seeding at `E₀ = M + e·sin(M)` (Danby series) for
  accuracy up to `e → 1`, with a `debug_assert!` bounds guard.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
