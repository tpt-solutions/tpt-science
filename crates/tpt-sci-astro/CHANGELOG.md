# Changelog

All notable changes to `tpt-sci-astro` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- J2 perturbation model: `OrbitalElements::propagate_j2` and `j2_secular_rates`
  give first-order secular nodal-regression / apsidal-precession rates
  (`EARTH_J2` / `EARTH_RADIUS_EQ` supplied).
- Improved `solve_kepler` seeding at `E₀ = M + e·sin(M)` (Danby series) for
  accuracy up to `e → 1`, with a `debug_assert!` bounds guard.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-astro`.

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- From-scratch two-body / Keplerian orbital-mechanics primitives on
  `tpt-math-linalg`. `rapier` is not involved; `quantrs`/`quant-physics` not
  relevant.
- `OrbitalElements`: validated classical Keplerian elements, conversion to/from
  ECI Cartesian state vectors, and time propagation via Kepler's equation.
- Assumes an ideal point-mass central body (no perturbations in the base model).

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
