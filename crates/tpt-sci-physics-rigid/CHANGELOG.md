# Changelog

All notable changes to `tpt-sci-physics-rigid` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Rigid-body rotation: `Body` carries an orientation quaternion, angular velocity,
  and isotropic `inertia`; `apply_torque` / `spin` / `quat_to_matrix` etc.
- Quaternion helpers `quat_mul`, `quat_normalize`, `quat_to_matrix`.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-physics-rigid`.

### Added

- From-scratch rigid-body (sphere) physics world with analytic collision
  resolution. `rapier` is disqualified per ADR 0007 (Apache-2.0-only).
- `Body` / `World`: constant gravity, axis-aligned bounding walls, and pairwise
  elastic collisions with a configurable restitution coefficient. Semi-implicit
  (symplectic) Euler integration plus a Baumgarte-style positional correction.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
