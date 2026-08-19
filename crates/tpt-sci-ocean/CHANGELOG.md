# Changelog

All notable changes to `tpt-sci-ocean` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `ShallowWater` — 2-D shallow-water model: height `h` + depth-averaged `u`/`v`,
  continuity + momentum with gravity and Coriolis `f`, on a uniform grid
  (`new`, `perturb_center`, `step`, `max_speed`).

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-ocean` (spec2.txt expanded vision).

### Added

- 2-D shallow-water / primitive-equation ocean circulation built on
  `tpt-sci-cfd-core`. `pravash` was audited and rejected (GPL-3.0-only), same as
  CFD.

### Scope (v1)

- 2-D shallow-water circulation (geostrophic balance, gravity waves). Full 3-D
  primitive-equation ocean GCM, hydrostatic/non-hydrostatic vertical coordinates,
  and data assimilation are out of scope for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
