# Changelog

All notable changes to `tpt-sci-image` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `ImageError` type (`error.rs`): `EmptyImage`, `EmptyAngles`, `AngleCountMismatch`.
  `radon_transform`, `filtered_back_projection`, and `naive_back_projection` now
  return `Result` instead of silently accepting malformed inputs.
- `volume` module: 3-D parallel-beam CT — `radon_transform_3d` and
  `filtered_back_projection_3d` (`Volume`), rotating the beam about `z` so each `z`
  slice is reconstructed independently.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-image`.

### Added

- From-scratch 2-D parallel-beam computed tomography on `tpt-math-signal-fft` and
  `tpt-math-linalg`: `radon_transform`, ram-lak `filtered_back_projection`, and
  `naive_back_projection` (unfiltered adjoint, for comparison).

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
