# Changelog

All notable changes to `tpt-sci` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Initial umbrella crate: re-exports every `tpt-sci-*` domain crate as a module
  (`tpt_sci::ode`, `tpt_sci::grid`, …), each behind a Cargo feature of the same
  name. Default features: `ode`, `grid`, `sim-core`; `full` enables all domains.

## [0.1.0]

- Not released (`publish = false`). See the workspace `RELEASE.md`.
