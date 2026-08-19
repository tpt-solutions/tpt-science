# Changelog

All notable changes to `tpt-sci-sim-core` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `OdeSubModel` wraps any `tpt-sci-ode` problem as a `SubModel` (used directly by
  the `multi_scale_cookbook` example).
- `DiffusionSubModel` drives a 1-D `tpt-sci-grid` Laplacian field as a sub-model.
- `Coupling` / `CouplingFn` for cross-scale state mapping after every sub-step.
- `Simulation::snapshot` / `Simulation::restore` (`Checkpoint`) for resumable,
  reproducible runs.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-sim-core`.

### Added

- Multi-scale simulation orchestration above `tpt-sci-ode` and `tpt-sci-grid`.
- Time-stepping across heterogeneous sub-models: each `SubModel` advances on its
  own internal `max_step` while a `Simulation` drives them all to a shared target
  time (`step_until`).
- Cross-scale coupling and checkpoint snapshot/restore, verified by tests.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
