# Changelog

All notable changes to `tpt-sci-sim-core` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-sim-core` to crates.io.

### Added

- `cardiac_electromechanics` example: a Ten Tusscher ventricular cell
  (`tpt-sci-electrophys`) coupled through its Ca²⁺ transient to the Land et al.
  (2017) active-tension model (`LandActiveTension`), with the authors' constant-
  velocity shortening protocol as the prescribed length. The two sub-models are
  stepped by the orchestrator. The checks are the AP plateau and APD, the tension
  lagging the calcium peak, and shortening lowering the tension.

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- Multi-scale simulation orchestration above `tpt-sci-ode` and `tpt-sci-grid`.
- Time-stepping across heterogeneous sub-models: each `SubModel` advances on its
  own internal `max_step` while a `Simulation` drives them all to a shared target
  time (`step_until`).
- Cross-scale coupling and checkpoint snapshot/restore, verified by tests.
- `OdeSubModel` wraps any `tpt-sci-ode` problem as a `SubModel` (used directly by
  the `multi_scale_cookbook` example).
- `DiffusionSubModel` drives a 1-D `tpt-sci-grid` Laplacian field as a sub-model.
- `Coupling` / `CouplingFn` for cross-scale state mapping after every sub-step.
- `Simulation::snapshot` / `Simulation::restore` (`Checkpoint`) for resumable,
  reproducible runs.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
