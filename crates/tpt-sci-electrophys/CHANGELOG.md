# Changelog

All notable changes to `tpt-sci-electrophys` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-electrophys` to crates.io (spec2.txt expanded
vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- Cardiac electrophysiology built on `tpt-sci-ode` (membrane kinetics) and
  `tpt-sci-grid` (extracellular/bidomain diffusion).
- `HodgkinHuxley` — classic giant-axon model: gating `m, h, n` with
  voltage-dependent rate laws and ionic current
  `I_ion = ḡNa·m³h·(V−E_Na) + ḡK·n⁴·(V−E_K) + ḡL·(V−E_L)`; `state`, `step`, `voltage`.
- `Tissue` — 2-D monodomain sheet coupling the HH membrane to a 5-point Laplacian
  diffusion (`dVm/dt = −I_ion/Cm + D·∇²Vm`), so an action potential propagates.

### Scope (v1)

- Single-cell HH + 2-D monodomain propagation. Full bidomain (intra/extra split),
  ionic models beyond HH (e.g. Ten Tusscher), and anisotropy are out of scope for
  v1.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
