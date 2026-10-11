# Changelog

All notable changes to `tpt-sci-electrophys` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Fixed

- `TenTusscher` now follows the published epicardial CellML model
  (`tentusscher_noble_noble_panfilov_2004_epi`). The previous gate time constants
  were 1–3 ms, which gave no action-potential plateau. The gating kinetics,
  conductances, and Ca²⁺/Na⁺/K⁺ balance with the sarcoplasmic reticulum are now
  the published ones. The single-cell AP has a resting potential of about −86 mV,
  a peak of about +35 mV, a plateau, and an APD (−40 mV) of about 260 ms.

### Added

- `LandActiveTension`: the Land et al. (2017) human-cardiomyocyte active-tension
  model (intact, 37 °C), with length and velocity dependence, ported from the
  authors' release (`Land2016.m`). The length ratio and its rate are inputs. The
  output matches an independent port to six digits, including a shortening protocol.

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- `HodgkinHuxley` — classic giant-axon model: gating `m, h, n` with
  voltage-dependent rate laws and ionic current
  `I_ion = ḡNa·m³h·(V−E_Na) + ḡK·n⁴·(V−E_K) + ḡL·(V−E_L)`; `state`, `step`, `voltage`.
- `Tissue` — 2-D monodomain sheet coupling the HH membrane to a 5-point Laplacian
  diffusion (`dVm/dt = −I_ion/Cm + D·∇²Vm`), so an action potential propagates.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-electrophys` (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.

- Cardiac electrophysiology built on `tpt-sci-ode` (membrane kinetics) and
  `tpt-sci-grid` (extracellular/bidomain diffusion).

### Scope (v1)

- Single-cell HH + 2-D monodomain propagation. Full bidomain (intra/extra split),
  ionic models beyond HH (e.g. Ten Tusscher), and anisotropy are out of scope for
  v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
