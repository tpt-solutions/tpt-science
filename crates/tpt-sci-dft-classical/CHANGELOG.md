# Changelog

All notable changes to `tpt-sci-dft-classical` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-dft-classical` to crates.io (spec2.txt expanded
vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- Classical / soft-matter DFT provided by wrapping the `feos` framework
  (`feos_dft`, MIT OR Apache-2.0). Functional derivatives, FFT convolutions, and
  Picard / Anderson-mixing solvers are performed by `feos`; this crate assembles
  inputs and exposes a tidy result. `examples/adsorption.rs` shows a complete
  PC-SAFT density-profile solve.
- `ClassicalDft` — a handle owning any concrete `feos` Helmholtz energy
  functional (PC-SAFT, PeTS, …) and the entry point for 1-D profile solves.
- Re-exports of `feos` and `feos_dft` so downstream code can reach the full
  machinery (`DFTProfile`, `DFTSolver`, `ConvolverFFT`, geometries) directly.

### Scope (v1)

- Planar/interface density profiles and adsorption isotherms via a wrapped EOS
  functional. Building a new functional from scratch and 3-D molecular DFT are out
  of scope (use `feos`'s functionals directly).

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
