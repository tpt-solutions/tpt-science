# tpt-sci-dft-classical

Classical / soft-matter **density functional theory (DFT)** for the
`tpt-science` pillar, provided by wrapping the [`feos`](https://docs.rs/feos)
framework ([`feos_dft`], MIT OR Apache-2.0).

## What's here

- `ClassicalDft` — a handle that owns any concrete `feos` Helmholtz energy
  functional (PC-SAFT, PeTS, …) and is the entry point for profile solves.
- Re-exports of `feos` and `feos_dft` so downstream code can reach the full
  machinery (1-D/2-D/3-D `DFTProfile`, `DFTSolver`, `ConvolverFFT`, geometries)
  directly.

The numerical heavy lifting — functional derivatives, FFT-based convolutions,
Picard / Anderson-mixing solvers, grand-potential minimization — is done by
`feos` itself. `examples/adsorption.rs` shows a complete PC-SAFT density-profile
solve (density profile + adsorption isotherm) wired through this wrapper.

## Scope (v1)

Planar/interface density profiles and adsorption isotherms via a wrapped EOS
functional. Building a *new* functional from scratch and 3-D molecular DFT are
out of scope (use `feos`'s functionals directly).

Dual-licensed under MIT OR Apache-2.0.
