# Changelog

All notable changes to `tpt-sci-dft-electronic` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `Grid1D` — uniform 1-D real-space grid.
- `lda_xc` — LDA exchange-correlation energy density `e_xc(ρ)` (Slater exchange +
  Perdew–Zunger-style correlation), `e_xc ≤ 0`.
- `KohnSham` / `KohnShamResult` — self-consistent 1-D Kohn–Sham solver:
  finite-difference kinetic-energy Laplacian, Hartree (1-D Poisson), XC, Jacobi
  diagonalization, occupied orbitals + total energy.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-dft-electronic` (spec2.txt expanded vision).

### Added

- From-scratch **1-D Kohn–Sham** electronic-structure DFT (LDA XC), scoped to 1-D
  model systems. This crate was `flagged-needs-audit-first` in spec2.txt; no Rust
  prior art exists for Kohn–Sham LDA/GGA/band-structure, so it is treated as a
  multi-phase undertaking like `tpt-sci-physics-rigid` / `tpt-sci-quantum`.

### Scope (v1)

- 1-D model systems only. Multi-electron 3-D atoms, GGA/meta-GGAs,
  pseudopotentials, and band structures are out of scope for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
