# Changelog

All notable changes to `tpt-sci-dft-electronic` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [Unreleased]

### Fixed

- `eigen::jacobi` ran at most 100 rotations, which left matrices above about
  n = 10 undiagonalised. It now runs cyclic sweeps until the off-diagonal norm
  converges.
- `jacobi` returns eigenpairs in ascending order, as `solve()` assumes.
- The Hartree potential had the wrong sign in the Thomas solve and a zero
  Dirichlet boundary that pushed electrons to the walls. It now uses the exact
  1-D kernel `V(x) = π ∫ |x − x′| ρ(x′) dx′`.
- The total energy double-counted `V_ext` and `V_H`. It is now
  `E = Σ f·ε − E_H + ∫ ρ (ε_xc − v_xc)`.
- `hydrogen_atom` counts nodes only above a noise threshold. Its absolute
  energies are not yet converged.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-dft-electronic` to crates.io (spec2.txt expanded
vision). This crate was `flagged-needs-audit-first` in spec2.txt; no Rust
prior art exists for Kohn–Sham LDA/GGA/band-structure, so it is treated as a
multi-phase undertaking like `tpt-sci-physics-rigid` / `tpt-sci-quantum`.

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- From-scratch **1-D Kohn–Sham** electronic-structure DFT (LDA XC), scoped to 1-D
  model systems.
- `Grid1D` — uniform 1-D real-space grid.
- `lda_xc` — LDA exchange-correlation energy density `e_xc(ρ)` (Slater exchange +
  Perdew–Zunger-style correlation), `e_xc ≤ 0`.
- `KohnSham` / `KohnShamResult` — self-consistent 1-D Kohn–Sham solver:
  finite-difference kinetic-energy Laplacian, Hartree (1-D Poisson), XC, Jacobi
  diagonalization, occupied orbitals + total energy.
- `KohnSham3D` / `KohnSham3DResult` — self-consistent **3-D** real-space-grid
  Kohn–Sham solver on the `tpt-sci-grid` 3-D sparse Laplacian (Lanczos
  diagonalization, Hartree + XC SCF, plus a non-interacting `solve_bare` mode).
- `XcFunctional` trait with `Lda` (wraps `lda_xc`) and `Pbe` (Perdew–Burke–
  Ernzerhof GGA, with analytic `∂ε/∂ρ` and `∂ε/∂|∇ρ|` derivatives).
- `Pseudopotential` — softened local `V_ps(r) = −Z·erf(r/σ)/r` (finite at the
  origin, `−Z/r` asymptote).
- `PeriodicPotential1D` — periodic-boundary-condition band structure (Bloch /
  phase-twisted plane-wave expansion) with Monkhorst–Pack k-point sampling.

### Scope (v1)

3-D solver, GGA, pseudopotentials, and band structures are implemented. PAW /
non-local pseudopotentials, hybrid / meta-GGAs, and spin polarization remain
out of scope.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
