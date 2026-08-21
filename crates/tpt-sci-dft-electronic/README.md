# tpt-sci-dft-electronic

Simple **1-D electronic-structure DFT** (Kohn–Sham, LDA exchange-correlation)
for the `tpt-science` pillar, built from scratch.

This crate was `flagged-needs-audit-first` in spec2.txt. No Rust prior art
exists for Kohn–Sham LDA/GGA/band-structure, so per the 2026-08 audit it is
scoped as a multi-phase undertaking like `tpt-sci-physics-rigid` /
`tpt-sci-quantum`.

## Features (v1)

* `Grid1D` — uniform 1-D real-space grid (`x()`, `n`, `dx`).
* `lda_xc` — LDA exchange-correlation energy density `e_xc(ρ)` (Slater exchange +
  Perdew–Zunger-style correlation), with `e_xc ≤ 0` (attractive).
* `KohnSham` / `KohnShamResult` — self-consistent 1-D Kohn–Sham solver:
  finite-difference kinetic-energy Laplacian, Hartree (1-D Poisson), XC, Jacobi
  diagonalization, returning occupied orbitals and the total energy.
* `KohnSham3D` / `KohnSham3DResult` — self-consistent **3-D** real-space-grid
  Kohn–Sham solver built on the `tpt-sci-grid` 3-D sparse Laplacian. Diagonalizes
  `H = −½∇² + V_eff` with a Lanczos eigensolver and converges the density
  self-consistently (Hartree + XC). A non-interacting `solve_bare` mode is also
  provided for analytic checks.
* `XcFunctional` trait with two implementations: `Lda` (wraps `lda_xc`) and `Pbe`
  (the Perdew–Burke–Ernzerhof GGA, depending on `ρ` and `|∇ρ|`, with analytic
  partial derivatives `∂ε/∂ρ` and `∂ε/∂|∇ρ|`).
* `Pseudopotential` — a softened, local, norm-conserving-style analytic
  `V_ps(r) = −Z·erf(r/σ)/r` (finite at the origin, `−Z/r` asymptote), making
  multi-electron 3-D atoms tractable as a local-potential problem.
* `PeriodicPotential1D` — periodic-boundary-condition band structure via Bloch /
  phase-twisted plane-wave expansion, with Monkhorst–Pack k-point sampling and a
  basic `E(k)` band structure.

## Example

```rust
use tpt_sci_dft_electronic::{Grid1D, KohnSham, lda_xc};

let grid = Grid1D::new(101, -5.0, 5.0).unwrap();
// Harmonic-like external well.
let v_ext: Vec<f64> = grid.x().iter().map(|&x| 0.5 * x * x).collect();
let mut ks = KohnSham::new(grid, v_ext, 2).unwrap(); // 2 electrons
let res = ks.solve(50);
assert!(res.total_energy.is_finite());
// LDA XC energy density must be <= 0 (attractive exchange).
assert!(lda_xc(1.0) <= 0.0);
```

The `harmonic_atom` example (`cargo run --example harmonic_atom -p tpt-sci-dft-electronic`)
runs a two-electron solve in a harmonic confining well.

## Scope (v1)

This crate now covers, from scratch:

* **1-D Kohn–Sham LDA** (the original v1 scope).
* **3-D real-space-grid Kohn–Sham** (self-consistent, sparse Laplacian, Lanczos
  diagonalization) — closes the "3-D solver" out-of-scope item.
* **GGA / PBE** exchange-correlation functional alongside LDA — closes the
  "GGA/meta-GGA" item.
* **Local (norm-conserving-style) pseudopotentials** — closes the
  "pseudopotentials" item.
* **Periodic boundary conditions + k-point sampling + band structure** — closes
  the "band structures" item.

Still out of scope (per the research-grade, no-prior-art constraints): PAW / non-local
pseudopotentials, hybrid and meta-GGA functionals, spin-polarized (collinear/
non-collinear) magnetization, and relativistic / scalar-relativistic treatments.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
