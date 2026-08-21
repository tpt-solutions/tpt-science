# tpt-sci-md

Classical molecular dynamics (MD) for the `tpt-science` pillar, built entirely
from scratch — no wrapped engine (`lumol` was audited and rejected: BSD-3-Clause,
alpha/stale).

## Features

* `Particle` — point particle with position/velocity/force/mass/species,
  validated construction (`new`, `new_with_species`).
* `lennard_jones` / `Forces::lennard_jones` — pairwise Lennard-Jones 12-6
  interactions with cut-off + shift and minimum-image periodic boundaries.
* `EamParams` / `eam_forces` / `Forces::eam` — a Finnis-Sinclair-style
  embedded-atom-method (EAM) potential (pairwise repulsion `φ` + a
  square-root embedding function `F(ρ) = -A·√ρ` of a local electron-density
  sum `ρ_i = Σ f(r_ij)`), as a drop-in alternative to Lennard-Jones.
* `Ewald` — Ewald summation for periodic long-range electrostatics
  (`Ewald::energy_forces`): real-space `erfc(α·r)/r` sum + a direct
  reciprocal-space Fourier sum + self-energy/background corrections.
* `Bond` / `Shake` — SHAKE position constraints and a RATTLE-style velocity
  projection to hold pairwise bond lengths fixed, driven through
  `Integrator::velocity_verlet_constrained`.
* `CellList` / `neighbor_pairs_brute_force` / `Forces::lennard_jones_cells` —
  linked-cell (cell-list) neighbor finding for a cut-off radius, as a faster
  alternative to the `O(n²)` pairwise scan for Lennard-Jones.
* `Integrator` — velocity-Verlet stepping with kinetic-energy, temperature, and a
  Berendsen-style thermostat; `velocity_verlet`, `temperature`, `thermostat`,
  plus a generic `step_with` hook so other potentials (EAM, Ewald, ...) can
  drive the same integrator.
* `rdf` — radial distribution function `g(r)` for structural analysis.

The engine models mono-/few-species systems in a cubic periodic box and is
sized for teaching, prototyping, and coupling into the broader `tpt-science`
platform — not for production-scale biomolecular simulation.

## Example

```rust
use tpt_sci_md::{Particle, Integrator, Forces};
use tpt_math_linalg::tpt_math_linalg_dense::DVector;

let mut parts = vec![
    Particle::new(0, DVector::from_row_slice(&[0.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
    Particle::new(1, DVector::from_row_slice(&[1.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
];
let int = Integrator::new(10.0, 1.0, 0.005).unwrap();
for _ in 0..100 {
    int.velocity_verlet(&mut parts);
}
```

The `lennard_jones` example (`cargo run --example lennard_jones -p tpt-sci-md`)
runs a short LJ trajectory and reports temperature / energy.

## Scope (v1)

Mono/few-species systems in a cubic periodic box. EAM, long-range
electrostatics, constrained bonds, and neighbour lists are implemented, with
the following honest simplifications relative to a production MD package:

* **EAM**: a generic Finnis-Sinclair-style form (polynomial pairwise
  repulsion + square-root embedding). Parameters are chosen only to be
  physically sound (short-range repulsive, longer-range attractive) — they
  are not fit to reproduce any specific real metal.
* **Long-range electrostatics**: real Ewald summation, not PPPM. The
  reciprocal-space sum is a **direct discrete Fourier sum** over `k`-vectors
  (`O(N·kmax³)`), not an FFT-based particle-mesh solve — this crate has no
  FFT dependency available (ADR 0007: hand-rolled code only). The direct sum
  is exact for the reciprocal term itself, just less scalable than PPPM for
  large systems or a large `kmax`.
* **Constrained bonds**: full SHAKE (position constraint) plus a
  RATTLE-style velocity projection (`Shake::constrain_velocities`) are both
  implemented — pairwise bond-length constraints only (no angle/dihedral
  constraints).
* **Neighbour lists**: a linked-cell (`CellList`) structure is available as
  an alternative, cross-checked-against-brute-force pair-finding path,
  currently wired up for Lennard-Jones (`Forces::lennard_jones_cells`); EAM
  and the Ewald real-space sum still use the `O(n²)` scan internally.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
