# tpt-sci-md

Classical molecular dynamics (MD) for the `tpt-science` pillar, built entirely
from scratch — no wrapped engine (`lumol` was audited and rejected: BSD-3-Clause,
alpha/stale).

## What's here

- `Particle` — point particle with position/velocity/force/mass/species.
- `lennard_jones` / `Forces::lennard_jones` — pairwise Lennard-Jones 12-6
  interactions with cut-off and minimum-image periodic boundaries.
- `Integrator` — velocity-Verlet stepping with kinetic-energy, temperature, and
  a Berendsen-style thermostat.
- `rdf` — radial distribution function `g(r)` for structural analysis.

## Example

```rust
use tpt_sci_md::{Particle, Integrator, Forces};
use tpt_math_linalg::tpt_math_linalg_dense::DVector;

let mut parts = vec![
    Particle::new(0, DVector::from_row_slice(&[0.0, 0.0, 0.0]), DVector::zeros(3)).unwrap(),
    Particle::new(1, DVector::from_row_slice(&[1.0, 0.0, 0.0]), DVector::zeros(3)).unwrap(),
];
let int = Integrator::new(10.0, 1.0, 0.005).unwrap();
for _ in 0..100 {
    int.velocity_verlet(&mut parts);
}
```

## Scope (v1)

Mono/few-species Lennard-Jones fluids in a cubic periodic box. EAM, long-range
electrostatics (PPPM), constrained bonds, and neighbour lists are out of scope
for v1.

Dual-licensed under MIT OR Apache-2.0.
