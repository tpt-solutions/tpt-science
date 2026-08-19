# tpt-sci-cfd-core

Core **incompressible Navier–Stokes** finite-volume solver for the
`tpt-science` pillar, built from scratch (`pravash` was audited and rejected:
GPL-3.0-only).

## Features

* `CollocatedGrid` — uniform 2-D collocated grid (`nx × ny`, physical size
  `lx × ly`) with cell-centred velocity storage (`idx`, `len`).
* `Step` — fractional-step (Chorin) scheme: explicit advection–diffusion
  (`momentum`) plus a pressure-Poisson projection (`project`) that enforces
  `∇·u = 0`. `advance()` runs one full fractional step; `max_divergence()` is a
  quality metric.
* `Boundary` walls (top/bottom/left/right) with `Step::set_boundary` for
  moving-lid / no-slip conditions.
* `turbulence::eddy_viscosity` — algebraic Smagorinsky eddy-viscosity estimate.

The solver is intentionally 2-D and explicit — a teaching / coupling primitive,
not a production CFD code. It is the foundation `tpt-sci-hemodynamics` and
`tpt-sci-ocean` build on.

## Example

```rust
use tpt_sci_cfd_core::{CollocatedGrid, Step, Boundary};

// 32×32 cavity, lid moving at u = 1.0 on the top wall.
let grid = CollocatedGrid::new(32, 32, 1.0, 1.0).unwrap();
let mut step = Step::new(grid, 1e-2, 0.01, 1.0);
step.set_boundary(Boundary::Top, 1.0);
assert!(step.advance());
```

The `cavity` example (`cargo run --example cavity -p tpt-sci-cfd-core`) drives a
lid-driven cavity to steady state.

## Scope (v1)

2-D, uniform, explicit. Unstructured meshes, implicit / SIMPLE solvers, and a
full coupled `k`-`ω` SST two-equation turbulence model are out of scope for v1
(the `turbulence` module provides an algebraic eddy viscosity only).

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
