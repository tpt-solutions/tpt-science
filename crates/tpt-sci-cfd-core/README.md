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
* `Boundary` walls with `Step::set_boundary` for moving-lid / no-slip
  conditions. `Step::apply_walls` currently enforces `Top`/`Bottom`; `Left`/
  `Right` values are accepted by the API but not yet applied.
* `turbulence::eddy_viscosity` — algebraic Smagorinsky eddy-viscosity estimate.
* `SimpleSolver` — SIMPLE/PISO-style **implicit** pressure-correction on the
  same structured grid. [`SimpleSolver::predict`] does an explicit momentum
  step, [`SimpleSolver::solve_pressure`] builds the pressure Poisson equation
  and solves it with `tpt-sci-grid`'s sparse conjugate-gradient solver, and
  [`SimpleSolver::correct`] subtracts the pressure gradient to enforce
  `∇·u = 0`. [`SimpleSolver::advance`] runs a full step.
* `KOmegaSst` — full two-equation **`k`-`ω` SST** (Menter, 2003) closure:
  `k`/`ω` transport with the SST `F1`/`F2` blend, `a1` production limiter, and
  the `k-ω`/`k-ε` cross-diffusion term. [`KOmegaSst::eddy_viscosity_at`]
  returns the blended eddy viscosity; [`KOmegaSst::step`] advances the model.
* `UnstructuredMesh` — additive **unstructured finite-volume** path: triangular
  cells in 2-D (tetrahedra extend analogously), with a gradient/reconstruction
  helper ([`UnstructuredMesh::cell_gradient`]) and a diffusion + upwind
  advection residual ([`UnstructuredMesh::residual`]). [`UnstructuredMesh::solve_poisson`]
  converges to the analytic solution on a triangulated unit square.

The structured explicit scheme and algebraic turbulence model remain the
teaching / coupling primitive; the three additions above close the v1
"out of scope" items. The solver is not a production CFD code. It is the
foundation `tpt-sci-hemodynamics` and `tpt-sci-ocean` build on.

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

2-D, uniform collocated grid. The following are now implemented (v1):

* **Implicit pressure-correction** — `SimpleSolver` (SIMPLE/PISO-style) solves
  the pressure Poisson equation with the sparse conjugate-gradient solver and
  corrects the velocity to enforce continuity.
* **Two-equation `k`-`ω` SST turbulence** — `KOmegaSst` (Menter SST), running
  alongside the algebraic `turbulence::eddy_viscosity`.
* **Unstructured (triangular) finite-volume** — `UnstructuredMesh` with
  gradient reconstruction and a diffusion + upwind-advection residual assembly,
  alongside the structured `CollocatedGrid`.

Still out of scope for v1: 3-D tetrahedral assembly (the 2-D triangle code is
structured to extend analogously), moving `Left`/`Right` wall enforcement, and
parallel / production-grade solvers.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
