# tpt-sci-cfd-core

Core **incompressible Navier–Stokes** finite-volume solver for the
`tpt-science` pillar, built from scratch (`pravash` audited and rejected:
GPL-3.0-only).

## What's here

- `CollocatedGrid` — uniform 2-D collocated grid.
- `Step` — fractional-step (Chorin) scheme: explicit advection–diffusion
  (`momentum`) + pressure-Poisson projection (`project`) to enforce `∇·u = 0`.
- `turbulence` — algebraic Smagorinsky eddy-viscosity estimate.

This crate is the foundation `tpt-sci-hemodynamics` and `tpt-sci-ocean` build
on. It is intentionally 2-D and explicit — a teaching / coupling primitive.

## Scope (v1)

2-D, uniform, explicit. Unstructured meshes, implicit / SIMPLE solvers, and a
full coupled `k`-`ω` SST two-equation turbulence model are out of scope for v1
(the `turbulence` module provides an algebraic eddy viscosity only).

Dual-licensed under MIT OR Apache-2.0.
