# tpt-sci-ocean

2-D **shallow-water / primitive-equation** ocean circulation for the
`tpt-science` pillar, built on
[`tpt-sci-cfd-core`](https://docs.rs/tpt-sci-cfd-core) (`pravash` audited and
rejected: GPL-3.0-only, same as CFD).

## What's here

- `ShallowWater` — 2-D shallow-water model: height + depth-averaged `u`/`v`,
  continuity + momentum with gravity and Coriolis `f`, on a uniform grid.

## Scope (v1)

2-D shallow-water circulation (geostrophic balance, gravity waves). Full 3-D
primitive-equation ocean GCM, hydrostatic/non-hydrostatic vertical coordinates,
and data assimilation are out of scope for v1.

Dual-licensed under MIT OR Apache-2.0.
