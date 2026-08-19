# tpt-sci-climate

Reduced-order **climate modelling** for the `tpt-science` pillar, built on
[`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).

## What's here

- `EnergyBalanceModel` — 0-D global energy balance with CO₂ forcing
  `ΔF = 5.35·ln(C/C0)`.
- `grey_radiative_transfer` — single-layer grey-atmosphere surface temperature.
- `ChemistryBox` — constant-production / first-order-loss tracer.

## Scope (v1)

0-D EBM, simple grey radiative transfer, single-tracer chemistry. GCMs, full
radiative-transfer bands, and 3-D atmospheric chemistry are out of scope for v1.

Dual-licensed under MIT OR Apache-2.0.
