# tpt-sci-hemodynamics

1-D **compliant-vessel hemodynamics** for the `tpt-science` pillar, built on
[`tpt-sci-cfd-core`](https://docs.rs/tpt-sci-cfd-core) and
[`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).

## What's here

- `Vessel` — 1-D compliant artery: cross-sectional area, flow, linear tube-law
  pressure `p = β·(√A − √A0)`, wave speed `c`.
- `tube_law_beta` — wall stiffness from Young's modulus + thickness.
- `womersley_velocity` — analytic pulsatile (Womersley) profile amplitude.
- `casson_viscosity` — shear-thinning (non-Newtonian) correction.
- `Network` — method-of-lines 1-D area/flow advance.

## Scope (v1)

1-D reduced-order vascular flow. 3-D patient-specific, full 0-D/1-D/3-D
coupling, and a real Womersley complex Bessel solve are out of scope for v1.

Dual-licensed under MIT OR Apache-2.0.
