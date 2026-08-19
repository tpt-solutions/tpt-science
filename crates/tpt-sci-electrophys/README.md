# tpt-sci-electrophys

Cardiac **electrophysiology** for the `tpt-science` pillar, built on
[`tpt-sci-ode`](https://docs.rs/tpt-sci-ode) and
[`tpt-sci-grid`](https://docs.rs/tpt-sci-grid).

## What's here

- `HodgkinHuxley` — classic giant-axon model: gating `m, h, n` with
  voltage-dependent rate laws and ionic current `I_ion`.
- `Tissue` — 2-D monodomain sheet coupling the HH membrane to a 5-point
  Laplacian diffusion, so an action potential propagates.

## Scope (v1)

Single-cell HH + 2-D monodomain propagation. Full bidomain (intra/extra
split), ionic models beyond HH (e.g. Ten Tusscher), and anisotropy are out of
scope for v1.

Dual-licensed under MIT OR Apache-2.0.
