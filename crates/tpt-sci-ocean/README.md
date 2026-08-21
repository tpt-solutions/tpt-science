# tpt-sci-ocean

2-D **shallow-water / primitive-equation** ocean circulation for the
`tpt-science` pillar, built on
[`tpt-sci-cfd-core`](https://docs.rs/tpt-sci-cfd-core) (`pravash` was audited and
rejected: GPL-3.0-only, same as CFD).

## Features

* `ShallowWater` — 2-D shallow-water model: free-surface height `h` +
  depth-averaged `u`/`v` on a uniform grid. Integrates the continuity and
  momentum equations with gravity and Coriolis `f` (`new`, `perturb_center`,
  `step`, `max_speed`).
* Reuses `tpt-sci-cfd-core::CollocatedGrid` for discretization, so the same
  grid/convergence machinery carries over.
* `OceanError` — construction errors for invalid model parameters.

This is a reduced-order circulation primitive — not a full 3-D primitive-equation
ocean GCM.

## Example

```rust
use tpt_sci_ocean::ShallowWater;

// 64×64 basin, gravity g = 9.81, Coriolis f = 1e-4.
let mut sw = ShallowWater::new(64, 64, 1.0, 1.0, 9.81, 1e-4, 0.001);
// Perturb the free surface; it should evolve without blowing up.
sw.perturb_center(1.0);
for _ in 0..10 {
    sw.step(0.001);
}
assert!(sw.max_speed().is_finite());
```

The `shallow_water` example (`cargo run --example shallow_water -p tpt-sci-ocean`)
seeds a gravity-wave bump and steps the basin forward.

## Scope (v1)

2-D shallow-water circulation (geostrophic balance, gravity waves). Full 3-D
primitive-equation ocean GCM, hydrostatic/non-hydrostatic vertical coordinates,
and data assimilation are out of scope for v1.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
