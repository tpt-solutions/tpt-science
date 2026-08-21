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
* `Ocean3D` — 3-D z-level ocean core: a stack of `nz` layers with hydrostatic
  pressure from a linear equation of state `ρ = ρ0 − α·(T − T0) + β·(S − S0)`,
  prognostic temperature `T` / salinity `S` tracers, constant-coefficient
  vertical mixing, and a hydrostatic `step_3d`. The vertical velocity is not
  prognosed in hydrostatic mode. (`new`, `density`, `hydrostatic_pressure`,
  `mix_vertical`, `step_3d`).
* `Ocean3D::nonhydrostatic_correct` / `step_3d_nonhydrostatic` — optional
  non-hydrostatic pressure-correction: after the hydrostatic step a 3-D pressure
  Poisson equation `∇²φ = (∇·u*)/dt` is solved with
  `tpt-sci-grid::sparse::conjugate_gradient` on the structured grid and the
  provisional velocity is projected to be divergence-free.
* Data assimilation (`data_assim`): `nudge` relaxes a state toward sparse
  observations; `EnsembleKalmanFilter` is a stochastic EnKF; `Var3D` is a
  3D-Var-lite analysis with a background-error covariance.
* Reuses `tpt-sci-cfd-core::CollocatedGrid` and `tpt-sci-grid` (with the
  `sparse` feature) for discretization, so the same grid/convergence machinery
  carries over.
* `OceanError` — construction and linear-algebra errors for invalid model
  parameters.

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
seeds a gravity-wave bump and steps the basin forward. The `ocean3d` example
(`cargo run --example ocean3d -p tpt-sci-ocean`) tours the 3-D core, the
non-hydrostatic projection, and the data-assimilation schemes.

## Scope (v1)

2-D shallow-water circulation (geostrophic balance, gravity waves) plus a 3-D
z-level hydrostatic ocean core (density stratification, tracer transport, vertical
mixing), an optional non-hydrostatic pressure-correction projection, and a
nudging / EnKF / 3D-Var data-assimilation module. A full 3-D primitive-equation
ocean GCM with sigma/terrain-following coordinates and a complete assimilation
system remains out of scope for v1.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
