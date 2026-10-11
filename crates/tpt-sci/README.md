# tpt-sci

Umbrella crate for the [tpt-science](https://github.com/tpt-solutions/tpt-science)
simulation substrate. Depend on this one crate and reach every domain crate
through a short module path:

```rust
use tpt_sci::ode::{Method, OdeProblem};

// dy/dt = -y  ->  y(1) = e^{-1}
let prob = OdeProblem::new(|_t, y, dy| dy[0] = -y[0], vec![1.0], 0.0).unwrap();
let y = prob.solve(Method::Tsit45, 1.0).unwrap();
assert!((y[0] - (-1.0f64).exp()).abs() < 1e-5);
```

## Features

Each domain is a Cargo feature named after its crate (`ode`, `grid`,
`sim-core`, `reaction-network`, `ppl`, `image`, `physics-rigid`, `quantum`,
`astro`, `md`, `dft-classical`, `kinetics`, `cfd-core`, `hemodynamics`,
`electrophys`, `climate`, `ocean`, `dft-electronic`).

- Default: `ode`, `grid`, `sim-core`.
- `full`: every domain. Note that `dft-classical` pulls in the `feos` stack, so
  it is the slowest to compile; leave it out unless you need it.

The module map (module name → crate) is in the crate documentation.

## Scope

This crate contains no simulation code of its own. It only re-exports the
`tpt-sci-*` crates, so their APIs, versioning, and feature flags are defined in
those crates.
