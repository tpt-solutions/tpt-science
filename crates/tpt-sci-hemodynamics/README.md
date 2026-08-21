# tpt-sci-hemodynamics

1-D **compliant-vessel hemodynamics** for the `tpt-science` pillar, built on
[`tpt-sci-cfd-core`](https://docs.rs/tpt-sci-cfd-core) and
[`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).

## Features

* `Vessel` — 1-D compliant artery: cross-sectional area `A`, flow `Q`, linear
  tube-law pressure `p = β·(√A − √A0)`, and wave speed `c`.
* `tube_law_beta` — wall stiffness `β` from Young's modulus + thickness.
* `womersley_velocity` — analytic pulsatile (Womersley) profile amplitude
  (parabolic at low Womersley number, flattening toward plug flow as α grows).
* `casson_viscosity` — shear-thinning (non-Newtonian) Casson correction.
* `Network` — method-of-lines 1-D area/flow advance (`rhs`, `step`), integrating
  the augmented Navier–Stokes reduction by `tpt-sci-ode`.
* Exact Womersley solution — `womersley_velocity_profile` / `womersley_complex_velocity`
  evaluated via self-contained complex `J0`/`J1` (`bessel_j0`, `bessel_j1`), with
  the analytic flow-rate formula `womersley_flow_rate_analytic`.
* 0-D/1-D/3-D coupling — `Windkessel` (3-element RCR) terminal load, the `couple`
  step that drives a `Network` outlet from a `Windkessel`, and the `CfdCoupling`
  trait a `tpt-sci-cfd-core` 3-D domain can implement.

The model uses the 1-D augmented Navier–Stokes equations reduced to a vessel
centerline.

## Example

```rust
use tpt_sci_hemodynamics::{Vessel, tube_law_beta};

// Aortic-scale vessel, A0 = 1 cm², wall stiffness beta.
let beta = tube_law_beta(1.0e5, 1.0, 1.0);
let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
assert!(v.area > 0.0);
```

The `arterial_segment` example (`cargo run --example arterial_segment -p tpt-sci-hemodynamics`)
steps a compliant vessel network forward in time.

## Scope (v1)

1-D reduced-order vascular flow, the **exact** Womersley complex-Bessel velocity
profile (`womersley_velocity_profile` + self-contained `J0`/`J1`), and a
0-D/1-D/3-D coupling mechanism: a `Windkessel` (RCR) lumped outlet model driven
by / driving a `Network`, exposed through the `CfdCoupling` trait for a
`tpt-sci-cfd-core` 3-D domain. Patient-specific 3-D meshing and a full
multi-scale 3-D solver stay out of scope (repo-wide unstructured-FEM exclusion).

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
