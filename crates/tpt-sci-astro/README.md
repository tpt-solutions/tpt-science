# tpt-sci-astro

Orbital-mechanics and coordinate-frame primitives for the `tpt-science` pillar,
built entirely from scratch on top of the in-house `tpt-math-linalg` dense
linear algebra (no external astrodynamics or geometry wrappers).

The crate implements the **classical two-body problem** in an Earth-Centered
Inertial (ECI) reference frame:

* Classical (Keplerian) `OrbitalElements`, validated on construction.
* Conversion between Keplerian elements and ECI Cartesian state vectors
  (`OrbitalElements::state_vector` and `OrbitalElements::from_state`).
* Time propagation via Kepler's equation
  (`state -> mean anomaly -> advance -> solve -> true anomaly`) in
  `OrbitalElements::propagate`.

All angles are in **radians**. The model assumes an ideal point-mass central
body and neglects perturbations (J2, drag, third bodies, SRP, ...), so it is
exact only for the two-body problem.

Depends on `tpt-math-linalg` (published).

## Example

```rust
use tpt_math_linalg::tpt_math_linalg_dense::{DVector, DMatrix};
use tpt_sci_astro::OrbitalElements;

// A unit circular orbit about a unit-mass body.
let el = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
let (r, _v) = el.state_vector();
assert!((r.norm() - 1.0).abs() < 1e-9);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
