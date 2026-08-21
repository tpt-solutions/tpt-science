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
* First-order J2 (oblateness) perturbation: `OrbitalElements::propagate_j2`
  propagates elements including secular nodal-regression / apsidal-precession
  drift, and `OrbitalElements::j2_secular_rates` returns the raw rates.
  Constants `EARTH_MU`, `EARTH_J2`, `EARTH_RADIUS_EQ` are provided for Earth.
* Combined J2 + J4 zonal-harmonic secular perturbation:
  `OrbitalElements::propagate_j4` / `OrbitalElements::j4_secular_rates` extend
  the J2 model with the next zonal term (`EARTH_J4`).
* Atmospheric drag: `atmospheric_density` (single-band exponential Earth
  atmosphere model) plus `OrbitalElements::drag_da_dt` /
  `OrbitalElements::propagate_drag` for the secular along-track decay of the
  semi-major axis (Vallado's standard averaged decay-rate formula).
* Simplified third-body perturbation: `OrbitalElements::third_body_secular_rates`
  / `OrbitalElements::propagate_third_body` give the leading secular
  (Kozai-Lidov, quadrupole-order) node/apsis/inclination/eccentricity drift
  from a Sun- or Moon-like perturber (`SUN_MU`, `MOON_MU`, `ASTRONOMICAL_UNIT_KM`,
  `MOON_DISTANCE_KM`), restricted to a perturber on a circular orbit lying in
  the reference plane.
* Solar radiation pressure: `srp_acceleration` / `OrbitalElements::srp_acceleration_vector`
  implement a cannonball SRP model (`F = P_srp · Cr · A/m`), automatically
  zeroed by the cylindrical Earth-shadow eclipse test `in_earth_shadow`.

All angles are in **radians**. The model assumes an ideal point-mass central
body for the pure two-body propagation, with each perturbation above modeled
as an independent first-order secular add-on (they are not combined into a
single integrated force model, and none captures short-periodic
oscillations). The third-body model is restricted to a circular, coplanar
perturber; the SRP model is a simple cannonball (no attitude-dependent
area or self-shadowing); the atmosphere is a single exponential band, not a
full reference atmosphere; and only J2/J4 are modeled, not the full zonal
harmonic series or tesseral/sectoral terms.

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
