//! Public-API invariants for two-body orbital mechanics.

use tpt_sci_astro::{EARTH_MU, OrbitalElements};

fn leo() -> OrbitalElements {
    // 7000 km semi-major axis, slightly eccentric, inclined.
    OrbitalElements::new(7000.0, 0.01, 0.9, 0.4, 0.3, 0.2, EARTH_MU).unwrap()
}

#[test]
fn propagating_one_period_returns_to_the_same_state() {
    let el = leo();
    let (r0, v0) = el.state_vector();
    let (r1, v1) = el.propagate(el.period()).state_vector();
    assert!(
        (r1.clone() - r0.clone()).norm() < 1e-6 * r0.norm(),
        "position drift"
    );
    assert!(
        (v1.clone() - v0.clone()).norm() < 1e-9 * v0.norm(),
        "velocity drift"
    );
}

#[test]
fn state_vector_round_trips_through_elements() {
    let el = leo();
    let (r, v) = el.state_vector();
    let back = OrbitalElements::from_state(&r, &v, EARTH_MU).unwrap();
    assert!((back.a - el.a).abs() < 1e-6 * el.a);
    assert!((back.e - el.e).abs() < 1e-9);
}

#[test]
fn vis_viva_energy_matches_semi_major_axis() {
    let el = leo();
    let (r, v) = el.state_vector();
    let energy = 0.5 * v.norm().powi(2) - EARTH_MU / r.norm();
    let expected = -EARTH_MU / (2.0 * el.a);
    assert!((energy - expected).abs() < 1e-9 * expected.abs());
}
