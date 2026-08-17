//! Two-body orbital propagation for a low-Earth orbit, plus the dominant J2
//! (oblateness) secular perturbation on the right ascension of the node.
//!
//! Run with: `cargo run --example leo_orbit -p tpt-sci-astro`

use tpt_sci_astro::{EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements};

fn main() {
    // A ~700 km-altitude circular LEO orbit, 50° inclination.
    let a = EARTH_RADIUS_EQ + 700.0; // km
    let el = OrbitalElements::new(a, 0.0, 50.0_f64.to_radians(), 0.0, 0.0, 0.0, EARTH_MU).unwrap();

    let (r0, v0) = el.state_vector();
    println!(
        "initial altitude = {:.1} km, speed = {:.3} km/s",
        r0.norm() - EARTH_RADIUS_EQ,
        v0.norm()
    );

    // Propagate one full orbit: a two-body orbit should return to its start.
    let period = el.period();
    let el_end = el.propagate(period);
    let (r1, _) = el_end.state_vector();
    let drift = (r1 - r0).norm();
    println!("after one period (two-body): position drift = {drift:.6e} km");

    // First-order secular J2: RAAN regression over one day.
    let (raan_dot, _argp_dot) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
    println!(
        "J2 RAAN rate = {:.6e} rad/s ({:.3} deg/day)",
        raan_dot,
        raan_dot.to_degrees() * 86_400.0
    );
    let el_j2 = el.propagate_j2(86_400.0, EARTH_J2, EARTH_RADIUS_EQ);
    println!("RAAN after 1 day (with J2) = {:.4} rad", el_j2.raan);

    // Sanity: the two-body drift should be within floating-point tolerance.
    assert!(drift < 1e-6);
}
