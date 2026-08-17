//! Two-body Kepler propagation and first-order secular J2 nodal regression.
use tpt_sci_astro::{EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements};

fn main() {
    let el =
        OrbitalElements::new(7000.0, 0.01, 50.0_f64.to_radians(), 0.0, 0.0, 0.0, EARTH_MU).unwrap();
    let period = el.period();
    println!("Orbital period: {period:.1} s");

    let half = el.propagate(period / 2.0);
    let (r0, _) = el.state_vector();
    let (r1, _) = half.state_vector();
    println!(
        "After half an orbit, radius {:.4} -> {:.4} (should be ~equal)",
        r0.norm(),
        r1.norm()
    );

    let day = 86_400.0;
    let adv = el.propagate_j2(day, EARTH_J2, EARTH_RADIUS_EQ);
    let (raan_dot, _) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
    println!(
        "RAAN after 1 day: {:.4} rad; drift {:.4} deg/day (J2 regression)",
        adv.raan,
        raan_dot * day * 180.0 / std::f64::consts::PI
    );
}
