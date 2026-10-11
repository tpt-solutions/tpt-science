//! Tutorial: how Earth's oblateness and the atmosphere change a low orbit.
//!
//! A satellite in low Earth orbit (LEO) is perturbed by two effects of very
//! different size:
//!
//! * `J₂` (oblateness) makes the orbital plane precess. For an ISS-like orbit
//!   the node drifts by about −5° per day.
//! * Atmospheric drag lowers the semi-major axis slowly.
//!
//! The tutorial propagates a 400 km near-circular orbit for one day under each
//! effect and checks both magnitudes.
//!
//! Run with: `cargo run --release --example tutorial_orbit_j2_drag -p tpt-sci --features full`

use std::f64::consts::PI;

use tpt_sci::astro::{EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements};

const DAY: f64 = 86_400.0;

fn main() {
    // A 400 km near-circular orbit inclined at 51.6° (the ISS inclination).
    let a = EARTH_RADIUS_EQ + 400.0;
    let el = OrbitalElements::new(a, 0.0005, 51.6f64.to_radians(), 0.0, 0.0, 0.0, EARTH_MU)
        .expect("valid orbit");

    // Step 1 — the period, from Kepler's third law.
    println!("step 1: period = {:.1} min", el.period() / 60.0);

    // Step 2 — J₂ alone: node regression over one day.
    let after_j2 = el.propagate_j2(DAY, EARTH_J2, EARTH_RADIUS_EQ);
    let raan_drift_deg = (after_j2.raan - el.raan).to_degrees();
    // Wrap into (−180°, 180°] so the drift reads as a small signed number.
    let raan_drift_deg = ((raan_drift_deg + 180.0).rem_euclid(360.0)) - 180.0;
    println!("step 2: J₂ node drift over one day = {raan_drift_deg:.2}° (ISS: about −5°)");
    assert!(
        (raan_drift_deg + 5.0).abs() < 0.5,
        "J₂ node drift {raan_drift_deg} should be about −5° per day"
    );

    // Step 3 — drag alone: semi-major-axis decay over one day. `cd_a_over_m` is
    // C_D·A/m in the model's units; 0.02 is typical of a compact LEO spacecraft.
    let after_drag = el.propagate_drag(DAY, 0.02);
    let decay_km = el.a - after_drag.a;
    println!("step 3: drag lowers the semi-major axis by {decay_km:.4} km per day");
    assert!(decay_km > 0.0, "drag must lower the orbit");

    // Step 4 — the node change is a bounded secular term over one revolution.
    let node_change = (after_j2.raan - el.raan).rem_euclid(2.0 * PI);
    println!(
        "step 4: |ΔΩ| over the day = {:.2}° (a small secular change, not a runaway)",
        node_change.min(2.0 * PI - node_change).to_degrees()
    );
}
