//! # lambert_transfer.rs — orbital-transfer analysis with the two-body API
//!
//! `tpt-sci-astro` implements the classical Kepler two-body problem: validated
//! [`OrbitalElements`], elements↔state-vector conversion, two-body
//! propagation, and first-order secular `J₂` drift. It does **not** ship a
//! general Lambert (two-point boundary value) solver, so this example performs
//! the closest genuinely-supported transfer computation: a coplanar
//! circular-to-circular **Hohmann transfer** between two Earth orbits.
//!
//! This exercises only real public functions:
//!
//! 1. The transfer ellipse is built directly as [`OrbitalElements`] from the two
//!    circular radii (`a = (r₁+r₂)/2`, `e = (r₂−r₁)/(r₂+r₁)`), and its
//!    periapsis/apoapsis radii are verified against the endpoints via
//!    [`OrbitalElements::state_vector`] and a half-period
//!    [`OrbitalElements::propagate`].
//! 2. The elements are recovered from a propagated state with
//!    [`OrbitalElements::from_state`] (round-trip self-check).
//! 3. The two impulse Δv's are computed with the vis-viva relation and
//!    cross-checked against the speeds read off the transfer orbit's state
//!    vectors.
//! 4. The `J₂` nodal drift of the low and high circular orbits is compared via
//!    [`OrbitalElements::j2_secular_rates`], showing why transfer timing
//!    between inclined altitudes is sensitive to oblateness.
//!
//! Run with: `cargo run --example lambert_transfer -p tpt-sci-astro`

use std::f64::consts::PI;
use tpt_sci_astro::{EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements};

/// Relative tolerance for the closed-form checks below.
const TOL: f64 = 1e-6;

fn main() {
    // A LEO parking orbit -> GEO transfer around Earth.
    let r1 = 7000.0; // km, LEO circular radius
    let r2 = 42164.0; // km, GEO circular radius
    let a_t = 0.5 * (r1 + r2); // transfer semi-major axis
    let e_t = (r2 - r1) / (r2 + r1); // transfer eccentricity

    println!("=== Hohmann transfer (LEO r1={r1} km -> GEO r2={r2} km) ===\n");
    println!("Transfer ellipse: a = {a_t:.1} km, e = {e_t:.4}",);

    // ---------------------------------------------------------------------
    // 1. Build the transfer orbit and confirm its endpoints line up.
    // ---------------------------------------------------------------------
    let transfer =
        OrbitalElements::new(a_t, e_t, 0.0, 0.0, 0.0, 0.0, EARTH_MU).expect("valid transfer orbit");

    // At nu = 0 the state vector sits at periapsis: r = a(1 - e) = r1.
    let (r_peri, v_peri_vec) = transfer.state_vector();
    assert!(
        (r_peri.norm() - r1).abs() / r1 < TOL,
        "transfer periapsis must equal r1"
    );

    // After half a period the true anomaly reaches pi -> apoapsis: r = r2.
    let half = transfer.propagate(transfer.period() / 2.0);
    let (r_apo, v_apo_vec) = half.state_vector();
    assert!(
        (r_apo.norm() - r2).abs() / r2 < TOL,
        "transfer apoapsis after half period must equal r2"
    );

    // Elements -> state -> elements must round-trip.
    let (r0, v0) = transfer.state_vector();
    let recovered =
        OrbitalElements::from_state(&r0, &v0, EARTH_MU).expect("recover transfer from state");
    assert!((recovered.a - a_t).abs() / a_t < TOL, "a round-trips");
    assert!((recovered.e - e_t).abs() < TOL, "e round-trips");

    println!("Geometry (state-vector + propagation):");
    println!("  periapsis r = {:>11.3} km (expect {r1})", r_peri.norm());
    println!("  apoapsis  r = {:>11.3} km (expect {r2})", r_apo.norm());
    println!(
        "  transfer period T = {:>11.1} s (expect {:.1})",
        transfer.period(),
        2.0 * PI * (a_t.powi(3) / EARTH_MU).sqrt()
    );

    // ---------------------------------------------------------------------
    // 2. Impulse budget via vis-viva, cross-checked against state speeds.
    // ---------------------------------------------------------------------
    let v_circ1 = (EARTH_MU / r1).sqrt();
    let v_circ2 = (EARTH_MU / r2).sqrt();
    let v_peri = (EARTH_MU * (2.0 / r1 - 1.0 / a_t)).sqrt();
    let v_apo = (EARTH_MU * (2.0 / r2 - 1.0 / a_t)).sqrt();
    let dv1 = v_peri - v_circ1; // burn at LEO to enter transfer
    let dv2 = v_circ2 - v_apo; // burn at GEO to circularise
    let dv_total = dv1 + dv2;

    // The state-vector speeds must agree with the vis-viva prediction.
    assert!(
        (v_peri_vec.norm() - v_peri).abs() < 1e-3,
        "v_peri from state vector matches vis-viva"
    );
    assert!(
        (v_apo_vec.norm() - v_apo).abs() < 1e-3,
        "v_apo from state vector matches vis-viva"
    );

    println!("\nImpulse budget (vis-viva, cross-checked vs state speeds):");
    println!("  v_circ(LEO)     = {:>10.3} km/s", v_circ1);
    println!(
        "  v_transfer(peri)= {:>10.3} km/s   -> Δv1 = {:>8.3} km/s",
        v_peri, dv1
    );
    println!("  v_circ(GEO)     = {:>10.3} km/s", v_circ2);
    println!(
        "  v_transfer(apo) = {:>10.3} km/s   -> Δv2 = {:>8.3} km/s",
        v_apo, dv2
    );
    println!("  total Δv        = {:>10.3} km/s", dv_total);
    // Sanity: the canonical LEO->GEO Hohmann total is ~3.9 km/s.
    assert!(
        dv_total > 3.5 && dv_total < 4.5,
        "total Δv in expected band"
    );

    // ---------------------------------------------------------------------
    // 3. J2 secular nodal drift of the two endpoint circular orbits.
    // ---------------------------------------------------------------------
    let inc = 50.0_f64.to_radians();
    let low = OrbitalElements::new(r1, 0.0, inc, 0.0, 0.0, 0.0, EARTH_MU).expect("LEO circular");
    let high = OrbitalElements::new(r2, 0.0, inc, 0.0, 0.0, 0.0, EARTH_MU).expect("GEO circular");
    let (raan_dot_low, _) = low.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
    let (raan_dot_high, _) = high.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
    // Lower orbit (smaller p = a(1-e^2)) drifts faster: rate ∝ (r_eq/p)^2.
    assert!(
        raan_dot_low.abs() > raan_dot_high.abs(),
        "lower orbit must regress faster"
    );

    let deg_day = |rate: f64| rate * 86_400.0 * 180.0 / PI;
    println!("\nJ2 nodal drift of endpoints (i = 50 deg, via j2_secular_rates):");
    println!(
        "  LEO (r={r1}) RAAN drift  = {:>9.4} deg/day",
        deg_day(raan_dot_low)
    );
    println!(
        "  GEO (r={r2}) RAAN drift  = {:>9.4} deg/day",
        deg_day(raan_dot_high)
    );
    println!("  (lower orbit regresses faster — a real transfer must account for it)");

    println!("\nAll transfer checks passed.");
}
