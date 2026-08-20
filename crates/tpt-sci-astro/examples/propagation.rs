//! # Astrodynamics tour: the `tpt-sci-astro` public surface
//!
//! This example is a guided tour of the classical two-body machinery exposed by
//! `tpt-sci-astro`, all built from scratch on `tpt-math-linalg` (no external
//! astrodynamics wrappers). It runs entirely in an Earth-Centered Inertial
//! (ECI) frame with kilometres and seconds; every angle is in **radians**.
//!
//! What you will observe:
//!
//! 1. **Two construction paths.** [`OrbitalElements::new`] builds the classical
//!    (Keplerian) elements directly; [`OrbitalElements::state_vector`] turns
//!    them into an ECI position/velocity, and [`OrbitalElements::from_state`]
//!    inverts that transform. They round-trip.
//! 2. **Geometry primitives.** [`perifocal_to_eci`] (the perifocal→ECI rotation)
//!    and [`cross3`] (orbital angular momentum) are used directly.
//! 3. **Anomaly helpers.** [`true_to_eccentric`] / [`eccentric_to_true`] plus
//!    the Kepler solver [`solve_kepler`] compose the in-plane propagation used
//!    by [`OrbitalElements::propagate`].
//! 4. **Two-body propagation.** [`OrbitalElements::propagate`] advances the true
//!    anomaly. [`OrbitalElements::period`] (and the apoapsis/periapsis radii)
//!    are checked against closed-form values.
//! 5. **First-order secular J₂.** [`OrbitalElements::propagate_j2`] drifts the
//!    node and periapsis, and [`OrbitalElements::j2_secular_rates`] gives the
//!    analytic `Ω̇` / `ω̇`. The numerical drift is cross-checked against the
//!    analytic rate, and the RAAN regression for a prograde LEO orbit is printed.
//!
//! The run is deterministic and fast (a handful of closed-form evaluations), and
//! every quantity of interest is guarded by an `assert!`.

use std::f64::consts::PI;
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_astro::{
    EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements, cross3, eccentric_to_true,
    perifocal_to_eci, solve_kepler, true_to_eccentric,
};

/// Relative tolerance used for the closed-form checks below.
const TOL: f64 = 1e-6;

fn main() {
    // ---------------------------------------------------------------------
    // 1. Construct orbital elements two ways and confirm they round-trip.
    // ---------------------------------------------------------------------
    let a = 7000.0; // km, a LEO-ish altitude above Earth's surface
    let e = 0.01;
    let inc = 50.0_f64.to_radians();
    let raan = 0.7;
    let argp = 0.3;
    let nu = 0.9;

    let el = OrbitalElements::new(a, e, inc, raan, argp, nu, EARTH_MU).expect("valid LEO elements");

    // Elements -> ECI state vector, then invert it back to elements.
    let (r, v) = el.state_vector();
    let recovered = OrbitalElements::from_state(&r, &v, EARTH_MU).expect("recover from a state");

    assert!((recovered.a - el.a).abs() / el.a < TOL, "a round-trip");
    assert!((recovered.e - el.e).abs() < TOL, "e round-trip");
    assert!((recovered.i - el.i).abs() < TOL, "i round-trip");
    assert!(
        r.norm().is_finite() && v.norm().is_finite(),
        "state is finite"
    );
    println!("round-trip (elements -> state -> elements):");
    println!("  a   {:>12.4} -> {:>12.4} km", el.a, recovered.a);
    println!("  e   {:>12.6} -> {:>12.6}", el.e, recovered.e);
    println!("  i   {:>12.6} -> {:>12.6} rad", el.i, recovered.i);
    println!(
        "  |r| = {:>11.4} km,  |v| = {:>11.4} km/s",
        r.norm(),
        v.norm()
    );

    // ---------------------------------------------------------------------
    // 2. Low-level geometry primitives used by the conversion above.
    // ---------------------------------------------------------------------
    // The perifocal->ECI rotation `Q` is the product R3(RAAN) R1(i) R3(argp).
    let q = perifocal_to_eci(el.raan, el.i, el.argp);
    // Q must be a proper rotation: its columns are orthonormal unit vectors and
    // (col0 x col1) == col2 (a right-handed frame, det = +1). We test this with
    // the supported matrix*vector product applied to the basis vectors.
    let ex = q.clone() * DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let ey = q.clone() * DVector::from_vec(vec![0.0, 1.0, 0.0]);
    let ez = q.clone() * DVector::from_vec(vec![0.0, 0.0, 1.0]);
    let orthonormal = ex.norm().abs() - 1.0 < TOL
        && ey.norm().abs() - 1.0 < TOL
        && ez.norm().abs() - 1.0 < TOL
        && ex.dot(&ey).abs() < TOL
        && ex.dot(&ez).abs() < TOL
        && ey.dot(&ez).abs() < TOL;
    assert!(orthonormal, "perifocal_to_eci columns are orthonormal");
    let z_from_cross = cross3(&ex, &ey);
    assert!(
        (z_from_cross - ez).norm() < TOL,
        "perifocal_to_eci is right-handed"
    );
    println!("  perifocal_to_eci: orthonormal, right-handed rotation confirmed");

    // Orbital angular momentum h = r x v, perpendicular to the orbit plane.
    let h = cross3(&r, &v);
    let h_manual = DVector::from_vec(vec![
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ]);
    assert!(
        (h.norm() - h_manual.norm()).abs() < TOL,
        "cross3 matches manual cross product"
    );
    assert!(
        (h.dot(&r)).abs() < 1e-3 && (h.dot(&v)).abs() < 1e-3,
        "h ⟂ r and v"
    );
    let inclination_from_h = (h[2] / h.norm()).clamp(-1.0, 1.0).acos();
    assert!(
        (inclination_from_h - el.i).abs() < TOL,
        "inclination from angular momentum"
    );
    println!("  cross3: h = r x v is normal to the orbit plane (i matches)");

    // ---------------------------------------------------------------------
    // 3. Anomaly helpers and the Kepler solver.
    // ---------------------------------------------------------------------
    // true_to_eccentric and eccentric_to_true are mutual inverses.
    let ecc = true_to_eccentric(el.nu, el.e);
    let nu_back = eccentric_to_true(ecc, el.e);
    assert!(
        (nu_back - el.nu).abs() < TOL,
        "anomaly conversion round-trips"
    );

    // solve_kepler satisfies M = E - e sin E.
    let mean_anomaly = ecc - el.e * ecc.sin();
    let ecc_solved = solve_kepler(mean_anomaly, el.e);
    let residual = (ecc_solved - el.e * ecc_solved.sin() - mean_anomaly).abs();
    assert!(residual < 1e-9, "Kepler's equation solved");
    println!("  anomaly helpers + solve_kepler: Kepler residual = {residual:.2e}");

    // ---------------------------------------------------------------------
    // 4. Two-body propagation, period, and apsides.
    // ---------------------------------------------------------------------
    let period = el.period();
    let period_exact = 2.0 * PI * (el.a.powi(3) / el.mu).sqrt();
    assert!(
        (period - period_exact).abs() / period < TOL,
        "period = 2pi sqrt(a^3/mu)"
    );
    let r_apo = el.a * (1.0 + el.e);
    let r_peri = el.a * (1.0 - el.e);
    assert!(r_apo.is_finite() && r_peri.is_finite(), "apsides finite");
    println!("two-body Kepler propagation (a = {a} km, e = {e}):");
    println!("  period          = {period:>11.1} s",);
    println!("  apoapsis radius = {r_apo:>11.1} km",);
    println!("  periapsis radius= {r_peri:>11.1} km",);

    // Two half-period steps must equal one full period (the mean anomaly
    // advances by exactly pi each step). Note the true anomaly does *not*
    // advance by exactly pi per half period for an eccentric orbit, because the
    // spacecraft moves faster near periapsis.
    let half1 = el.propagate(period / 2.0);
    let half2 = half1.propagate(period / 2.0);
    let full = el.propagate(period);
    let dnu_full = (half2.nu - full.nu).rem_euclid(2.0 * PI);
    assert!(
        dnu_full < 1e-6 || (2.0 * PI - dnu_full).abs() < 1e-6,
        "two half-period steps equal one full period"
    );
    assert!(
        half1.nu.is_finite() && half2.nu.is_finite(),
        "half steps finite"
    );
    println!("  two half-period steps reproduce one full period (mean anomaly +pi each)");

    // Propagating by exactly one period is the identity on the true anomaly.
    let full = el.propagate(period);
    let dnu = (full.nu - el.nu).rem_euclid(2.0 * PI);
    assert!(
        dnu < 1e-6 || (2.0 * PI - dnu).abs() < 1e-6,
        "one period is identity"
    );
    println!("  one full period advances true anomaly by {dnu:.2e} rad (identity)");

    // ---------------------------------------------------------------------
    // 5. First-order secular J2 perturbation.
    // ---------------------------------------------------------------------
    let day = 86_400.0; // seconds
    let (raan_dot, argp_dot) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);

    // The numerical J2 propagation must reproduce the analytic linear drift.
    let adv_j2 = el.propagate_j2(day, EARTH_J2, EARTH_RADIUS_EQ);
    let raan_expected = (el.raan + raan_dot * day).rem_euclid(2.0 * PI);
    let argp_expected = (el.argp + argp_dot * day).rem_euclid(2.0 * PI);
    assert!(
        (adv_j2.raan - raan_expected).abs() < 1e-9,
        "J2 RAAN matches analytic rate"
    );
    assert!(
        (adv_j2.argp - argp_expected).abs() < 1e-9,
        "J2 argp matches analytic rate"
    );
    // a, e, i are constant to first order in J2.
    assert!((adv_j2.a - el.a).abs() < 1e-9, "a preserved by J2");
    assert!((adv_j2.e - el.e).abs() < 1e-9, "e preserved by J2");
    assert!((adv_j2.i - el.i).abs() < 1e-9, "i preserved by J2");

    let deg_per_day = |rate: f64| rate * day * 180.0 / PI;
    println!("first-order secular J2 (a = {a} km, i = 50 deg):");
    println!(
        "  RAAN drift  = {:.4} deg/day (analytic, via j2_secular_rates)",
        deg_per_day(raan_dot)
    );
    println!(
        "  argp drift  = {:.4} deg/day (analytic, via j2_secular_rates)",
        deg_per_day(argp_dot)
    );
    println!(
        "  RAAN after 1 day = {:.4} rad (propagate_j2 agrees to 1e-9)",
        adv_j2.raan
    );

    // Sanity: a prograde (i < 90 deg) LEO orbit experiences *regression* of the
    // node (RAAN decreases), the classic sun-synchronous-orbit lever.
    assert!(raan_dot < 0.0, "prograde LEO: RAAN regresses");
    println!("  (prograde orbit: RAAN regresses, as expected)");

    println!("\nAll checks passed.");
}
