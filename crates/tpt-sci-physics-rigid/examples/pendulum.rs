//! # tpt-sci-physics-rigid: rigid physical pendulum
//!
//! A second, self-contained runnable example — distinct from `collision.rs`.
//! It drives a **rigid physical pendulum**: a spherical body of mass `m` and
//! radius `r` whose center of mass sits a distance `L` from a fixed pivot. The
//! only force is gravity, so the dynamics are pure rotational:
//!
//! * **Gravity torque about the pivot** — `τ_z = -m·g·L·sin θ` — is fed into
//!   [`Body::apply_torque`] (with the pivot inertia `I = I_cm + m·L²` set via
//!   [`Body::set_inertia`], parallel-axis theorem).
//! * **Orientation integration** — [`Body::spin`] advances the orientation
//!   quaternion under the updated angular velocity.
//! * **Swing angle & energy** — the angle `θ` is recovered from the orientation
//!   quaternion, the center-of-mass position is rebuilt from `θ`, and the total
//!   energy `E = m·g·L·(1 - cos θ) + ½·I·ω²` is bookkept to confirm the
//!   (frictionless) pendulum conserves energy.
//! * **Period** — zero-crossings of `θ` measure the observed period and compare
//!   it against the small-angle prediction `T = 2π·√(I / (m·g·L))`.
//!
//! Run with `cargo run --example pendulum -p tpt-sci-physics-rigid`. Every
//! assertion documents an invariant the crate guarantees (unit quaternion,
//! energy conservation, finite state). The pivot constraint itself is held
//! outside the crate (the crate models free rigid bodies, not joints), but the
//! rotational update uses *only* the public `Body` API.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::Body;

/// Convert an orientation quaternion `[w,x,y,z]` that encodes a rotation about
/// the `z` axis into the signed rotation angle (radians).
#[must_use]
fn angle_about_z(q: &[f64; 4]) -> f64 {
    2.0 * q[3].atan2(q[0])
}

/// Recover the center-of-mass position from the swing angle `theta`, given a
/// pivot and an arm length `L`. `theta = 0` means hanging straight down.
#[must_use]
fn com_position(pivot: &[f64; 2], l: f64, theta: f64) -> DVector<f64> {
    DVector::from_row_slice(&[pivot[0] + l * theta.sin(), pivot[1] - l * theta.cos()])
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

fn main() {
    println!("=== tpt-sci-physics-rigid: rigid physical pendulum ===\n");

    // ---------------------------------------------------------------------
    // Physical parameters.
    // ---------------------------------------------------------------------
    let m = 1.0_f64; // mass (kg)
    let r = 0.2_f64; // collision radius (m)
    let l = 2.0_f64; // pivot-to-center-of-mass arm length (m)
    let g = 9.8_f64; // gravitational acceleration (m/s^2)
    let pivot = [0.0_f64, 0.0_f64];

    let i_cm = 0.4 * m * r * r; // default isotropic sphere inertia
    let i_pivot = i_cm + m * l * l; // parallel-axis theorem
    let theta0 = 0.5_f64; // initial release angle from downward (rad)

    println!(
        "# parameters: m={m}, r={r}, L={l}, g={g}, I_cm={i_cm:.4}, I_pivot={i_pivot:.4}"
    );
    println!("  initial release angle θ0 = {theta0:.3} rad\n");

    // Build the body at the initial COM position and orientation.
    let mut theta = theta0;
    let pos0 = com_position(&pivot, l, theta);
    let half = theta0 / 2.0;
    let mut body = Body::new(
        0,
        pos0,
        DVector::from_row_slice(&[0.0, 0.0]),
        m,
        r,
    )
    .unwrap();
    body.set_inertia(i_pivot); // rotate about the pivot, not the COM
    body.set_angular_velocity([0.0, 0.0, 0.0]);
    body.set_orientation([half.cos(), 0.0, 0.0, half.sin()]); // θ about +z

    println!(
        "  initial COM = [{:.3}, {:.3}], q = [{:.3}, {:.3}, {:.3}, {:.3}]",
        body.position[0],
        body.position[1],
        body.orientation[0],
        body.orientation[1],
        body.orientation[2],
        body.orientation[3]
    );

    // ---------------------------------------------------------------------
    // Integrate the pendulum.
    // ---------------------------------------------------------------------
    let dt = 1e-3_f64;
    let steps = 6000_usize; // ~6 s of simulated time

    // Energy bookkeeping at t = 0.
    let energy = |theta: f64, omega: f64| m * g * l * (1.0 - theta.cos()) + 0.5 * i_pivot * omega * omega;
    let e0 = energy(theta0, 0.0);

    let mut prev_theta = theta;
    let mut crossing_times: Vec<f64> = Vec::new();
    let mut max_theta = 0.0_f64;
    let mut min_energy = e0;
    let mut max_energy = e0;
    let mut t = 0.0_f64;

    println!("\n# time integration (dt = {dt})");
    for step in 0..steps {
        // Current angle and gravity torque about the pivot.
        theta = angle_about_z(&body.orientation);
        let tau_z = -m * g * l * theta.sin();

        // Rigid-body rotational update via the public API.
        body.apply_torque([0.0, 0.0, tau_z], dt);
        body.spin(dt);

        // Recover the new angle and rebuild the COM position (pivot constraint
        // maintained externally; the crate integrates free rigid bodies).
        let new_theta = angle_about_z(&body.orientation);
        body.position = com_position(&pivot, l, new_theta);
        theta = new_theta;
        let omega = body.angular_velocity[2];

        t += dt;
        let e = energy(theta, omega);
        min_energy = min_energy.min(e);
        max_energy = max_energy.max(e);
        max_theta = max_theta.max(theta.abs());

        // Detect θ zero-crossings (swing through the bottom) to measure period.
        if prev_theta * theta < 0.0 {
            crossing_times.push(t);
        }
        prev_theta = theta;

        if step % 1000 == 0 || step == steps - 1 {
            println!(
                "  t={t:6.3}s  θ={theta:+7.4} rad  ω={omega:+7.4} rad/s  E={e:9.6} J"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Diagnostics & assertions.
    // ---------------------------------------------------------------------
    println!("\n# energy conservation (frictionless pendulum)");
    println!("  E0 = {e0:.6} J");
    println!(
        "  Emin = {min_energy:.6} J, Emax = {max_energy:.6} J, drift = {:.3e} J",
        (max_energy - min_energy).abs()
    );
    assert!(
        approx_eq(min_energy, max_energy, 1e-2),
        "total energy should be conserved to ~1e-2 J"
    );
    assert!((max_energy - e0).abs() <= 1e-2);

    println!("\n# orientation quaternion integrity");
    let q = body.orientation;
    let qnorm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();
    println!("  |q| = {qnorm:.9} (must stay unit under spin)");
    assert!(approx_eq(qnorm, 1.0, 1e-9));
    assert!(body.position.iter().all(|x| x.is_finite()));

    println!("\n# measured swing amplitude");
    println!("  max |θ| = {max_theta:.4} rad (released from {theta0:.4})");
    assert!(max_theta <= theta0 + 1e-2, "amplitude must not grow");

    // Observed period from consecutive zero-crossings of the same direction.
    if crossing_times.len() >= 2 {
        let observed = 2.0 * (crossing_times[1] - crossing_times[0]); // half-period -> full
        let t_small = 2.0 * std::f64::consts::PI * (i_pivot / (m * g * l)).sqrt();
        println!("\n# period");
        println!("  observed full period T ≈ {observed:.4} s ({}/2 crossings)", crossing_times.len());
        println!("  small-angle prediction T0 = {t_small:.4} s", );
        // For θ0 = 0.5 rad the true period exceeds the small-angle value by
        // ~1.6%; allow a generous tolerance.
        assert!(
            (observed - t_small).abs() <= 0.15 * t_small,
            "observed period should be close to the small-angle prediction"
        );
    } else {
        println!("\n# period: not enough zero-crossings captured (increase steps)");
    }

    println!("\nAll invariants held: unit quaternion, energy conserved, finite state.");
}
