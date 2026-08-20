//! # tpt-sci-physics-rigid tour
//!
//! A runnable tour of the `tpt-sci-physics-rigid` public surface. It exercises:
//!
//! * **World construction & environment** — [`World::new`],
//!   [`World::with_gravity`], [`World::set_bounds`], [`World::set_restitution`].
//! * **Body construction & validation** — [`Body::new`] (returns a
//!   [`PhysicsError`] on bad input) plus the `position`/`velocity`/`mass`/
//!   `radius`/`inertia`/`orientation` fields.
//! * **Linear (point-mass) dynamics** — semi-implicit Euler integration under
//!   gravity, a perfectly elastic wall bounce, and a head-on elastic
//!   body-body collision with momentum/energy bookkeeping.
//! * **Rigid-body rotation** — [`Body::set_angular_velocity`],
//!   [`Body::apply_torque`], [`Body::spin`] (quaternion kinematics),
//!   [`Body::set_orientation`], [`Body::set_inertia`], and
//!   [`Body::orientation_matrix`].
//! * **Quaternion helpers** — free functions [`quat_to_matrix`],
//!   [`quat_mul`], and [`quat_normalize`].
//!
//! Everything is deterministic and fast (a handful of `step`s). Run it with
//! `cargo run --example collision -p tpt-sci-physics-rigid` and watch the
//! labeled diagnostics; every assertion documents an invariant the crate
//! guarantees.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::{Body, PhysicsError, World, quat_mul, quat_normalize, quat_to_matrix};

/// Total linear momentum of the world (Σ m·v). Note: the integrator's elastic
/// collisions conserve this exactly.
fn total_momentum(world: &World) -> DVector<f64> {
    let dim = world.bodies().first().map_or(2, |b| b.position.len());
    world.bodies().iter().fold(DVector::zeros(dim), |acc, b| {
        acc + b.velocity.clone() * b.mass
    })
}

/// Total translational kinetic energy of the world (Σ ½·m·v²).
fn total_kinetic_energy(world: &World) -> f64 {
    world
        .bodies()
        .iter()
        .map(|b| 0.5 * b.mass * b.velocity.dot(&b.velocity))
        .sum()
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

fn assert_finite_vec(v: &DVector<f64>) {
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite vector component detected"
    );
}

fn main() {
    println!("=== tpt-sci-physics-rigid: rigid-body surface tour ===\n");

    // ---------------------------------------------------------------------
    // 1. Linear motion under gravity + bounding walls + restitution.
    // ---------------------------------------------------------------------
    println!("# 1. Gravity, bounding box, and elastic wall bounce");
    let mut world = World::with_gravity(DVector::from_row_slice(&[0.0, -9.8]));
    world.set_bounds(DVector::from_row_slice(&[5.0, 5.0]));
    world.set_restitution(1.0);
    world
        .add_body(
            Body::new(
                0,
                DVector::from_row_slice(&[0.0, 4.0]),
                DVector::from_row_slice(&[0.0, 0.0]),
                1.0,
                0.5,
            )
            .unwrap(),
        )
        .unwrap();

    // Free fall for dt = 0.5 s: Δy = ½·g·t² (exact for constant acceleration).
    let y0 = world.body(0).unwrap().position[1];
    world.step(0.5);
    let b = world.body(0).unwrap();
    let y1 = b.position[1];
    println!(
        "  free fall: y {y0:.3} -> {y1:.3} (Δy = {:.3}, expected {:.3})",
        y1 - y0,
        -0.5 * 9.8 * 0.25
    );
    assert!(approx_eq(y1, 4.0 - 0.5 * 9.8 * 0.25, 1e-9));
    assert_finite_vec(&b.position);
    assert_finite_vec(&b.velocity);

    // Drive the body into the +x wall and confirm a perfect reflection.
    let wall = world.body_mut(0).unwrap();
    wall.position = DVector::from_row_slice(&[4.8, 0.0]);
    wall.velocity = DVector::from_row_slice(&[1.0, 0.0]);
    world.step(0.5);
    let b = world.body(0).unwrap();
    println!(
        "  wall bounce: vx +1.000 -> {:.3} (reflected, restitution = 1.0)",
        -b.velocity[0]
    );
    assert!(b.velocity[0] < 0.0, "wall should reflect +x velocity");
    assert!(approx_eq(b.velocity[0], -1.0, 1e-9));

    // `add_body` rejects duplicate ids (this also demonstrates the error type).
    let dup = Body::new(
        0,
        DVector::from_row_slice(&[0.0, 0.0]),
        DVector::from_row_slice(&[0.0, 0.0]),
        1.0,
        0.5,
    )
    .unwrap();
    let err = world.add_body(dup).unwrap_err();
    assert!(matches!(err, PhysicsError::DuplicateId(0)));
    println!("  duplicate-id rejected as {err}");

    // ---------------------------------------------------------------------
    // 2. Head-on elastic collision: momentum & energy bookkeeping.
    // ---------------------------------------------------------------------
    println!("\n# 2. Elastic body-body collision (momentum & energy)");
    let mut world = World::new();
    world.set_restitution(1.0);
    world
        .add_body(
            Body::new(
                0,
                DVector::from_row_slice(&[0.0, 0.0]),
                DVector::from_row_slice(&[1.0, 0.0]),
                1.0,
                0.5,
            )
            .unwrap(),
        )
        .unwrap();
    world
        .add_body(
            Body::new(
                1,
                DVector::from_row_slice(&[1.0, 0.0]),
                DVector::from_row_slice(&[0.0, 0.0]),
                1.0,
                0.5,
            )
            .unwrap(),
        )
        .unwrap();

    let p_before = total_momentum(&world);
    let ke_before = total_kinetic_energy(&world);
    println!(
        "  before: p = [{:.3}, {:.3}], KE = {:.3}",
        p_before[0], p_before[1], ke_before
    );

    world.step(0.5);

    let p_after = total_momentum(&world);
    let ke_after = total_kinetic_energy(&world);
    println!(
        "  after : p = [{:.3}, {:.3}], KE = {:.3}",
        p_after[0], p_after[1], ke_after
    );

    // Equal masses, head-on, perfectly elastic => velocities exchange.
    let a = world.body(0).unwrap();
    let c = world.body(1).unwrap();
    println!(
        "  velocities exchanged: v0 = {:.3}, v1 = {:.3}",
        a.velocity[0], c.velocity[0]
    );
    assert!(approx_eq(a.velocity[0], 0.0, 1e-6));
    assert!(approx_eq(c.velocity[0], 1.0, 1e-6));
    // Momentum and kinetic energy are conserved to machine precision.
    assert!(approx_eq(p_after[0], p_before[0], 1e-9));
    assert!(approx_eq(p_after[1], p_before[1], 1e-9));
    assert!(approx_eq(ke_after, ke_before, 1e-9));
    for body in world.bodies() {
        assert_finite_vec(&body.position);
        assert_finite_vec(&body.velocity);
    }

    // ---------------------------------------------------------------------
    // 3. Rigid-body rotation: torque -> angular velocity -> spin -> matrix.
    // ---------------------------------------------------------------------
    println!("\n# 3. Rotational dynamics (torque, spin, orientation matrix)");
    let mut spinner = Body::new(
        0,
        DVector::from_row_slice(&[0.0, 0.0]),
        DVector::from_row_slice(&[0.0, 0.0]),
        1.0,
        0.5,
    )
    .unwrap();
    println!(
        "  default isotropic inertia (2/5·m·r²) = {:.3}",
        spinner.inertia
    );
    assert!(approx_eq(spinner.inertia, 0.4 * 1.0 * 0.25, 1e-12));

    // Override the inertia to make the arithmetic obvious, then apply a torque.
    spinner.set_inertia(0.1); // 0.4·m·r² for m=1, r=0.5 would be 0.1 anyway.
    spinner.set_angular_velocity([0.0, 0.0, 2.0]);
    spinner.apply_torque([0.0, 0.0, 1.0], 0.1); // ω += τ·dt / I = 1·0.1/0.1 = 1.0
    println!(
        "  after τ=(0,0,1)·dt=0.1 with I={:.2}: ω_z = {:.3} (was 2.0)",
        spinner.inertia, spinner.angular_velocity[2]
    );
    assert!(approx_eq(spinner.angular_velocity[2], 3.0, 1e-12));

    // Integrate orientation forward by a quarter turn at ω_z = 3 rad/s.
    let q0 = spinner.orientation;
    spinner.spin(0.25); // θ ≈ 3·0.25 = 0.75 rad about z (small-step integration)
    let q1 = spinner.orientation;
    println!(
        "  orientation quaternion: [{:.3}, {:.3}, {:.3}, {:.3}] -> [{:.3}, {:.3}, {:.3}, {:.3}]",
        q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]
    );
    // The quaternion must stay (approximately) unit under `spin`.
    let qnorm = (q1[0].powi(2) + q1[1].powi(2) + q1[2].powi(2) + q1[3].powi(2)).sqrt();
    assert!(approx_eq(qnorm, 1.0, 1e-9));

    // `orientation_matrix` returns the 3x3 rotation R for the current quaternion.
    let r = spinner.orientation_matrix();
    println!("  orientation_matrix (R):");
    for row in 0..3 {
        println!(
            "    [{:+.3}, {:+.3}, {:+.3}]",
            r[(row, 0)],
            r[(row, 1)],
            r[(row, 2)]
        );
    }
    // The free `quat_to_matrix` helper must agree with the body method.
    let r2 = quat_to_matrix(&spinner.orientation);
    for row in 0..3 {
        for col in 0..3 {
            assert!(approx_eq(r[(row, col)], r2[(row, col)], 1e-12));
        }
    }

    // Quaternion helpers: compose two 180°-about-z rotations via `quat_mul`
    // (should yield the identity up to sign) and normalise a non-unit quaternion.
    let q_half = [0.0, 0.0, 0.0, 1.0]; // 180° about z
    let composed = quat_mul(q_half, q_half); // 360° about z == identity (q or -q)
    println!(
        "  quat_mul(180°z, 180°z) ~ identity quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]",
        composed[0], composed[1], composed[2], composed[3]
    );
    assert!(approx_eq(composed[0].abs(), 1.0, 1e-9));
    assert!(approx_eq(composed[1], 0.0, 1e-9));
    assert!(approx_eq(composed[2], 0.0, 1e-9));
    assert!(approx_eq(composed[3], 0.0, 1e-9));

    let unnorm = quat_normalize(2.0, 0.0, 0.0, 0.0); // (2,0,0,0) -> (1,0,0,0)
    assert!(approx_eq(unnorm[0], 1.0, 1e-12));
    spinner.set_orientation(unnorm);
    println!(
        "  quat_normalize + set_orientation applied; new q = [{:.3}, {:.3}, {:.3}, {:.3}]",
        unnorm[0], unnorm[1], unnorm[2], unnorm[3]
    );

    println!("\nAll invariants held: states finite, momentum & energy conserved.");
}
