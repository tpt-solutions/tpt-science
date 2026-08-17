//! Elastic collision plus rotational dynamics with `tpt-sci-physics-rigid`.
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::{Body, World};

fn main() {
    let mut world = World::new();
    world
        .add_body(Body::new(0, DVector::from_row_slice(&[0.0, 0.0]), DVector::from_row_slice(&[1.0, 0.0]), 1.0, 0.5).unwrap())
        .unwrap();
    world
        .add_body(Body::new(1, DVector::from_row_slice(&[1.0, 0.0]), DVector::from_row_slice(&[0.0, 0.0]), 1.0, 0.5).unwrap())
        .unwrap();
    let p0: f64 = world.bodies().iter().map(|b| b.mass * b.velocity[0]).sum();
    world.step(0.5);
    let a = world.body(0).unwrap();
    let b = world.body(1).unwrap();
    println!(
        "After collision: v0 = {:.3}, v1 = {:.3} (velocities exchange)",
        a.velocity[0], b.velocity[0]
    );
    let p1: f64 = world.bodies().iter().map(|b| b.mass * b.velocity[0]).sum();
    println!("Momentum conserved: {p0:.3} -> {p1:.3}");

    // Rotation: spin a body a quarter turn about z.
    let mut spinner = Body::new(2, DVector::from_row_slice(&[0.0, 0.0]), DVector::from_row_slice(&[0.0, 0.0]), 1.0, 0.5).unwrap();
    spinner.set_angular_velocity([0.0, 0.0, 2.0]);
    spinner.spin(0.25);
    let q = spinner.orientation;
    println!(
        "After a quarter-turn spin, orientation quaternion = [{:.3}, {:.3}, {:.3}, {:.3}]",
        q[0], q[1], q[2], q[3]
    );
}
