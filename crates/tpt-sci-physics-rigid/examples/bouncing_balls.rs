//! A small rigid-body (sphere) world: two balls under gravity, bouncing inside
//! an axis-aligned box.
//!
//! Run with: `cargo run --example bouncing_balls -p tpt-sci-physics-rigid`

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::{Body, World};

fn main() {
    let mut world = World::with_gravity(DVector::from_row_slice(&[0.0, -9.8]));
    world.set_bounds(DVector::from_row_slice(&[5.0, 5.0]));
    world.set_restitution(0.9);

    world
        .add_body(
            Body::new(
                0,
                DVector::from_row_slice(&[0.0, 4.0]),
                DVector::from_row_slice(&[0.5, 0.0]),
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
                DVector::from_row_slice(&[-1.0, 0.0]),
                DVector::from_row_slice(&[1.0, 0.0]),
                1.0,
                0.5,
            )
            .unwrap(),
        )
        .unwrap();

    let dt = 0.01;
    for step in 0..200 {
        world.step(dt);
        if step % 40 == 0 {
            let b = world.body(0).unwrap();
            println!(
                "t = {:.2}s  ball0 = ({:.2}, {:.2})",
                step as f64 * dt,
                b.position[0],
                b.position[1]
            );
        }
    }
}
