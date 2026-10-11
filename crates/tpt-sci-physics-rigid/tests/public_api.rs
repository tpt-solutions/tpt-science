//! Public-API invariants for rigid-body dynamics.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::{Body, World};

#[test]
fn free_fall_matches_one_half_g_t_squared() {
    let g = 9.8;
    let mut world = World::with_gravity(DVector::from_row_slice(&[0.0, -g]));
    let ball = Body::new(
        0,
        DVector::from_row_slice(&[0.0, 4.0]),
        DVector::from_row_slice(&[0.0, 0.0]),
        1.0,
        0.2,
    )
    .unwrap();
    world.add_body(ball).unwrap();
    let dt = 0.001;
    let steps = 500;
    for _ in 0..steps {
        world.step(dt);
    }
    let t = dt * steps as f64;
    let dy = world.body(0).unwrap().position[1] - 4.0;
    assert!((dy + 0.5 * g * t * t).abs() < 1e-2, "dy = {dy}");
}
