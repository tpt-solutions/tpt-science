use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_physics_rigid::{Body, World};

fn make_world(n: usize) -> World {
    let mut world = World::with_gravity(DVector::from_row_slice(&[0.0_f64, -9.81]));
    let side = (n as f64).sqrt().ceil() as usize;
    for i in 0..n {
        let (ix, iy) = (i % side, i / side);
        let body = Body::new(
            i,
            DVector::from_row_slice(&[ix as f64 * 1.1, iy as f64 * 1.1 + 1.0]),
            DVector::from_row_slice(&[(i as f64).sin(), 0.0]),
            1.0,
            0.5,
        )
        .unwrap();
        world.add_body(body).unwrap();
    }
    world
}

fn bench_world_step(c: &mut Criterion) {
    c.bench_function("world_step_100_bodies", |b| {
        let mut world = make_world(black_box(100));
        b.iter(|| world.step(black_box(1.0 / 60.0)));
    });
}

criterion_group!(benches, bench_world_step);
criterion_main!(benches);
