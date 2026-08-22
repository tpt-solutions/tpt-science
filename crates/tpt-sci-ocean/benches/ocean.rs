use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_ocean::{Ocean3D, ShallowWater};

fn bench_shallow_water_step(c: &mut Criterion) {
    c.bench_function("shallow_water_step_64x64", |b| {
        let mut sw = ShallowWater::new(64, 64, 1.0e5, 1.0e5, 9.81, 1.0e-4, 1000.0);
        sw.perturb_center(1.0);
        b.iter(|| sw.step(black_box(0.001)));
    });
}

fn bench_ocean3d_step(c: &mut Criterion) {
    c.bench_function("ocean3d_hydrostatic_step_16x16x8", |b| {
        let mut ocean = Ocean3D::new(
            black_box(16),
            black_box(16),
            black_box(8),
            1.0e6,
            1.0e6,
            4000.0,
            1025.0,
            0.2,
            0.8,
            10.0,
            35.0,
            9.81,
            1.0e-4,
            1.0e-4,
        )
        .unwrap();
        // A warm surface patch drives a real pressure-gradient step.
        let warm = ocean.index(0, 0, 7);
        b.iter(|| {
            ocean.t[warm] += 2.0;
            ocean.step_3d(black_box(60.0))
        });
    });
}

criterion_group!(benches, bench_shallow_water_step, bench_ocean3d_step);
criterion_main!(benches);
