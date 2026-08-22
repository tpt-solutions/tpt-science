use criterion::{Criterion, criterion_group, criterion_main};
use tpt_sci_cfd_core::{Boundary, CollocatedGrid, Step};

fn bench_explicit_advance(c: &mut Criterion) {
    let grid = CollocatedGrid::new(64, 64, 1.0, 1.0).unwrap();
    let mut step = Step::new(grid, 1.0, 1e-3, 1.0);
    step.set_boundary(Boundary::Top, 1.0);
    c.bench_function("fractional_step_advance_64x64", |b| {
        b.iter(|| step.advance());
    });
}

criterion_group!(benches, bench_explicit_advance);
criterion_main!(benches);
