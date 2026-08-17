use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_grid::{Boundary, UniformGrid1D, laplacian_1d};

fn bench_laplacian_assembly(c: &mut Criterion) {
    c.bench_function("laplacian_1d_201", |b| {
        b.iter(|| {
            let g = UniformGrid1D::new(black_box(201), 0.0, 1.0).unwrap();
            laplacian_1d(&g, Boundary::Dirichlet)
        });
    });
}

criterion_group!(benches, bench_laplacian_assembly);
criterion_main!(benches);
