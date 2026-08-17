use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_sci_grid::{laplacian_1d, Boundary, UniformGrid1D};

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
