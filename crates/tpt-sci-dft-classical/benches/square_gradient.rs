use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_dft_classical::{PlanarSolve, SquareGradientDft, VdWParams};
use tpt_sci_grid::{Boundary, UniformGrid1D};

fn bench_square_gradient_solve_1d(c: &mut Criterion) {
    c.bench_function("square_gradient_solve_1d_201", |b| {
        b.iter(|| {
            let grid = UniformGrid1D::new(black_box(201), 0.0, 10.0).unwrap();
            let params = VdWParams::new(1.0, 1.0, 0.5);
            let dft = SquareGradientDft::new(params);
            let bulk = dft.bulk_density(1.5).unwrap();
            let initial = vec![bulk; 201];
            let cfg = PlanarSolve {
                mu: 1.5,
                initial,
                boundary: Boundary::Dirichlet,
                external_potential: None,
                tol: 1e-8,
                max_iter: 200,
            };
            dft.solve_1d(&grid, &cfg).unwrap()
        });
    });
}

criterion_group!(benches, bench_square_gradient_solve_1d);
criterion_main!(benches);
