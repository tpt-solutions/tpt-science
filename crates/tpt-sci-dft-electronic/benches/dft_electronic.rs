use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_dft_electronic::{Grid1D, KohnSham};

fn bench_kohn_sham_solve(c: &mut Criterion) {
    c.bench_function("kohn_sham_1d_101pts_5scf", |b| {
        b.iter(|| {
            let grid = Grid1D::new(black_box(101), -5.0, 5.0).unwrap();
            let v_ext: Vec<f64> = grid.x().iter().map(|&x| 0.5 * x * x).collect();
            let mut ks = KohnSham::new(grid, v_ext, 2).unwrap();
            ks.solve(black_box(5))
        });
    });
}

criterion_group!(benches, bench_kohn_sham_solve);
criterion_main!(benches);
