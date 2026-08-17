use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{linspace, radon_transform};

fn bench_radon_64(c: &mut Criterion) {
    let image = DMatrix::from_fn(64, 64, |i, j| {
        if (i as f64 - 31.5).hypot(j as f64 - 31.5) < 10.0 {
            1.0
        } else {
            0.0
        }
    });
    let angles = linspace(0.0, std::f64::consts::PI, 90);
    c.bench_function("radon_64x64_90angles", |b| {
        b.iter(|| radon_transform(black_box(&image), &angles).unwrap());
    });
}

criterion_group!(benches, bench_radon_64);
criterion_main!(benches);
