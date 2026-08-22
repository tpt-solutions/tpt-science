use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_math_prob_core::SplitMix64;
use tpt_sci_ppl::ModelBuilder;

fn gaussian_model() -> ModelBuilder {
    let data = [2.0_f64, 3.0, 2.5, 3.5, 3.0];
    let mut m = ModelBuilder::new();
    m.gaussian_parameter(0.0, 5.0);
    m.set_data(data.to_vec());
    m.likelihood(|t, v, d| {
        let mut s = t.constant(0.0);
        for &x in d.iter() {
            let z = (v[0] - x) / 1.0;
            s += -0.5 * z * z;
        }
        s
    });
    m
}

fn bench_nuts_short_run(c: &mut Criterion) {
    c.bench_function("nuts_fit_50_samples", |b| {
        b.iter(|| {
            let model = gaussian_model().build().unwrap();
            let mut rng = SplitMix64::seed_from_u64(42);
            model.fit(&mut rng, black_box(50)).unwrap()
        });
    });
}

criterion_group!(benches, bench_nuts_short_run);
criterion_main!(benches);
