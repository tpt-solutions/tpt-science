use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_kinetics::{
    ArrheniusRate, CoverageDependentArrheniusRate, langmuir_hinshelwood_coverages,
    multi_site_langmuir_hinshelwood_coverages,
};

fn bench_arrhenius_rate_constant(c: &mut Criterion) {
    let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
    c.bench_function("arrhenius_rate_constant", |b| {
        b.iter(|| r.rate_constant(black_box(800.0)));
    });
}

fn bench_langmuir_hinshelwood(c: &mut Criterion) {
    c.bench_function("langmuir_hinshelwood_4", |b| {
        let ks = [1.0, 2.0, 0.5, 3.0];
        let p = [0.5, 1.0, 0.2, 0.8];
        b.iter(|| langmuir_hinshelwood_coverages(black_box(&ks), black_box(&p)).unwrap());
    });
}

fn bench_multi_site_coverages(c: &mut Criterion) {
    c.bench_function("multi_site_langmuir_hinshelwood", |b| {
        let ks = [1.0, 2.0, 0.5, 3.0];
        let p = [0.5, 1.0, 0.2, 0.8];
        let sites = [0, 0, 1, 1];
        b.iter(|| {
            multi_site_langmuir_hinshelwood_coverages(
                black_box(&ks),
                black_box(&p),
                black_box(&sites),
            )
            .unwrap()
        });
    });
}

fn bench_coverage_dependent_rate(c: &mut Criterion) {
    let r = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, -20_000.0).unwrap();
    c.bench_function("coverage_dependent_rate_constant", |b| {
        b.iter(|| r.rate_constant(black_box(800.0), black_box(0.4)));
    });
}

criterion_group!(
    benches,
    bench_arrhenius_rate_constant,
    bench_langmuir_hinshelwood,
    bench_multi_site_coverages,
    bench_coverage_dependent_rate
);
criterion_main!(benches);
