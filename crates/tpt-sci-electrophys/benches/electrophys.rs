use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_electrophys::Tissue;

fn bench_monodomain_step(c: &mut Criterion) {
    c.bench_function("monodomain_step_32x32", |b| {
        let mut tissue = Tissue::new(32, 32, 0.001).unwrap();
        // Launch a stimulus at one corner so the step does real work.
        tissue.vm[0] = 20.0;
        b.iter(|| tissue.step(black_box(0.01)));
    });
}

criterion_group!(benches, bench_monodomain_step);
criterion_main!(benches);
