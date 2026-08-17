use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_sci_quantum::State;

fn bench_hadamard_chain(c: &mut Criterion) {
    c.bench_function("hadamard_20qubits", |b| {
        b.iter(|| {
            let mut s = State::new(black_box(20)).unwrap();
            for q in 0..20 {
                s.h(q).unwrap();
            }
        });
    });
}

criterion_group!(benches, bench_hadamard_chain);
criterion_main!(benches);
