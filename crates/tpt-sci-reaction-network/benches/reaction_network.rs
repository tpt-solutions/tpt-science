use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_reaction_network::ReactionNetwork;

fn build_system() -> tpt_sci_reaction_network::ReactionSystem {
    let mut model = ReactionNetwork::from_dsl(
        "kB, S + E --> SE
         kD, SE --> S + E
         kP, SE --> P + E",
    )
    .unwrap();
    model.set_parameter("kB", 0.01).unwrap();
    model.set_parameter("kD", 0.1).unwrap();
    model.set_parameter("kP", 0.1).unwrap();
    model
}

fn bench_reaction_rates(c: &mut Criterion) {
    let sys = build_system();
    let y = [50.0, 10.0, 0.0, 0.0];
    c.bench_function("reaction_rates", |b| {
        b.iter(|| sys.reaction_rates(black_box(&y)));
    });
}

fn bench_ssa_short(c: &mut Criterion) {
    let sys = build_system();
    let y0 = [50.0, 10.0, 0.0, 0.0];
    c.bench_function("ssa_t_max_1", |b| {
        b.iter_batched(
            || (0u64, 0x2545_F491_4F6C_DD1Du64),
            |mut state| {
                // xorshift64* uniform variates in [0, 1).
                let mut rng = move || {
                    let mut x = state.1;
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    state.1 = x;
                    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
                };
                sys.simulate_ssa(&y0, black_box(1.0), &mut rng).unwrap()
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_reaction_rates, bench_ssa_short);
criterion_main!(benches);
