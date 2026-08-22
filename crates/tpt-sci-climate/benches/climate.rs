use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_climate::{AtmosphereGcm, EnergyBalanceModel};

fn bench_ebm_equilibrium(c: &mut Criterion) {
    c.bench_function("ebm_step", |b| {
        let mut ebm = EnergyBalanceModel::new(1.0e23, 0.3, 0.61, 280.0).unwrap();
        b.iter(|| ebm.step(black_box(3600.0 * 24.0 * 30.0)));
    });
}

fn bench_gcm_step(c: &mut Criterion) {
    c.bench_function("gcm_step_16x16x8", |b| {
        let mut gcm = AtmosphereGcm::new(
            black_box(16),
            black_box(16),
            black_box(8),
            1.0e6,
            1.0e6,
            1.0e4,
            1.0,
            1.0e-3,
            250.0,
            9.81,
            1.0e-4,
            1.0e-11,
            1.0e4,
        )
        .unwrap();
        gcm.t_eq = 250.0;
        b.iter(|| gcm.step(black_box(60.0)));
    });
}

criterion_group!(benches, bench_ebm_equilibrium, bench_gcm_step);
criterion_main!(benches);
