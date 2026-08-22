use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_hemodynamics::{Network, Vessel, tube_law_beta, womersley_velocity_profile};

fn bench_network_step(c: &mut Criterion) {
    c.bench_function("network_step", |b| {
        let beta = tube_law_beta(1.0e5, 1.0, 1.0);
        let vessel = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
        let mut net = Network::new(vessel, 1060.0, 0.01).unwrap();
        b.iter(|| net.step(black_box(1.0e-4)));
    });
}

fn bench_womersley_profile(c: &mut Criterion) {
    c.bench_function("womersley_velocity_profile", |b| {
        b.iter(|| {
            womersley_velocity_profile(
                black_box(0.005),
                black_box(0.01),
                black_box(4.0),
                black_box(2.0 * std::f64::consts::PI),
                black_box(1060.0),
                black_box(0.05),
            )
        });
    });
}

criterion_group!(benches, bench_network_step, bench_womersley_profile);
criterion_main!(benches);
