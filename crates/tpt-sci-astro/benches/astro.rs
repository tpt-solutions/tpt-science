use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_astro::{EARTH_J2, EARTH_MU, EARTH_RADIUS_EQ, OrbitalElements};

fn elements() -> OrbitalElements {
    // ISS-like orbit.
    OrbitalElements::new(
        6778.0, // a (km)
        0.0003, // e
        51.6_f64.to_radians(),
        30.0_f64.to_radians(),
        120.0_f64.to_radians(),
        0.0, // M0
        EARTH_MU,
    )
    .unwrap()
}

fn bench_two_body_propagate(c: &mut Criterion) {
    c.bench_function("two_body_propagate", |b| {
        let el = elements();
        b.iter(|| el.propagate(black_box(60.0)));
    });
}

fn bench_j2_rates(c: &mut Criterion) {
    c.bench_function("j2_secular_rates", |b| {
        let el = elements();
        b.iter(|| el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ));
    });
}

criterion_group!(benches, bench_two_body_propagate, bench_j2_rates);
criterion_main!(benches);
