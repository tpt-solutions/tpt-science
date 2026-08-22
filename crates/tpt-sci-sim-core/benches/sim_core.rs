use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_sci_sim_core::{OdeSubModel, Simulation};

fn bench_step_until(c: &mut Criterion) {
    c.bench_function("sim_step_until_0p1_two_models", |b| {
        b.iter(|| {
            let mut sim = Simulation::new();
            sim.add_model(OdeSubModel::new(
                "decay",
                |_t, y, dydt| dydt[0] = -y[0],
                vec![1.0],
                0.0,
            ))
            .unwrap();
            sim.add_model(OdeSubModel::new(
                "growth",
                |_t, y, dydt| dydt[0] = 0.5 * y[0],
                vec![1.0],
                0.0,
            ))
            .unwrap();
            sim.step_until(black_box(0.1)).unwrap();
        });
    });
}

criterion_group!(benches, bench_step_until);
criterion_main!(benches);
