use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Forces, Integrator, Particle};

fn make_particles(n: usize) -> Vec<Particle> {
    // Simple cubic-ish placement inside a periodic box.
    let side = (n as f64).cbrt().ceil() as usize;
    let spacing = 1.5_f64;
    let mut parts = Vec::with_capacity(n);
    let mut id = 0usize;
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                if parts.len() == n {
                    break;
                }
                let p = Particle::new(
                    id,
                    DVector::from_row_slice(&[
                        ix as f64 * spacing,
                        iy as f64 * spacing,
                        iz as f64 * spacing,
                    ]),
                    DVector::zeros(3),
                    1.0,
                )
                .unwrap();
                parts.push(p);
                id += 1;
            }
        }
    }
    parts
}

fn bench_lennard_jones_pairwise(c: &mut Criterion) {
    c.bench_function("lj_forces_200", |b| {
        let mut parts = make_particles(black_box(200));
        b.iter(|| Forces::lennard_jones(&mut parts, 30.0, 1.0));
    });
}

fn bench_velocity_verlet_step(c: &mut Criterion) {
    c.bench_function("velocity_verlet_step_200", |b| {
        let mut parts = make_particles(black_box(200));
        let int = Integrator::new(30.0, 1.0, 0.005).unwrap();
        b.iter(|| int.velocity_verlet(&mut parts));
    });
}

criterion_group!(
    benches,
    bench_lennard_jones_pairwise,
    bench_velocity_verlet_step
);
criterion_main!(benches);
