use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{EamParams, Ewald, Forces, Integrator, Particle, Pppm};

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

/// Periodic box side that fits the cubic lattice from `make_particles(n)`.
fn box_for(n: usize) -> f64 {
    ((n as f64).cbrt().ceil() * 1.5).max(30.0)
}

fn bench_lennard_jones_pairwise(c: &mut Criterion) {
    c.bench_function("lj_forces_200", |b| {
        let mut parts = make_particles(black_box(200));
        b.iter(|| Forces::lennard_jones(&mut parts, 30.0, 1.0));
    });
}

fn bench_lennard_jones_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lj_scaling");
    group.sample_size(10);
    for n in [1000usize, 5000] {
        let box_len = box_for(n);
        group.bench_function(format!("all_pairs_{n}"), |b| {
            let mut parts = make_particles(n);
            b.iter(|| Forces::lennard_jones(&mut parts, box_len, 1.0));
        });
        group.bench_function(format!("cells_{n}"), |b| {
            let mut parts = make_particles(n);
            b.iter(|| Forces::lennard_jones_cells(&mut parts, box_len, 1.0).unwrap());
        });
        #[cfg(feature = "rayon")]
        group.bench_function(format!("parallel_{n}"), |b| {
            let mut parts = make_particles(n);
            b.iter(|| Forces::lennard_jones_parallel(&mut parts, box_len, 1.0));
        });
    }
    group.finish();
}

fn bench_eam(c: &mut Criterion) {
    let params = EamParams::generic();
    let mut group = c.benchmark_group("eam");
    group.sample_size(10);
    for n in [200usize, 1000] {
        group.bench_function(format!("forces_{n}"), |b| {
            let mut parts = make_particles(n);
            b.iter(|| Forces::eam(&mut parts, f64::INFINITY, &params));
        });
    }
    group.finish();
}

fn bench_electrostatics(c: &mut Criterion) {
    let mut group = c.benchmark_group("electrostatics");
    group.sample_size(10);
    for n in [200usize, 1000] {
        let parts = make_particles(n);
        let charges: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let box_len = box_for(n);
        let ewald = Ewald::new(box_len, 0.5, 4, 0.5 * box_len).unwrap();
        group.bench_function(format!("ewald_{n}"), |b| {
            b.iter(|| ewald.energy_forces(&parts, &charges).unwrap());
        });
        let pppm = Pppm::new(box_len, 0.5, 0.5 * box_len, 16, 4).unwrap();
        group.bench_function(format!("pppm_{n}"), |b| {
            b.iter(|| pppm.energy_forces(&parts, &charges).unwrap());
        });
    }
    group.finish();
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
    bench_lennard_jones_scaling,
    bench_eam,
    bench_electrostatics,
    bench_velocity_verlet_step
);
criterion_main!(benches);
