//! Classical MD demo: a small Lennard-Jones gas in a periodic box relaxes under
//! velocity-Verlet integration while a thermostat holds the target temperature.
//!
//! Run with: `cargo run --example lennard_jones -p tpt-sci-md`

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Integrator, Particle};

fn main() {
    let n = 64;
    let box_len = 8.0;
    let sigma = 1.0;
    let dt = 0.005;

    // Seed a face-centered-ish lattice with small random-ish velocities.
    let mut particles: Vec<Particle> = (0..n)
        .map(|i| {
            let x = (i % 8) as f64 * (box_len / 8.0);
            let y = ((i / 8) % 8) as f64 * (box_len / 8.0);
            let z = (i / 64) as f64;
            let vel = DVector::from_row_slice(&[
                (((i * 7) % 10) as f64 - 5.0) * 0.1,
                (((i * 3) % 10) as f64 - 5.0) * 0.1,
                (((i * 5) % 10) as f64 - 5.0) * 0.1,
            ]);
            Particle::new(
                i,
                DVector::from_row_slice(&[x, y, z]),
                vel,
                1.0,
            )
            .unwrap()
        })
        .collect();

    let int = Integrator::new(box_len, sigma, dt).unwrap();
    let target_t = 1.0;

    println!("step  t(K)        E_kin       E_pot");
    for step in 0..2000 {
        let epot = int.velocity_verlet(&mut particles);
        int.thermostat(&mut particles, target_t, 0.5);
        if step % 200 == 0 {
            let ekin = int.kinetic_energy(&particles);
            let t = int.temperature(&particles);
            println!("{step:5}  {t:9.4}  {ekin:10.4}  {epot:10.4}");
        }
    }
}
