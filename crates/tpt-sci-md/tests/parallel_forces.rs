//! Parallel Lennard-Jones kernel agrees with the serial one (requires `rayon`).
#![cfg(feature = "rayon")]

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Forces, Particle};

fn lattice() -> Vec<Particle> {
    (0..60)
        .map(|i| {
            let f = i as f64;
            let pos = DVector::from_row_slice(&[
                (f * 0.73).sin() * 4.0,
                (f * 1.31).cos() * 4.0,
                (f * 0.37).sin() * 4.0,
            ]);
            Particle::new(i, pos, DVector::zeros(3), 1.0).unwrap()
        })
        .collect()
}

#[test]
fn parallel_lennard_jones_matches_serial() {
    let mut serial = lattice();
    let mut parallel = lattice();
    let e_serial = Forces::lennard_jones(&mut serial, 9.0, 1.0);
    let e_parallel = Forces::lennard_jones_parallel(&mut parallel, 9.0, 1.0);
    assert!((e_serial - e_parallel).abs() <= 1e-10 * e_serial.abs().max(1.0));
    for (s, p) in serial.iter().zip(&parallel) {
        let diff = (s.force.clone() - p.force.clone()).norm();
        assert!(diff <= 1e-10 * s.force.norm().max(1.0), "force diff {diff}");
    }
}
