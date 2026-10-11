//! Public-API invariants for classical molecular dynamics.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Forces, Integrator, Particle};

fn pair() -> Vec<Particle> {
    vec![
        Particle::new(
            0,
            DVector::from_row_slice(&[0.0, 0.0, 0.0]),
            DVector::zeros(3),
            1.0,
        )
        .unwrap(),
        Particle::new(
            1,
            DVector::from_row_slice(&[1.2, 0.0, 0.0]),
            DVector::zeros(3),
            1.0,
        )
        .unwrap(),
    ]
}

fn total_momentum(parts: &[Particle]) -> DVector<f64> {
    parts.iter().fold(DVector::zeros(3), |acc, p| {
        acc + p.velocity.clone() * p.mass
    })
}

#[test]
fn lennard_jones_pair_conserves_energy_and_momentum() {
    let mut parts = pair();
    let box_len = 10.0;
    let sigma = 1.0;
    let pe0 = Forces::lennard_jones(&mut parts, box_len, sigma);
    let e0 = pe0 + {
        let int = Integrator::new(box_len, sigma, 0.001).unwrap();
        int.kinetic_energy(&parts)
    };
    let int = Integrator::new(box_len, sigma, 0.001).unwrap();
    let mut pe = pe0;
    for _ in 0..400 {
        pe = int.velocity_verlet(&mut parts);
    }
    let e1 = pe + int.kinetic_energy(&parts);
    assert!(
        (e1 - e0).abs() < 1e-4 * e0.abs().max(1.0),
        "energy drift {e0} -> {e1}"
    );
    assert!(
        total_momentum(&parts).norm() < 1e-9,
        "momentum not conserved"
    );
}
