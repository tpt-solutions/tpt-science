//! PME (particle–mesh Ewald) against the direct Ewald sum.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Ewald, Particle, Pppm};

/// A deterministic neutral 3-D ionic configuration.
fn ions() -> (Vec<Particle>, Vec<f64>) {
    let l = 10.0;
    let mut particles = Vec::new();
    let mut charges = Vec::new();
    for i in 0..16usize {
        let f = i as f64;
        let pos = DVector::from_row_slice(&[
            l * ((f * 0.61).sin() * 0.5 + 0.5),
            l * ((f * 1.17).cos() * 0.5 + 0.5),
            l * ((f * 0.29).sin() * 0.5 + 0.5),
        ]);
        particles.push(Particle::new(i, pos, DVector::zeros(3), 1.0).unwrap());
        charges.push(if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    (particles, charges)
}

#[test]
fn pme_energy_and_forces_match_direct_ewald() {
    let (parts, charges) = ions();
    let (l, alpha, rcut) = (10.0, 0.6, 4.5);
    let direct = Ewald::new(l, alpha, 12, rcut).unwrap();
    let (e_ref, f_ref) = direct.energy_forces(&parts, &charges).unwrap();

    let pme = Pppm::new(l, alpha, rcut, 32, 4).unwrap();
    let (e_pme, f_pme) = pme.energy_forces(&parts, &charges).unwrap();

    let rel_e = (e_pme - e_ref).abs() / e_ref.abs();
    assert!(
        rel_e < 1e-4,
        "energy: PME {e_pme} vs Ewald {e_ref} (rel {rel_e:e})"
    );

    let rms_ref = (f_ref.iter().map(|f| f.norm().powi(2)).sum::<f64>() / f_ref.len() as f64).sqrt();
    let worst = f_ref
        .iter()
        .zip(&f_pme)
        .map(|(a, b)| (a.clone() - b.clone()).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 1e-3 * rms_ref,
        "force error {worst:e} vs rms {rms_ref:e}"
    );
}

#[test]
fn finer_mesh_is_more_accurate() {
    let (parts, charges) = ions();
    let (l, alpha, rcut) = (10.0, 0.6, 4.5);
    let (e_ref, _) = Ewald::new(l, alpha, 12, rcut)
        .unwrap()
        .energy_forces(&parts, &charges)
        .unwrap();
    let err = |mesh| {
        let (e, _) = Pppm::new(l, alpha, rcut, mesh, 4)
            .unwrap()
            .energy_forces(&parts, &charges)
            .unwrap();
        (e - e_ref).abs()
    };
    assert!(
        err(32) < err(8),
        "mesh refinement should reduce the energy error"
    );
}

#[test]
fn pppm_rejects_bad_parameters() {
    assert!(Pppm::new(10.0, 0.6, 4.5, 1, 4).is_err());
    assert!(Pppm::new(10.0, 0.6, 4.5, 16, 1).is_err());
    assert!(Pppm::new(10.0, 0.6, 4.5, 8, 9).is_err());
}
