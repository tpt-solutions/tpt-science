//! Public-API invariants for the square-gradient classical DFT bulk functional.

use tpt_sci_dft_classical::{SquareGradientDft, VdWParams};

fn params() -> VdWParams {
    VdWParams::new(1.0, -0.8, 0.1)
}

#[test]
fn dmu_dn_matches_numerical_derivative_of_chemical_potential() {
    let p = params();
    for &n in &[0.05, 0.1, 0.3] {
        let h = 1e-6 * n;
        let numeric = (p.chemical_potential(n + h) - p.chemical_potential(n - h)) / (2.0 * h);
        let analytic = p.dmu_dn(n);
        assert!(
            (numeric - analytic).abs() < 1e-5 * analytic.abs().max(1.0),
            "n = {n}: numeric {numeric} vs analytic {analytic}"
        );
    }
}

#[test]
fn bulk_density_inverts_chemical_potential() {
    let p = params();
    let n = 0.2;
    let mu = p.chemical_potential(n);
    let dft = SquareGradientDft::new(params());
    let back = dft.bulk_density(mu).unwrap();
    assert!((back - n).abs() < 1e-6, "round trip {back} vs {n}");
}
