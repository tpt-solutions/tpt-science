//! Public-API invariants for the 1-D Kohn–Sham LDA solver.

use tpt_sci_dft_electronic::{Grid1D, KohnSham, lda_xc};

#[test]
fn lda_exchange_correlation_is_attractive() {
    for rho in [1e-3, 0.01, 0.1, 1.0, 10.0] {
        assert!(lda_xc(rho) < 0.0, "e_xc({rho}) should be negative");
    }
}

#[test]
fn harmonic_well_gives_finite_total_energy() {
    let grid = Grid1D::new(101, -5.0, 5.0).unwrap();
    let v_ext: Vec<f64> = grid.x().iter().map(|&x| 0.5 * x * x).collect();
    let mut ks = KohnSham::new(grid, v_ext, 2).unwrap();
    let res = ks.solve(50);
    assert!(res.total_energy.is_finite());
}
