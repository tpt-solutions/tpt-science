//! Property-based checks for the grid operators: dense and sparse Laplacians
//! agree, linearity holds, and the Neumann Laplacian conserves total mass.

use proptest::prelude::*;
use tpt_sci_grid::operator::laplacian_1d;
use tpt_sci_grid::sparse::laplacian_1d_sparse;
use tpt_sci_grid::{Boundary, UniformGrid1D};

fn dense_mul(m: &tpt_math_linalg::tpt_math_linalg_dense::DMatrix<f64>, v: &[f64]) -> Vec<f64> {
    (0..v.len())
        .map(|i| (0..v.len()).map(|j| m[(i, j)] * v[j]).sum())
        .collect()
}

fn bc_strategy() -> impl Strategy<Value = Boundary> {
    prop_oneof![Just(Boundary::Dirichlet), Just(Boundary::Neumann)]
}

proptest! {
    #[test]
    fn dense_and_sparse_laplacians_agree(
        n in 3usize..40,
        bc in bc_strategy(),
        seed in proptest::collection::vec(-10.0f64..10.0, 40),
    ) {
        let grid = UniformGrid1D::new(n, 0.0, 1.0).unwrap();
        let v = &seed[..n];
        let dense = dense_mul(&laplacian_1d(&grid, bc), v);
        let sparse = laplacian_1d_sparse(&grid, bc).mul_vec(v);
        for (d, s) in dense.iter().zip(&sparse) {
            prop_assert!((d - s).abs() <= 1e-9 * (1.0 + d.abs()), "dense {d} vs sparse {s}");
        }
    }

    #[test]
    fn laplacian_is_linear(
        n in 3usize..30,
        bc in bc_strategy(),
        a in -5.0f64..5.0,
        u in proptest::collection::vec(-10.0f64..10.0, 30),
        w in proptest::collection::vec(-10.0f64..10.0, 30),
    ) {
        let grid = UniformGrid1D::new(n, 0.0, 2.0).unwrap();
        let lap = laplacian_1d(&grid, bc);
        let combo: Vec<f64> = (0..n).map(|i| a * u[i] + w[i]).collect();
        let lhs = dense_mul(&lap, &combo);
        let lu = dense_mul(&lap, &u[..n]);
        let lw = dense_mul(&lap, &w[..n]);
        for i in 0..n {
            let rhs = a * lu[i] + lw[i];
            prop_assert!((lhs[i] - rhs).abs() <= 1e-8 * (1.0 + rhs.abs()));
        }
    }

    // The Neumann rows are the half-cell finite-volume form (2(u₁−u₀)/dx² at the
    // ends), so the discrete mass conserved by the operator is the trapezoid sum,
    // with half weight on the two end nodes. The plain sum is not conserved.
    #[test]
    fn neumann_laplacian_conserves_trapezoid_mass(
        n in 3usize..30,
        u in proptest::collection::vec(-10.0f64..10.0, 30),
    ) {
        let grid = UniformGrid1D::new(n, 0.0, 1.0).unwrap();
        let lu = dense_mul(&laplacian_1d(&grid, Boundary::Neumann), &u[..n]);
        let weight = |i: usize| if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
        let total: f64 = lu.iter().enumerate().map(|(i, x)| weight(i) * x).sum();
        let scale: f64 = lu.iter().map(|x| x.abs()).sum::<f64>() + 1.0;
        prop_assert!(total.abs() <= 1e-9 * scale, "trapezoid flux {total}");
    }
}
