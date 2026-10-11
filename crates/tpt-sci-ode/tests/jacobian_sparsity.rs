//! Sparsity-aware Jacobians: detection, coloring, and agreement with the full
//! finite-difference Jacobian.

use std::cell::Cell;

use tpt_sci_ode::OdeError;
use tpt_sci_ode::sparse::{CsrMatrix, JacobianPattern};

/// Tridiagonal reaction-diffusion-like RHS: `f_i = y_{i-1} - 2 y_i + y_{i+1} + y_i²`.
/// Counts evaluations so the test can check the coloring saves calls.
fn tridiagonal(
    counter: &Cell<usize>,
) -> impl Fn(f64, &[f64], &mut [f64]) -> Result<(), OdeError> + '_ {
    move |_t, y, out| {
        counter.set(counter.get() + 1);
        let n = y.len();
        for i in 0..n {
            let left = if i > 0 { y[i - 1] } else { 0.0 };
            let right = if i + 1 < n { y[i + 1] } else { 0.0 };
            out[i] = left - 2.0 * y[i] + right + y[i] * y[i];
        }
        Ok(())
    }
}

#[test]
fn detected_pattern_is_tridiagonal() {
    let n = 40;
    let counter = Cell::new(0);
    let y: Vec<f64> = (0..n).map(|i| 0.3 + 0.01 * i as f64).collect();
    let pat = JacobianPattern::detect(tridiagonal(&counter), 0.0, &y).unwrap();
    assert_eq!(pat.nnz(), 3 * n - 2);
    for i in 0..n {
        let expected: Vec<usize> = (i.saturating_sub(1)..=(i + 1).min(n - 1)).collect();
        assert_eq!(pat.row(i), expected.as_slice(), "row {i}");
    }
}

#[test]
fn tridiagonal_pattern_needs_three_colors() {
    let pat = JacobianPattern::new(
        (0..30usize)
            .map(|i| (i.saturating_sub(1)..=(i + 1).min(29)).collect())
            .collect(),
    )
    .unwrap();
    let (colors, n_colors) = pat.column_coloring();
    assert_eq!(n_colors, 3);
    // Adjacent columns never share a color.
    for j in 0..29 {
        assert_ne!(colors[j], colors[j + 1]);
    }
}

#[test]
fn colored_jacobian_matches_full_fd_with_fewer_evaluations() {
    let n = 60;
    let y: Vec<f64> = (0..n).map(|i| 0.2 + 0.005 * i as f64).collect();

    let full_counter = Cell::new(0);
    let f = tridiagonal(&full_counter);
    let mut f0 = vec![0.0; n];
    f(0.0, &y, &mut f0).unwrap();
    let full = CsrMatrix::jacobian(&f, 0.0, &y, &f0, None).unwrap();
    let full_evals = full_counter.get() - 1;

    let pat = JacobianPattern::detect(tridiagonal(&Cell::new(0)), 0.0, &y).unwrap();
    let colored_counter = Cell::new(0);
    let g = tridiagonal(&colored_counter);
    let colored = CsrMatrix::jacobian_colored(&g, 0.0, &y, &f0, &pat).unwrap();
    let colored_evals = colored_counter.get();

    assert_eq!(colored_evals, 3, "expected one RHS call per color");
    assert!(full_evals >= n, "full FD should evaluate once per column");
    assert!(colored_evals < full_evals);

    for r in 0..n {
        for c in 0..n {
            let (a, b) = (full.get(r, c), colored.get(r, c));
            assert!(
                (a - b).abs() < 1e-6 * (1.0 + a.abs()),
                "({r},{c}): {a} vs {b}"
            );
        }
    }
}

#[test]
fn pattern_rejects_out_of_range_columns() {
    assert!(JacobianPattern::new(vec![vec![0, 5], vec![1]]).is_err());
}

#[test]
fn colored_jacobian_rejects_mismatched_pattern() {
    let pat = JacobianPattern::dense(3);
    let y = vec![1.0, 2.0];
    let f0 = vec![0.0, 0.0];
    let r = CsrMatrix::jacobian_colored(
        |_t, _y, out: &mut [f64]| {
            out.fill(0.0);
            Ok(())
        },
        0.0,
        &y,
        &f0,
        &pat,
    );
    assert!(r.is_err());
}
