//! Small dense linear-algebra helpers for the implicit solvers.
//!
//! These wrap [`tpt_math_linalg_dense`]'s in-house `DMatrix`/`DVector` (the same
//! `faer`-free dense kernels used everywhere in the `tpt-math` substrate). We
//! deliberately do *not* pull in `nalgebra`/`faer` here, so the shipped
//! `tpt-sci-ode` crate carries only dual-licensed, TPT-owned math.

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

/// RHS signature: `f(t, y) -> dydt`, written into `out` to avoid allocations.
pub(crate) type RhsFn = dyn Fn(f64, &[f64], &mut [f64]);

/// Evaluate `f(t, y)` into a fresh `Vec<f64>`.
pub(crate) fn eval(f: &RhsFn, t: f64, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; y.len()];
    f(t, y, &mut out);
    out
}

/// Full Jacobian `J = df/dy` at `(t, y)` via forward finite differences.
///
/// `f0` is `f(t, y)` and is passed in to avoid one redundant evaluation. The
/// step is scaled per-component (ε·|yᵢ| + ε) so the Jacobian is well-conditioned
/// near zero states. Systems solved by this pillar are small (≲ a few hundred
/// states), so the O(n²) evaluation cost is negligible.
pub(crate) fn jacobian(f: &RhsFn, t: f64, y: &[f64], f0: &[f64]) -> DMatrix<f64> {
    let n = y.len();
    let mut data = vec![0.0; n * n];
    let sqrt_eps = f64::sqrt(f64::EPSILON);
    for j in 0..n {
        let dy = sqrt_eps * y[j].abs().max(1.0);
        let mut yp = y.to_vec();
        yp[j] += dy;
        let fp = eval(f, t, &yp);
        let inv = 1.0 / dy;
        for i in 0..n {
            // Column-major storage to match DMatrix's Index<(i, j)> layout.
            data[i * n + j] = (fp[i] - f0[i]) * inv;
        }
    }
    DMatrix::from_row_slice(n, n, &data)
}

/// Solve `(I - γ·J) · Δ = -F` for `Δ`, where `J` is the Jacobian and `F` the
/// residual, using `DMatrix::solve` (dense LU). Returns `None` if the linear
/// system is singular (the step should be retried with a smaller `h`).
pub(crate) fn solve_newton_system(jac: &DMatrix<f64>, gamma: f64, f: &[f64]) -> Option<Vec<f64>> {
    let n = jac.nrows();
    // A = I - γ·J  (row-major build, then handed to from_row_slice).
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j { 1.0 } else { 0.0 } - gamma * jac[(i, j)];
        }
    }
    let a = DMatrix::from_row_slice(n, n, &a);
    let b = DVector::from_row_slice(f); // -F supplied by caller as `f`
    a.solve(&b).ok().map(|v| v.iter().copied().collect())
}

/// Two-norm of a slice.
pub(crate) fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
