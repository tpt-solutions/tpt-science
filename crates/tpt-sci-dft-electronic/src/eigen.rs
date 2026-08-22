//! Internal eigensolver utilities for `tpt-sci-dft-electronic`.
//!
//! Provides a dense real-symmetric Jacobi diagonalizer (re-used by the plane-wave
//! periodic solver) and a Lanczos eigensolver that extracts the lowest few
//! eigenpairs of a large, sparse, real-symmetric operator (used by the 3-D
//! real-space Kohn–Sham solver where a full dense diagonalization is
//! infeasible).

/// Dot product of two equal-length real vectors.
#[must_use]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// In-place Euclidean normalization of a real vector.
///
/// If the vector is already zero (zero norm) it is left unchanged.
pub(crate) fn normalize(v: &mut [f64]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Symmetric eigendecomposition of a small real symmetric matrix `a` by Jacobi
/// rotations. Returns `(eigenvalues, eigenvectors_columns)`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn jacobi(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if m[i][j].abs() > max {
                    max = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let app = m[p][p];
        let aqq = m[q][q];
        let apq = m[p][q];
        // tan(2φ) = 2·a_pq / (a_qq − a_pp) is the classic Jacobi rotation angle
        // that zeros a_pq; `atan2` keeps this well-defined at a_pp == a_qq.
        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();
        // Rotate columns p,q for all rows, then rows p,q for all columns,
        // keeping the matrix symmetric. The diagonal and (p,q) entries are set
        // explicitly from the *original* submatrix to avoid reading values that
        // the column sweep has already overwritten.
        for i in 0..n {
            if i != p && i != q {
                let mip = m[i][p];
                let miq = m[i][q];
                let new_p = c * mip - s * miq;
                let new_q = s * mip + c * miq;
                m[i][p] = new_p;
                m[p][i] = new_p;
                m[i][q] = new_q;
                m[q][i] = new_q;
            }
        }
        m[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        m[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let mut eig = vec![0.0; n];
    for i in 0..n {
        eig[i] = m[i][i];
    }
    let vecs: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| v[i][j]).collect()).collect();
    (eig, vecs)
}

/// Extract the `num_eig` lowest eigenvalues and their (Ritz) eigenvectors of a
/// real-symmetric operator defined by the matrix–vector closure `matvec`.
///
/// A Lanczos iteration with full re-orthogonalization is used so that only
/// `O(num_eig)` extremal eigenpairs are resolved even when the operator acts on
/// `n` (potentially very large) degrees of freedom. The returned eigenvectors
/// are normalized so that `Σ_i v_i² = 1` (the discrete-grid normalization).
///
/// # Panics
///
/// Panics if `matvec` returns a vector whose length differs from `n`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn lanczos_lowest(
    n: usize,
    num_eig: usize,
    max_iter: usize,
    matvec: impl Fn(&[f64]) -> Vec<f64>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Deterministic start vector (fixed-seed xorshift) for reproducible results.
    let mut state: u64 = 0x9E37_79B9_2B29_7A2D;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut v_start = vec![0.0_f64; n];
    for x in &mut v_start {
        *x = rng() - 0.5;
    }
    normalize(&mut v_start);

    let mut q: Vec<Vec<f64>> = Vec::with_capacity(max_iter + 1);
    q.push(v_start.clone());

    let mut alpha: Vec<f64> = Vec::with_capacity(max_iter);
    let mut beta: Vec<f64> = Vec::with_capacity(max_iter);

    let mut v_prev = vec![0.0_f64; n];
    let mut beta_prev = 0.0_f64;
    let mut v_cur = v_start;

    for _ in 0..max_iter {
        let mut w = matvec(&v_cur);
        assert_eq!(w.len(), n, "matvec must preserve dimension");
        if beta_prev != 0.0 {
            for (wi, vp) in w.iter_mut().zip(&v_prev) {
                *wi -= beta_prev * vp;
            }
        }
        let a = dot(&v_cur, &w);
        for (wi, vc) in w.iter_mut().zip(&v_cur) {
            *wi -= a * vc;
        }
        // Full re-orthogonalization against all previous Lanczos vectors.
        for qv in &q {
            let proj = dot(qv, &w);
            for (wi, qvi) in w.iter_mut().zip(qv) {
                *wi -= proj * qvi;
            }
        }
        let b = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        alpha.push(a);
        if b < 1e-12 {
            beta.push(0.0);
            break;
        }
        let v_next: Vec<f64> = w.iter().map(|x| x / b).collect();
        beta.push(b);
        v_prev = std::mem::replace(&mut v_cur, v_next.clone());
        beta_prev = b;
        q.push(v_next);
    }

    let m = alpha.len();
    let mut tmat = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        tmat[i][i] = alpha[i];
        if i + 1 < m {
            tmat[i][i + 1] = beta[i + 1];
            tmat[i + 1][i] = beta[i + 1];
        }
    }
    let (teig, tevec) = jacobi(&tmat);

    // Sort eigenvalue indices ascending.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| teig[i].partial_cmp(&teig[j]).unwrap());

    let take = num_eig.min(m);
    let mut eigvals = Vec::with_capacity(take);
    let mut eigvecs = Vec::with_capacity(take);
    for &idx in order.iter().take(take) {
        let col = &tevec[idx];
        let mut r = vec![0.0_f64; n];
        for (qi, &c) in q.iter().zip(col) {
            for (ri, qvi) in r.iter_mut().zip(qi) {
                *ri += c * qvi;
            }
        }
        normalize(&mut r);
        eigvals.push(teig[idx]);
        eigvecs.push(r);
    }
    (eigvals, eigvecs)
}
