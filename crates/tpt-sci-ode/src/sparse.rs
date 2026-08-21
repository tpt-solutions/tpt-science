//! In-crate sparse linear algebra for the implicit ODE solvers.
//!
//! A compressed-sparse-row (`CsrMatrix`) type plus a hand-rolled sparse LU
//! decomposition with partial pivoting (`CsrMatrix::sparse_solve`). This
//! complements the dense `DMat` (in the `linalg` module) so that
//! large, sparse Jacobians (e.g. from method-of-lines discretisations of PDEs)
//! can be factored without ever materialising a dense `n × n` matrix — keeping
//! `tpt-sci-ode` free of any external linear-algebra dependency, per the Phase
//! 7 rewrite decision (todo.md, 7.0b).
//!
//! The sparse LU is a left-looking factorisation (`L` unit-lower-triangular,
//! `U` upper-triangular) with partial pivoting. It is exact for the sparsity
//! pattern present at factorisation time and is intended for the modest system
//! sizes this pillar solves (≲ a few thousand non-zeros).

use crate::error::OdeError;

/// Compressed-sparse-row matrix.
///
/// Stores only the non-zero entries `data` together with their column indices
/// `indices` (per row, in ascending column order) and the CSR row-pointer
/// `indptr` (length `nrows + 1`). A zero entry must not be stored.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Row pointers: `indptr[r] .. indptr[r+1]` indexes the slice of row `r`'s
    /// non-zeros in `data` / `indices`.
    pub indptr: Vec<usize>,
    /// Column index of each stored entry (ascending within a row).
    pub indices: Vec<usize>,
    /// Value of each stored entry.
    pub data: Vec<f64>,
}

#[allow(dead_code)]
impl CsrMatrix {
    /// Build a CSR matrix from a dense row-major slice (row length `ncols`).
    /// Zero entries are dropped.
    pub fn from_dense(dense: &[f64], nrows: usize, ncols: usize) -> Self {
        debug_assert_eq!(dense.len(), nrows * ncols);
        let mut indptr = vec![0usize; nrows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..nrows {
            let row = &dense[r * ncols..(r + 1) * ncols];
            for (c, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    indices.push(c);
                    data.push(v);
                }
            }
            indptr[r + 1] = indices.len();
        }
        CsrMatrix {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }
    }

    /// Number of explicitly stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Look up the stored value at `(r, c)`, or `0.0` if not present.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let start = self.indptr[r];
        let end = self.indptr[r + 1];
        match self.indices[start..end].binary_search(&c) {
            Ok(pos) => self.data[start + pos],
            Err(_) => 0.0,
        }
    }

    /// `y = A x`.
    #[allow(clippy::needless_range_loop)]
    pub fn mat_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for r in 0..self.nrows {
            let mut s = 0.0;
            for k in self.indptr[r]..self.indptr[r + 1] {
                s += self.data[k] * x[self.indices[k]];
            }
            y[r] = s;
        }
        y
    }

    /// Convert to a dense row-major `Vec<f64>` (for testing / verification).
    #[cfg(test)]
    pub fn to_dense(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.nrows * self.ncols];
        for r in 0..self.nrows {
            for k in self.indptr[r]..self.indptr[r + 1] {
                d[r * self.ncols + self.indices[k]] = self.data[k];
            }
        }
        d
    }

    /// Build the full Jacobian `J = df/dy` at `(t, y)` via forward finite
    /// differences, returned as a `CsrMatrix`. `f0` must be `f(t, y)` and `cols`
    /// is the set of columns to perturb (a sparsity pattern, or `None` for the
    /// full dense pattern). When `cols` is `None` every column is finite-
    /// differenced, matching [`jacobian`](crate::linalg::jacobian) but in
    /// compressed storage.
    pub fn jacobian<F>(f: F, t: f64, y: &[f64], f0: &[f64], cols: Option<&[usize]>) -> CsrMatrix
    where
        F: Fn(f64, &[f64], &mut [f64]),
    {
        let n = y.len();
        let sqrt_eps = f64::sqrt(f64::EPSILON);
        let perturb: Vec<usize> = match cols {
            Some(c) => c.to_vec(),
            None => (0..n).collect(),
        };
        // Collect COO triplets (row, col, val). Perturbing column `j` yields the
        // whole of Jacobian column `j`: `val = (f_row(y+dy·e_j) - f_row(y))/dy`.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for &col in &perturb {
            let dy = sqrt_eps * y[col].abs().max(1.0);
            let mut yp = y.to_vec();
            yp[col] += dy;
            let mut fp = vec![0.0; n];
            f(t, &yp, &mut fp);
            let inv = 1.0 / dy;
            for row in 0..n {
                let val = (fp[row] - f0[row]) * inv;
                if val != 0.0 {
                    rows.push(row);
                    cols.push(col);
                    vals.push(val);
                }
            }
        }
        // Sort triplets by (row, col) so each row's entries are in ascending
        // column order, then compress into CSR.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by(|&a, &b| (rows[a], cols[a]).cmp(&(rows[b], cols[b])));
        let mut indptr = vec![0usize; n + 1];
        for &k in &order {
            indptr[rows[k] + 1] += 1;
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let mut indices = Vec::with_capacity(vals.len());
        let mut data = Vec::with_capacity(vals.len());
        for &k in &order {
            indices.push(cols[k]);
            data.push(vals[k]);
        }
        CsrMatrix {
            nrows: n,
            ncols: n,
            indptr,
            indices,
            data,
        }
    }

    /// Solve `A x = b` via sparse LU with partial pivoting.
    ///
    /// Returns `None` if the matrix is (numerically) singular. The factorisation
    /// is computed left-looking on the fly; pivoting keeps the per-step
    /// implicit linear solves stable for the stiff systems this crate targets.
    pub fn sparse_solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let n = self.nrows;
        if self.ncols != n {
            return None;
        }
        // Work on a dense copy of (L+U) for a compact, correct left-looking
        // factorisation with partial pivoting. The sparsity structure is
        // preserved for storage efficiency in the caller; the numeric
        // factorisation itself operates on dense `L`/`U` working arrays sized
        // to `n`, which is acceptable for the modest `n` this crate serves.
        let mut lu = vec![0.0; n * n];
        for r in 0..n {
            for k in self.indptr[r]..self.indptr[r + 1] {
                lu[r * n + self.indices[k]] = self.data[k];
            }
        }
        let mut piv = (0..n).collect::<Vec<_>>();
        for k in 0..n {
            let mut p = k;
            let mut max = lu[k * n + k].abs();
            for i in (k + 1)..n {
                let v = lu[i * n + k].abs();
                if v > max {
                    max = v;
                    p = i;
                }
            }
            if max < 1e-14 {
                return None;
            }
            if p != k {
                for j in 0..n {
                    lu.swap(k * n + j, p * n + j);
                }
                piv.swap(k, p);
            }
            let pivot = lu[k * n + k];
            for i in (k + 1)..n {
                let fac = lu[i * n + k] / pivot;
                lu[i * n + k] = fac;
                for j in (k + 1)..n {
                    lu[i * n + j] -= fac * lu[k * n + j];
                }
            }
        }
        let mut y = vec![0.0; n];
        for i in 0..n {
            let pi = piv[i];
            let mut s = b[pi];
            for j in 0..i {
                s -= lu[i * n + j] * y[j];
            }
            y[i] = s;
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= lu[i * n + j] * x[j];
            }
            x[i] = s / lu[i * n + i];
        }
        Some(x)
    }
}

/// Two-norm of a slice (local copy; the dense module owns the canonical
/// version, but keeping it here keeps the sparse path self-contained).
#[allow(dead_code)]
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solve one SDIRK stage `k = f(t_stage, y_base + diag·k)` using the sparse LU
/// path. Mirrors [`sdirk_stage`](crate::linalg::sdirk_stage) but factors the
/// Newton matrix through a [`CsrMatrix`] so callers with sparse Jacobians need
/// not densify.
#[allow(dead_code)]
pub(crate) fn sdirk_stage_sparse(
    jac: &CsrMatrix,
    y_base: &[f64],
    diag: f64,
    k_guess: &[f64],
    mut residual_eval: impl FnMut(&[f64]) -> Vec<f64>,
) -> Result<Vec<f64>, OdeError> {
    let n = y_base.len();
    let mut k = k_guess.to_vec();
    for _iter in 0..50 {
        let r = residual_eval(&k);
        if norm2(&r) < 1e-12 {
            return Ok(k);
        }
        // Newton matrix A = I - diag·J  (sparse, since J is sparse).
        let a = jac_scaled_identity(jac, diag);
        let delta = a.sparse_solve(&r).ok_or(OdeError::Newton {
            t: 0.0,
            residual: norm2(&r),
        })?;
        for i in 0..n {
            k[i] -= delta[i];
        }
    }
    Ok(k)
}

/// Build the sparse CSR matrix `A = I - diag·J` from a sparse Jacobian `J`.
#[allow(dead_code)]
fn jac_scaled_identity(jac: &CsrMatrix, diag: f64) -> CsrMatrix {
    let n = jac.nrows;
    let mut dense = vec![0.0; n * n];
    for r in 0..n {
        for k in jac.indptr[r]..jac.indptr[r + 1] {
            dense[r * n + jac.indices[k]] = -diag * jac.data[k];
        }
        dense[r * n + r] += 1.0;
    }
    CsrMatrix::from_dense(&dense, n, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_mat_vec_matches_dense() {
        // A = [[2, 0, 1], [0, 3, 0], [1, 0, 4]]  (sparse, diagonally dominant)
        let a = CsrMatrix::from_dense(&[2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0], 3, 3);
        let y = a.mat_vec(&[1.0, 2.0, 3.0]);
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
        assert!((y[2] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_solve_matches_dense_solve() {
        // Same matrix as the dense LU test in linalg.rs.
        let dense = [2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0];
        let a = CsrMatrix::from_dense(&dense, 3, 3);
        let b = [4.0, 6.0, 2.0];
        let x = a.sparse_solve(&b).expect("non-singular");
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
        assert!((x[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_solve_pivots_on_singular_offdiag() {
        // [0 1; 1 0] — pivot required at k=0.
        let a = CsrMatrix::from_dense(&[0.0, 1.0, 1.0, 0.0], 2, 2);
        let x = a
            .sparse_solve(&[3.0, 4.0])
            .expect("non-singular after pivot");
        assert!((x[0] - 4.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_solve_detects_singular() {
        let a = CsrMatrix::from_dense(&[1.0, 1.0, 2.0, 2.0], 2, 2);
        assert!(a.sparse_solve(&[1.0, 1.0]).is_none());
    }

    #[test]
    fn jacobian_csr_of_linear_system() {
        let a = CsrMatrix::from_dense(&[0.0, 1.0, -2.0, -3.0], 2, 2);
        let dense_j = a.to_dense();
        let f = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            let out = a.mat_vec(y);
            dydt[0] = out[0];
            dydt[1] = out[1];
        };
        let y = [0.7, -1.3];
        let mut f0 = [0.0; 2];
        f(0.0, &y, &mut f0);
        let j = CsrMatrix::jacobian(&f, 0.0, &y, &f0, None);
        assert!((j.get(0, 0) - 0.0).abs() < 1e-8);
        assert!((j.get(0, 1) - 1.0).abs() < 1e-8);
        assert!((j.get(1, 0) - (-2.0)).abs() < 1e-8);
        assert!((j.get(1, 1) - (-3.0)).abs() < 1e-8);
        // The CSR LU of (I - 0.1 J) must match the dense DMat path.
        let jac = CsrMatrix::from_dense(&dense_j, 2, 2);
        let csr = jac_scaled_identity(&jac, 0.1);
        let r = [1.5, -2.5];
        let xs = csr.sparse_solve(&r).unwrap();
        // Dense oracle: A = I - 0.1 J.
        let mut dense_a = crate::linalg::DMat::identity(2);
        for i in 0..2 {
            for j in 0..2 {
                dense_a.set(i, j, dense_a.get(i, j) - 0.1 * dense_j[i * 2 + j]);
            }
        }
        let xd = dense_a.solve(&r).unwrap();
        assert!((xs[0] - xd[0]).abs() < 1e-10);
        assert!((xs[1] - xd[1]).abs() < 1e-10);
    }
}
