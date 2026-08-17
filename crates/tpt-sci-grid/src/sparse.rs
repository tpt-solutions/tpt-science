//! Feature-gated sparse-matrix backend for large PDE grids.
//!
//! Enabled by the `sparse` cargo feature. Provides a minimal
//! [compressed-sparse-row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
//! (`CsrMatrix`) alongside sparse Laplacian assemblers. This is additive: the
//! dense operators in [`crate::operator`] remain the default; the CSR variants
//! here let callers work with realistically-sized 1-D/2-D grids without the
//! O(n²) memory cost of a dense [`DMatrix`](tpt_math_linalg::tpt_math_linalg_dense::DMatrix).

use crate::grid::{Boundary, UniformGrid1D, UniformGrid2D};

/// A real, compressed-sparse-row (CSR) matrix.
///
/// `row_ptr` has length `nrows + 1`; the entries for row `i` are stored in
/// `col_ind[row_ptr[i]..row_ptr[i+1]]` with matching values in `values`.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    /// `row_ptr[i]` is the start offset of row `i` in the column/value arrays;
    /// `row_ptr[nrows]` is the total number of stored entries.
    pub row_ptr: Vec<usize>,
    /// Column indices, parallel to `values`.
    pub col_ind: Vec<usize>,
    /// Stored values, parallel to `col_ind`.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Build a CSR matrix from a dense [`DMatrix`](tpt_math_linalg::tpt_math_linalg_dense::DMatrix),
    /// dropping (near-)zero entries to keep it sparse.
    #[must_use]
    pub fn from_dense(m: &tpt_math_linalg::tpt_math_linalg_dense::DMatrix<f64>) -> Self {
        let nrows = m.nrows();
        let ncols = m.ncols();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        let mut offset = 0;
        row_ptr.push(0);
        for i in 0..nrows {
            for j in 0..ncols {
                let v = m[(i, j)];
                if v != 0.0 {
                    col_ind.push(j);
                    values.push(v);
                    offset += 1;
                }
            }
            row_ptr.push(offset);
        }
        Self {
            nrows,
            ncols,
            row_ptr,
            col_ind,
            values,
        }
    }

    /// Sparse matrix–vector product `y = A·x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.ncols()` (the dense `DMatrix` mat-vec would
    /// likewise panic on a dimension mismatch).
    #[must_use]
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols(), "vector length must match ncols");
        let mut y = vec![0.0; self.nrows];
        for (i, slot) in y.iter_mut().enumerate() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut acc = 0.0;
            for k in start..end {
                acc += self.values[k] * x[self.col_ind[k]];
            }
            *slot = acc;
        }
        y
    }
}

/// Assemble the discrete 1-D Laplacian as a [`CsrMatrix`] (mirrors
/// [`crate::laplacian_1d`]). See that function for the boundary-condition
/// treatment.
#[must_use]
pub fn laplacian_1d_sparse(grid: &UniformGrid1D, bc: Boundary) -> CsrMatrix {
    let n = grid.n();
    let dx2 = grid.dx() * grid.dx();
    let is_boundary = |k: usize| k == 0 || k == n - 1;

    // Collect the (col, value) entries per row first, then pack into CSR.
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, row) in rows.iter_mut().enumerate() {
        if bc == Boundary::Dirichlet && is_boundary(i) {
            row.push((i, 1.0));
            continue;
        }
        if bc == Boundary::Neumann && i == 0 {
            row.push((0, -2.0 / dx2));
            row.push((1, 2.0 / dx2));
            continue;
        }
        if bc == Boundary::Neumann && i == n - 1 {
            row.push((n - 1, -2.0 / dx2));
            row.push((n - 2, 2.0 / dx2));
            continue;
        }
        if i > 0 {
            let left = i - 1;
            if !(bc == Boundary::Dirichlet && is_boundary(left)) {
                row.push((left, 1.0 / dx2));
            }
        }
        row.push((i, -2.0 / dx2));
        if i + 1 < n {
            let right = i + 1;
            if !(bc == Boundary::Dirichlet && is_boundary(right)) {
                row.push((right, 1.0 / dx2));
            }
        }
    }

    pack_csr(n, n, &rows)
}

/// Assemble the discrete 2-D Laplacian (`d²/dx² + d²/dy²`) on a tensor-product
/// grid as a [`CsrMatrix`], using node ordering `index = ix + iy * nx` and the
/// standard 5-point stencil. Mirrors [`crate::laplacian_2d`] but without forming
/// a dense `n² × n²` matrix.
#[must_use]
pub fn laplacian_2d_sparse(grid: &UniformGrid2D, bc: Boundary) -> CsrMatrix {
    let nx = grid.nx();
    let ny = grid.ny();
    let n = nx * ny;
    let dx2 = grid.dx() * grid.dx();
    let dy2 = grid.dy() * grid.dy();
    let idx = |ix: usize, iy: usize| ix + iy * nx;

    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for iy in 0..ny {
        for ix in 0..nx {
            let i = idx(ix, iy);
            let on_boundary = ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1;
            if bc == Boundary::Dirichlet && on_boundary {
                rows[i].push((i, 1.0));
                continue;
            }
            // x-direction neighbours (Neumann clamps off-boundary to the node).
            let xm = if ix > 0 { ix - 1 } else { ix };
            let xp = if ix + 1 < nx { ix + 1 } else { ix };
            if bc == Boundary::Neumann || ix > 0 {
                rows[i].push((idx(xm, iy), 1.0 / dx2));
            }
            rows[i].push((i, -2.0 / dx2));
            if bc == Boundary::Neumann || ix + 1 < nx {
                rows[i].push((idx(xp, iy), 1.0 / dx2));
            }
            // y-direction neighbours.
            let ym = if iy > 0 { iy - 1 } else { iy };
            let yp = if iy + 1 < ny { iy + 1 } else { iy };
            if bc == Boundary::Neumann || iy > 0 {
                rows[i].push((idx(ix, ym), 1.0 / dy2));
            }
            rows[i].push((i, -2.0 / dy2));
            if bc == Boundary::Neumann || iy + 1 < ny {
                rows[i].push((idx(ix, yp), 1.0 / dy2));
            }
        }
    }

    pack_csr(n, n, &rows)
}

/// One explicit-Euler diffusion step `u_next = u + dt · D · (L · u)` for a field
/// `u` on a grid with diffusion coefficient `D` and Laplacian `L`.
///
/// The Laplacian is applied via the sparse mat-vec [`CsrMatrix::mul_vec`], so the
/// cost scales with the number of non-zeros rather than `n²`.
#[must_use]
pub fn diffuse_step(u: &[f64], laplacian: &CsrMatrix, dt: f64, diffusion: f64) -> Vec<f64> {
    let lap = laplacian.mul_vec(u);
    u.iter()
        .zip(&lap)
        .map(|(u_i, lap_i)| u_i + dt * diffusion * lap_i)
        .collect()
}

/// Pack per-row `(col, value)` lists into CSR storage.
fn pack_csr(nrows: usize, ncols: usize, rows: &[Vec<(usize, f64)>]) -> CsrMatrix {
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();
    let mut offset = 0;
    row_ptr.push(0);
    for r in rows {
        for &(c, v) in r {
            col_ind.push(c);
            values.push(v);
            offset += 1;
        }
        row_ptr.push(offset);
    }
    CsrMatrix {
        nrows,
        ncols,
        row_ptr,
        col_ind,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_linalg::tpt_math_linalg_dense::DVector;

    #[test]
    fn sparse_1d_matches_dense_laplacian() {
        let g = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
        let sparse = laplacian_1d_sparse(&g, Boundary::Dirichlet);
        let dense = crate::laplacian_1d(&g, Boundary::Dirichlet);
        for i in 0..g.n() {
            let start = sparse.row_ptr[i];
            let end = sparse.row_ptr[i + 1];
            for k in start..end {
                assert_abs_diff_eq!(
                    sparse.values[k],
                    dense[(i, sparse.col_ind[k])],
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn sparse_mul_vec_matches_dense() {
        let g = UniformGrid1D::new(11, 0.0, 1.0).unwrap();
        let sparse = laplacian_1d_sparse(&g, Boundary::Dirichlet);
        let dense = crate::laplacian_1d(&g, Boundary::Dirichlet);
        let u: Vec<f64> = (0..g.n()).map(|i| i as f64 * 0.1).collect();
        let y_sparse = sparse.mul_vec(&u);
        let y_dense = dense * DVector::from_vec(u);
        for i in 0..g.n() {
            assert_abs_diff_eq!(y_sparse[i], y_dense[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn sparse_2d_laplacian_of_quadratic() {
        // u = (x(1-x) + y(1-y))/4  ->  ∇²u = -1  ->  L·u ≈ -1 on the interior.
        let g = UniformGrid2D::new(41, 0.0, 1.0, 41, 0.0, 1.0).unwrap();
        let l = laplacian_2d_sparse(&g, Boundary::Dirichlet);
        let xs = g.x_coordinates();
        let ys = g.y_coordinates();
        let u: Vec<f64> = (0..g.len())
            .map(|k| {
                let ix = k % g.nx();
                let iy = k / g.nx();
                (xs[ix] * (1.0 - xs[ix]) + ys[iy] * (1.0 - ys[iy])) / 4.0
            })
            .collect();
        let lu = l.mul_vec(&u);
        let mid = g.len() / 2;
        assert_abs_diff_eq!(lu[mid], -1.0, epsilon = 2e-2);
    }

    #[test]
    fn diffuse_step_reduces_gradient() {
        let g = UniformGrid1D::new(51, 0.0, 1.0).unwrap();
        let l = laplacian_1d_sparse(&g, Boundary::Dirichlet);
        // A parabola pinned to zero at the boundaries: diffusion flattens it.
        let u: Vec<f64> = g.coordinates().iter().map(|&x| x * (1.0 - x)).collect();
        let u1 = diffuse_step(&u, &l, 0.0005, 1.0);
        let grad0: f64 = u.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let grad1: f64 = u1.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(grad1 < grad0, "diffusion should smooth the parabola");
    }
}
