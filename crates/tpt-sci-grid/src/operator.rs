use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

use crate::grid::{Boundary, UniformGrid1D, UniformGrid2D};
use crate::stencil::Stencil;

/// Kronecker product `A ⊗ B`.
pub fn kron(a: &DMatrix, b: &DMatrix) -> DMatrix {
    let (p, q) = (a.nrows(), a.ncols());
    let (r, s) = (b.nrows(), b.ncols());
    DMatrix::from_fn(p * r, q * s, |i, j| {
        let ai = i / r;
        let aj = j / s;
        let bi = i % r;
        let bj = j % s;
        a[(ai, aj)] * b[(bi, bj)]
    })
}

/// Assemble the discrete 1-D Laplacian `d^2/dx^2` on a uniform grid.
///
/// With [`Boundary::Dirichlet`] the first and last rows are identity rows, so
/// the linear system `L u = f` enforces `u = 0` on the boundary. With
/// [`Boundary::Neumann`] one-sided stencils give zero-flux boundaries.
pub fn laplacian_1d(grid: &UniformGrid1D, bc: Boundary) -> DMatrix {
    let n = grid.n();
    let dx2 = grid.dx() * grid.dx();
    let is_boundary = |k: usize| k == 0 || k == n - 1;
    DMatrix::from_fn(n, n, |i, j| {
        if bc == Boundary::Dirichlet && is_boundary(i) {
            if i == j { 1.0 } else { 0.0 }
        } else if bc == Boundary::Neumann && i == 0 {
            if j == 0 {
                -2.0 / dx2
            } else if j == 1 {
                2.0 / dx2
            } else {
                0.0
            }
        } else if bc == Boundary::Neumann && i == n - 1 {
            if j == n - 1 {
                -2.0 / dx2
            } else if j == n - 2 {
                2.0 / dx2
            } else {
                0.0
            }
        } else {
            let mut v = 0.0;
            if i == j {
                v -= 2.0 / dx2;
            }
            if i > 0 {
                let left = i - 1;
                if !(bc == Boundary::Dirichlet && is_boundary(left)) && j == left {
                    v += 1.0 / dx2;
                }
            }
            if i + 1 < n {
                let right = i + 1;
                if !(bc == Boundary::Dirichlet && is_boundary(right)) && j == right {
                    v += 1.0 / dx2;
                }
            }
            v
        }
    })
}

/// Assemble the discrete 2-D Laplacian `d^2/dx^2 + d^2/dy^2` on a tensor-product
/// grid, using node ordering `index = ix + iy * nx`.
///
/// # Panics
///
/// Panics if the supplied [`UniformGrid2D`] is internally inconsistent (its
/// constituent 1-D axes fail validation). In practice this never happens
/// because callers always pass a grid that was successfully constructed via
/// [`UniformGrid2D::new`].
pub fn laplacian_2d(grid: &UniformGrid2D, bc: Boundary) -> DMatrix {
    let gx = UniformGrid1D::new(grid.nx(), grid.x0(), grid.x1()).expect("valid x grid");
    let gy = UniformGrid1D::new(grid.ny(), grid.y0(), grid.y1()).expect("valid y grid");
    let dx = laplacian_1d(&gx, bc);
    let dy = laplacian_1d(&gy, bc);
    let ix = DMatrix::from_fn(grid.nx(), grid.nx(), |a, b| if a == b { 1.0 } else { 0.0 });
    let iy = DMatrix::from_fn(grid.ny(), grid.ny(), |a, b| if a == b { 1.0 } else { 0.0 });
    kron(&iy, &dx) + kron(&dy, &ix)
}

/// Assemble a 1-D finite-difference operator from a [`Stencil`].
///
/// Each stencil entry `(offset, coeff)` contributes `coeff / h^order` to the
/// row `i`, column `i + offset`, where `order` is `1` for the first-derivative
/// stencil and `2` for the second-derivative stencil (the `1/h^order` factor
/// follows the standard finite-difference scaling for unit-spaced
/// coefficients).
///
/// Near the boundary, offsets that fall outside `[0, n)` are simply dropped, so
/// the returned operator is exact in the interior and zero-padded at the edges.
/// For interior-only derivative operators on a fixed grid this matches the
/// behaviour of [`laplacian_1d`] away from the boundary.
pub fn derivative_1d(grid: &UniformGrid1D, stencil: Stencil) -> DMatrix {
    let n = grid.n();
    let h = grid.dx();
    let (offsets, coeffs) = stencil.coefficients();
    let order = if stencil == Stencil::CentralFirstDerivative {
        1
    } else {
        2
    };
    let h_pow = h.powi(order);
    DMatrix::from_fn(n, n, |i, j| {
        let needed_off = j as isize - i as isize;
        for (off, c) in offsets.iter().zip(&coeffs) {
            if *off == needed_off {
                return c / h_pow;
            }
        }
        0.0
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

    use super::*;
    use crate::grid::UniformGrid1D;

    #[test]
    fn linspace_basic() {
        let xs = super::super::linspace(0.0_f64, 1.0, 3);
        assert_eq!(xs, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn laplacian_of_quadratic_is_constant() {
        let g = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
        let l = laplacian_1d(&g, Boundary::Dirichlet);
        // u(x) = x^2  ->  u'' = 2
        let u: Vec<f64> = g.coordinates().iter().map(|x| x * x).collect();
        let lu = l * DVector::from_vec(u);
        let mid = g.n() / 2;
        assert_abs_diff_eq!(lu[mid], 2.0, epsilon = 1e-3);
    }

    #[test]
    fn poisson_1d_dirichlet() {
        // -u'' = 1, u(0)=u(1)=0  ->  u(x) = x(1-x)/2, u(0.5) = 0.125
        let g = UniformGrid1D::new(101, 0.0, 1.0).unwrap();
        let l = laplacian_1d(&g, Boundary::Dirichlet);
        let n = g.n();
        let b = DVector::from_vec(
            (0..n)
                .map(|k| if k == 0 || k == n - 1 { 0.0 } else { -1.0 })
                .collect(),
        );
        let u = l.solve(&b).expect("solve");
        let mid = n / 2;
        assert_abs_diff_eq!(u[mid], 0.125, epsilon = 1e-3);
    }

    #[test]
    fn laplacian_2d_of_quadratic_sum() {
        // u(x,y) = x^2 + y^2  ->  Laplacian = 4
        let g = UniformGrid2D::new(11, 0.0, 1.0, 11, 0.0, 1.0).unwrap();
        let l = laplacian_2d(&g, Boundary::Dirichlet);
        let xs = g.x_coordinates();
        let ys = g.y_coordinates();
        let u: Vec<f64> = (0..g.len())
            .map(|k| {
                let ix = k % g.nx();
                let iy = k / g.nx();
                xs[ix] * xs[ix] + ys[iy] * ys[iy]
            })
            .collect();
        let lu = l * DVector::from_vec(u);
        let mid = g.len() / 2;
        assert_abs_diff_eq!(lu[mid], 4.0, epsilon = 1e-2);
    }

    #[test]
    fn kron_identity() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
        let i = DMatrix::from_fn(2, 2, |p, q| if p == q { 1.0 } else { 0.0 });
        let k = kron(&i, &a);
        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 4);
        assert_abs_diff_eq!(k[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(k[(2, 3)], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn derivative_first_of_linear_is_constant() {
        // u(x) = x  ->  u' = 1 everywhere.
        let g = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
        let d = derivative_1d(&g, Stencil::CentralFirstDerivative);
        let u = g.coordinates();
        let du = d * DVector::from_vec(u);
        let mid = g.n() / 2;
        assert_abs_diff_eq!(du[mid], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn derivative_second_of_quadratic_is_constant() {
        // u(x) = x^2  ->  u'' = 2 everywhere.
        let g = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
        let d2 = derivative_1d(&g, Stencil::CentralSecondDerivative);
        let u: Vec<f64> = g.coordinates().iter().map(|x| x * x).collect();
        let d2u = d2 * DVector::from_vec(u);
        let mid = g.n() / 2;
        assert_abs_diff_eq!(d2u[mid], 2.0, epsilon = 1e-3);
    }
}
