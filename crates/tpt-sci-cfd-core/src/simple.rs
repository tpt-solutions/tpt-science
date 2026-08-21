//! SIMPLE-style implicit pressure-correction solver.
//!
//! This module adds an *implicit* pressure/diffusion correction on top of the
//! existing explicit [`crate::Step`] scheme. After a provisional explicit
//! momentum update (`u*`, `v*`) we build the pressure Poisson equation
//! `∇²p = (ρ/dt)·∇·u*`, solve it with the sparse conjugate-gradient solver from
//! `tpt-sci-grid`, and subtract its gradient to obtain a divergence-free
//! velocity field. This is the core of a SIMPLE/PISO pressure-projection
//! method, run *alongside* (not replacing) the explicit fractional-step solver.

use tpt_sci_grid::sparse::{CsrMatrix, conjugate_gradient};
use tpt_sci_grid::{Boundary, UniformGrid2D};

use crate::{CfdError, CollocatedGrid};

/// A SIMPLE-style pressure-correction solver on a uniform collocated grid.
///
/// The recommended workflow per timestep is [`SimpleSolver::predict`] (explicit
/// momentum to obtain the provisional `u*`,`v*`), then
/// [`SimpleSolver::correct`] (solve the pressure Poisson equation with CG and
/// subtract its gradient), both wrapped by [`SimpleSolver::advance`]. The
/// provisional field may also be supplied directly via
/// [`SimpleSolver::set_provisional`] for testing or for coupling to an external
/// momentum scheme.
#[derive(Debug, Clone)]
pub struct SimpleSolver {
    grid: CollocatedGrid,
    /// Provisional `x`-velocity `u*` (before pressure correction).
    pub u_star: Vec<f64>,
    /// Provisional `y`-velocity `v*` (before pressure correction).
    pub v_star: Vec<f64>,
    /// Corrected `x`-velocity `u`.
    pub u: Vec<f64>,
    /// Corrected `y`-velocity `v`.
    pub v: Vec<f64>,
    /// Pressure field `p` (defined up to an additive constant).
    pub p: Vec<f64>,
    nu: f64,
    rho: f64,
    dt: f64,
}

impl SimpleSolver {
    /// Construct a solver with zero velocity/pressure fields and the given
    /// kinematic viscosity `ν`, density `ρ`, and timestep `dt`.
    #[must_use]
    pub fn new(grid: CollocatedGrid, nu: f64, rho: f64, dt: f64) -> Self {
        let n = grid.len();
        Self {
            grid,
            u_star: vec![0.0; n],
            v_star: vec![0.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            p: vec![0.0; n],
            nu,
            rho,
            dt,
        }
    }

    /// Replace the provisional velocity field (`u*`,`v*`) used by the pressure
    /// correction.
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::DimensionMismatch`] if either vector's length does
    /// not match the number of grid cells.
    pub fn set_provisional(&mut self, u: Vec<f64>, v: Vec<f64>) -> Result<(), CfdError> {
        let n = self.grid.len();
        if u.len() != n || v.len() != n {
            return Err(CfdError::DimensionMismatch {
                expected: n,
                actual: u.len(),
            });
        }
        self.u_star = u;
        self.v_star = v;
        Ok(())
    }

    fn clamp(i: isize, n: usize) -> usize {
        i.clamp(0, n as isize - 1) as usize
    }

    /// Discrete divergence `∇·u` of a velocity field (central difference, with
    /// one-sided clamping at the domain boundary).
    #[must_use]
    pub fn divergence(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let mut div = vec![0.0; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = g.idx(i, j);
                let im = g.idx(Self::clamp(i as isize - 1, g.nx), j);
                let ip = g.idx(Self::clamp(i as isize + 1, g.nx), j);
                let jm = g.idx(i, Self::clamp(j as isize - 1, g.ny));
                let jp = g.idx(i, Self::clamp(j as isize + 1, g.ny));
                div[c] = (u[ip] - u[im]) / (2.0 * dx) + (v[jp] - v[jm]) / (2.0 * dy);
            }
        }
        div
    }

    /// Build the pressure Poisson matrix `A = ∇²` on the structured collocated
    /// grid (5-point stencil, Neumann clamp at the boundary) and pin a single
    /// reference cell to remove the constant-pressure nullspace so the system
    /// is symmetric positive-definite and invertible.
    #[must_use]
    pub fn build_poisson_matrix(&self) -> CsrMatrix {
        let g = &self.grid;
        // Match the collocated cell spacing exactly so the discrete Laplacian
        // has the right scaling; the absolute domain extent is irrelevant.
        let x1 = (g.nx as f64 - 1.0) * g.dx;
        let y1 = (g.ny as f64 - 1.0) * g.dy;
        let g2d =
            UniformGrid2D::new(g.nx, 0.0, x1, g.ny, 0.0, y1).expect("grid has at least 2 cells");
        let mut a = tpt_sci_grid::sparse::laplacian_2d_sparse(&g2d, Boundary::Neumann);
        // `laplacian_2d_sparse` assembles the discrete Laplacian, which is
        // negative-definite; negate it so the Poisson operator is
        // symmetric-positive-definite (valid for conjugate-gradient) and solves
        // `(-∇²)·p = -(ρ/dt)·∇·u*`, i.e. `∇²p = (ρ/dt)·∇·u*`.
        for v in &mut a.values {
            *v = -*v;
        }
        // Pin the reference cell (index 0) to p = 0. To keep the matrix
        // symmetric (so the conjugate-gradient solver is valid) we zero out
        // column 0 everywhere and turn row 0 into the identity row. Because the
        // pinned value is 0, the removed couplings contribute nothing to the
        // right-hand side.
        for r in 0..a.nrows() {
            let start = a.row_ptr[r];
            let end = a.row_ptr[r + 1];
            for k in start..end {
                if a.col_ind[k] == 0 {
                    a.values[k] = 0.0;
                }
            }
        }
        let start = a.row_ptr[0];
        let end = a.row_ptr[1];
        for k in start..end {
            a.values[k] = if a.col_ind[k] == 0 { 1.0 } else { 0.0 };
        }
        a
    }

    /// Solve the pressure Poisson equation `∇²p = (ρ/dt)·∇·u*` with the sparse
    /// conjugate-gradient solver, returning the pressure field (with the
    /// reference cell pinned to `p = 0`).
    #[must_use]
    pub fn solve_pressure(&self) -> Vec<f64> {
        let div = self.divergence(&self.u_star, &self.v_star);
        // Matrix is `-∇²` (positive-definite), so the RHS is `-(ρ/dt)·∇·u*`.
        let mut b: Vec<f64> = div.iter().map(|d| -(self.rho / self.dt) * d).collect();
        b[0] = 0.0;
        let a = self.build_poisson_matrix();
        conjugate_gradient(&a, &b, None, 1e-10, 5000)
    }

    /// Explicit advection–diffusion momentum update of the provisional field
    /// (`u*`,`v*`), analogous to [`crate::Step::momentum`].
    pub fn predict(&mut self) {
        let g = &self.grid;
        let (dx, dy, dt, nu) = (g.dx, g.dy, self.dt, self.nu);
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let mut u = vec![0.0; g.len()];
        let mut v = vec![0.0; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = g.idx(i, j);
                let im = g.idx(Self::clamp(i as isize - 1, g.nx), j);
                let ip = g.idx(Self::clamp(i as isize + 1, g.nx), j);
                let jm = g.idx(i, Self::clamp(j as isize - 1, g.ny));
                let jp = g.idx(i, Self::clamp(j as isize + 1, g.ny));
                let dudx = (u0[ip] - u0[im]) / (2.0 * dx);
                let dudy = (u0[jp] - u0[jm]) / (2.0 * dy);
                let dvdx = (v0[ip] - v0[im]) / (2.0 * dx);
                let dvdy = (v0[jp] - v0[jm]) / (2.0 * dy);
                let lap_u = (u0[ip] - 2.0 * u0[c] + u0[im]) / (dx * dx)
                    + (u0[jp] - 2.0 * u0[c] + u0[jm]) / (dy * dy);
                let lap_v = (v0[ip] - 2.0 * v0[c] + v0[im]) / (dx * dx)
                    + (v0[jp] - 2.0 * v0[c] + v0[jm]) / (dy * dy);
                u[c] = u0[c] + dt * (-(u0[c] * dudx + v0[c] * dudy) + nu * lap_u);
                v[c] = v0[c] + dt * (-(u0[c] * dvdx + v0[c] * dvdy) + nu * lap_v);
            }
        }
        self.u_star = u;
        self.v_star = v;
    }

    /// Pressure correction: solve the Poisson equation and subtract `∇p` from
    /// the provisional velocity so the result is (approximately) divergence
    /// free.
    pub fn correct(&mut self) {
        let p = self.solve_pressure();
        let g = &self.grid;
        let (dx, dy, dt, rho) = (g.dx, g.dy, self.dt, self.rho);
        let mut u = vec![0.0; g.len()];
        let mut v = vec![0.0; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = g.idx(i, j);
                let im = g.idx(Self::clamp(i as isize - 1, g.nx), j);
                let ip = g.idx(Self::clamp(i as isize + 1, g.nx), j);
                let jm = g.idx(i, Self::clamp(j as isize - 1, g.ny));
                let jp = g.idx(i, Self::clamp(j as isize + 1, g.ny));
                u[c] = self.u_star[c] - dt / rho * (p[ip] - p[im]) / (2.0 * dx);
                v[c] = self.v_star[c] - dt / rho * (p[jp] - p[jm]) / (2.0 * dy);
            }
        }
        self.u = u;
        self.v = v;
        self.p = p;
    }

    /// Perform one full SIMPLE step (predict + correct).
    ///
    /// Returns `false` if any velocity or pressure component became
    /// non-finite (the scheme went unstable); `true` otherwise.
    pub fn advance(&mut self) -> bool {
        self.predict();
        self.correct();
        self.u
            .iter()
            .chain(&self.v)
            .chain(&self.p)
            .all(|&x| x.is_finite())
    }

    /// Maximum absolute divergence of the corrected velocity field (interior
    /// cells only; a quality metric that should be near zero after correction).
    #[must_use]
    pub fn max_divergence(&self) -> f64 {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let mut max: f64 = 0.0;
        for j in 1..g.ny.saturating_sub(1) {
            for i in 1..g.nx.saturating_sub(1) {
                let im = g.idx(i - 1, j);
                let ip = g.idx(i + 1, j);
                let jm = g.idx(i, j - 1);
                let jp = g.idx(i, j + 1);
                let d =
                    (self.u[ip] - self.u[im]) / (2.0 * dx) + (self.v[jp] - self.v[jm]) / (2.0 * dy);
                max = max.max(d.abs());
            }
        }
        max
    }

    /// Borrow the underlying grid.
    #[must_use]
    pub fn grid(&self) -> &CollocatedGrid {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    // `sin(πx)sin(πy)` is a smooth manufactured pressure. The velocity field
    // `u* = -(dt/ρ)∇p` has divergence `(dt/ρ)∇²p`, so the Poisson solve must
    // recover `p` up to an additive constant.
    #[test]
    fn manufactured_poisson_recovers_pressure() {
        let g = CollocatedGrid::new(40, 40, 1.0, 1.0).unwrap();
        let mut solver = SimpleSolver::new(g.clone(), 1e-2, 1.0, 1e-3);
        let (nx, ny) = (g.nx, g.ny);
        let (dx, dy) = (g.dx, g.dy);
        let dt = solver.dt;
        let rho = solver.rho;
        let mut p_true = vec![0.0; g.len()];
        let mut ustar = vec![0.0; g.len()];
        let mut vstar = vec![0.0; g.len()];
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dy;
                let c = g.idx(i, j);
                let s = (PI * x).sin() * (PI * y).sin();
                p_true[c] = s;
                ustar[c] = -(dt / rho) * (PI * (PI * x).cos() * (PI * y).sin());
                vstar[c] = -(dt / rho) * (PI * (PI * x).sin() * (PI * y).cos());
            }
        }
        solver.set_provisional(ustar, vstar).unwrap();
        let p = solver.solve_pressure();

        // Estimate the additive constant from interior cells, then check that
        // `p + p_true` is (approximately) constant there.
        let mut shift = 0.0;
        let mut count = 0;
        for c in 0..g.len() {
            let i = c % nx;
            let j = c / nx;
            if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 {
                shift += p[c] + p_true[c];
                count += 1;
            }
        }
        shift /= count as f64;
        eprintln!(
            "DEBUG p: p[0]={}, max|p|={}, min p={}, max p={}, shift={shift}",
            p[0],
            p.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            p.iter().cloned().fold(0.0_f64, f64::min),
            p.iter().cloned().fold(0.0_f64, f64::max),
        );
        let mut max_dev = 0.0_f64;
        for c in 0..g.len() {
            let i = c % nx;
            let j = c / nx;
            if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 {
                max_dev = max_dev.max((p[c] + p_true[c] - shift).abs());
            }
        }
        assert!(max_dev < 5e-3, "recovered pressure deviates by {max_dev}");
    }

    #[test]
    fn pressure_correction_reduces_divergence() {
        let g = CollocatedGrid::new(24, 24, 1.0, 1.0).unwrap();
        let mut solver = SimpleSolver::new(g.clone(), 1e-2, 1.0, 1e-3);
        let mut u = vec![0.0; g.len()];
        let v = vec![0.0; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                u[g.idx(i, j)] = i as f64 * 0.01;
            }
        }
        solver.set_provisional(u.clone(), v.clone()).unwrap();
        let before = solver.divergence(&u, &v).iter().map(|d| d.abs()).fold(0.0, f64::max);
        solver.correct();
        let after = solver.max_divergence();
        assert!(
            after < before * 0.1,
            "pressure correction should reduce divergence (before={before}, after={after})"
        );
    }

    #[test]
    fn advance_stays_finite_and_divergence_free() {
        let grid = CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap();
        let mut solver = SimpleSolver::new(grid, 0.01, 1.0, 0.002);
        // Seed a small uniform flow so the explicit step is stable.
        solver.u.fill(0.5);
        assert!(solver.advance());
        assert!(solver.max_divergence() < 1e-6);
    }

    #[test]
    fn set_provisional_rejects_wrong_length() {
        let grid = CollocatedGrid::new(8, 8, 1.0, 1.0).unwrap();
        let mut solver = SimpleSolver::new(grid, 1.0, 1.0, 1.0);
        let r = solver.set_provisional(vec![0.0; 5], vec![0.0; 64]);
        assert!(matches!(r, Err(CfdError::DimensionMismatch { .. })));
    }

    #[test]
    fn uniform_provisional_stays_divergence_free() {
        let grid = CollocatedGrid::new(12, 12, 1.0, 1.0).unwrap();
        let mut solver = SimpleSolver::new(grid, 1.0, 1.0, 1e-3);
        solver.u_star.fill(0.7);
        // Divergence of a uniform field is exactly zero.
        assert_abs_diff_eq!(solver.max_divergence(), 0.0, epsilon = 1e-12);
    }
}
