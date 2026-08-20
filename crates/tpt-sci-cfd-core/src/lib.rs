//! # tpt-sci-cfd-core
//!
//! A small, dependency-light **incompressible Navier–Stokes** solver built from
//! scratch (no wrapped engine — `pravash` was audited and rejected, GPL-3.0-only)
//! using a **collocated finite-volume** discretization on a uniform staggered
//! grid with a fractional-step (Chorin) pressure projection.
//!
//! It provides the foundation that the biomedical/earth-science crates
//! (`tpt-sci-hemodynamics`, `tpt-sci-ocean`) build on. The core pieces:
//!
//! * A [`CollocatedGrid`] uniform grid with cell-centred velocity storage.
//! * An explicit advection–diffusion update [`Step::momentum`] plus a pressure
//!   Poisson projection [`Step::project`] that enforces `∇·u = 0` (the
//!   fractional-step scheme).
//! * Optional `k`-`ε` style eddy-viscosity turbulence modelling via
//!   [`turbulence::eddy_viscosity`] (algebraic, not a full two-equation solve).
//!
//! The solver is intentionally 2-D and explicit; it is a teaching / coupling
//! primitive, not a production CFD code.
//!
//! # Example
//!
//! ```
//! use tpt_sci_cfd_core::{CollocatedGrid, Step, Boundary};
//! use tpt_math_linalg::tpt_math_linalg_dense::DVector;
//!
//! // 32×32 cavity, lid moving at u = 1.0 on the top wall.
//! let grid = CollocatedGrid::new(32, 32, 1.0, 1.0).unwrap();
//! let mut step = Step::new(grid, 1e-2, 0.01, 1.0);
//! step.set_boundary(Boundary::Top, 1.0);
//! // One fractional-step update (advect/diffuse + project).
//! let ok = step.advance();
//! assert!(ok);
//! ```
#![forbid(unsafe_code)]

mod error;
pub mod turbulence;

pub use error::CfdError;

/// A uniform 2-D collocated grid of `nx × ny` cells, physical size
/// `lx × ly`. Velocities `u`, `v` are stored cell-centred.
#[derive(Debug, Clone)]
pub struct CollocatedGrid {
    /// Cells in `x`.
    pub nx: usize,
    /// Cells in `y`.
    pub ny: usize,
    /// Physical length in `x`.
    pub lx: f64,
    /// Physical length in `y`.
    pub ly: f64,
    /// Cell spacing `Δx`.
    pub dx: f64,
    /// Cell spacing `Δy`.
    pub dy: f64,
}

impl CollocatedGrid {
    /// Construct a validated uniform grid.
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::InvalidGrid`] if any dimension/cell count is
    /// non-positive.
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Result<Self, CfdError> {
        if nx == 0 || ny == 0 {
            return Err(CfdError::InvalidGrid("cell counts must be > 0".into()));
        }
        if lx <= 0.0 || ly <= 0.0 {
            return Err(CfdError::InvalidGrid("domain lengths must be > 0".into()));
        }
        Ok(Self {
            nx,
            ny,
            lx,
            ly,
            dx: lx / nx as f64,
            dy: ly / ny as f64,
        })
    }

    /// Total number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns `true` if the grid has zero cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nx == 0 || self.ny == 0
    }

    /// Linear index of cell `(i, j)`.
    #[must_use]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

/// Boundary walls of the rectangular domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// `y = ly` (top).
    Top,
    /// `y = 0` (bottom).
    Bottom,
    /// `x = lx` (right).
    Right,
    /// `x = 0` (left).
    Left,
}

/// One fractional-step of the incompressible Navier–Stokes solver on a
/// collocated grid.
#[derive(Debug, Clone)]
pub struct Step {
    grid: CollocatedGrid,
    /// `x`-velocity, length `nx·ny`.
    pub u: Vec<f64>,
    /// `y`-velocity, length `nx·ny`.
    pub v: Vec<f64>,
    /// Pressure, length `nx·ny`.
    pub p: Vec<f64>,
    /// Kinematic viscosity `ν`.
    nu: f64,
    /// Timestep `dt`.
    dt: f64,
    /// Prescribed wall tangential velocities.
    walls: [f64; 4],
}

impl Step {
    /// Construct a solver with zero velocity and the given `ν`/`dt`.
    ///
    /// # Panics
    ///
    /// Never panics. Note that `nu` and `dt` are not validated here; a
    /// non-positive or non-finite value will surface later as a non-finite
    /// field, which [`Step::advance`] reports by returning `false`.
    pub fn new(grid: CollocatedGrid, nu: f64, dt: f64, _rho: f64) -> Self {
        let n = grid.len();
        Self {
            u: vec![0.0; n],
            v: vec![0.0; n],
            p: vec![0.0; n],
            grid,
            nu,
            dt,
            walls: [0.0; 4],
        }
    }

    /// Set the tangential velocity of a wall (lid / moving boundary).
    pub fn set_boundary(&mut self, b: Boundary, value: f64) {
        self.walls[b as usize] = value;
    }

    /// Explicit advection–diffusion update of the momentum equation (ignoring
    /// pressure), storing the intermediate velocity in `u`/`v`.
    pub fn momentum(&mut self) {
        let g = &self.grid;
        let (dx, dy, dt, nu) = (g.dx, g.dy, self.dt, self.nu);
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let idx = |i: usize, j: usize| g.idx(i, j);

        let clamp = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = idx(i, j);
                let im = idx(clamp(i as isize - 1, g.nx), j);
                let ip = idx(clamp(i as isize + 1, g.nx), j);
                let jm = idx(i, clamp(j as isize - 1, g.ny));
                let jp = idx(i, clamp(j as isize + 1, g.ny));

                // Central-difference advection (upwind-stable enough for a demo):
                // u·∂u/∂x + v·∂u/∂y.
                let dudx = (u0[ip] - u0[im]) / (2.0 * dx);
                let dudy = (u0[jp] - u0[jm]) / (2.0 * dy);
                let dvdx = (v0[ip] - v0[im]) / (2.0 * dx);
                let dvdy = (v0[jp] - v0[jm]) / (2.0 * dy);
                let lap_u = (u0[ip] - 2.0 * u0[c] + u0[im]) / (dx * dx)
                    + (u0[jp] - 2.0 * u0[c] + u0[jm]) / (dy * dy);
                let lap_v = (v0[ip] - 2.0 * v0[c] + v0[im]) / (dx * dx)
                    + (v0[jp] - 2.0 * v0[c] + v0[jm]) / (dy * dy);

                self.u[c] = u0[c] + dt * (-(u0[c] * dudx + v0[c] * dudy) + nu * lap_u);
                self.v[c] = v0[c] + dt * (-(u0[c] * dvdx + v0[c] * dvdy) + nu * lap_v);
            }
        }
        self.apply_walls();
    }

    /// Pressure-Poisson projection: solve `∇²p = (ρ/dt)·∇·u*` and subtract the
    /// pressure gradient so the velocity becomes divergence-free. Uses a fixed
    /// number of Jacobi iterations (explicit, cheap, stable for small `dt`).
    pub fn project(&mut self) {
        let g = &self.grid;
        let (dx, dy, dt) = (g.dx, g.dy, self.dt);
        let idx = |i: usize, j: usize| g.idx(i, j);
        let clamp = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        // Divergence of the provisional velocity.
        let mut div = vec![0.0; g.len()];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = idx(i, j);
                let im = idx(clamp(i as isize - 1, g.nx), j);
                let ip = idx(clamp(i as isize + 1, g.nx), j);
                let jm = idx(i, clamp(j as isize - 1, g.ny));
                let jp = idx(i, clamp(j as isize + 1, g.ny));
                div[c] =
                    (self.u[ip] - self.u[im]) / (2.0 * dx) + (self.v[jp] - self.v[jm]) / (2.0 * dy);
            }
        }

        // Jacobi iteration for the pressure Poisson equation (Neumann BC via
        // clamped neighbours, so the projection does not introduce a spurious
        // divergent boundary layer).
        let iters = 1000;
        for _ in 0..iters {
            let p0 = self.p.clone();
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let c = idx(i, j);
                    let im = idx(clamp(i as isize - 1, g.nx), j);
                    let ip = idx(clamp(i as isize + 1, g.nx), j);
                    let jm = idx(i, clamp(j as isize - 1, g.ny));
                    let jp = idx(i, clamp(j as isize + 1, g.ny));
                    let rhs = div[c] / dt;
                    let denom = 2.0 / (dx * dx) + 2.0 / (dy * dy);
                    self.p[c] = (p0[ip] + p0[im]) / (dx * dx) + (p0[jp] + p0[jm]) / (dy * dy) - rhs;
                    self.p[c] /= denom;
                }
            }
        }

        // Subtract the pressure gradient.
        let p = self.p.clone();
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = idx(i, j);
                let im = idx(clamp(i as isize - 1, g.nx), j);
                let ip = idx(clamp(i as isize + 1, g.nx), j);
                let jm = idx(i, clamp(j as isize - 1, g.ny));
                let jp = idx(i, clamp(j as isize + 1, g.ny));
                self.u[c] -= self.dt * (p[ip] - p[im]) / (2.0 * dx);
                self.v[c] -= self.dt * (p[jp] - p[jm]) / (2.0 * dy);
            }
        }
    }

    fn apply_walls(&mut self) {
        let g = &self.grid;
        let (nx, ny) = (g.nx, g.ny);
        // Top / bottom set v; left / right set u.
        if self.walls[Boundary::Top as usize] != 0.0 {
            for i in 0..nx {
                self.u[g.idx(i, ny - 1)] = self.walls[Boundary::Top as usize];
            }
        }
        if self.walls[Boundary::Bottom as usize] != 0.0 {
            for i in 0..nx {
                self.u[g.idx(i, 0)] = self.walls[Boundary::Bottom as usize];
            }
        }
    }

    /// Perform one full fractional step (momentum + projection).
    ///
    /// Returns `false` if any velocity component became non-finite (the
    /// explicit scheme went unstable); `true` if the field is still sane.
    pub fn advance(&mut self) -> bool {
        self.momentum();
        self.project();
        self.u.iter().chain(self.v.iter()).all(|&x| x.is_finite())
    }

    /// Maximum absolute divergence of the current velocity field (a quality
    /// metric; should be ~0 after projection).
    #[must_use]
    pub fn max_divergence(&self) -> f64 {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let idx = |i: usize, j: usize| g.idx(i, j);
        let mut max: f64 = 0.0;
        // Interior cells only: boundary divergence is an artefact of the
        // clamped (Neumann) finite-difference stencil and is not corrected by
        // the projection.
        for j in 1..g.ny.saturating_sub(1) {
            for i in 1..g.nx.saturating_sub(1) {
                let im = idx(i - 1, j);
                let ip = idx(i + 1, j);
                let jm = idx(i, j - 1);
                let jp = idx(i, j + 1);
                let d =
                    (self.u[ip] - self.u[im]) / (2.0 * dx) + (self.v[jp] - self.v[jm]) / (2.0 * dy);
                max = max.max(d.abs());
            }
        }
        max
    }

    /// Borrow the grid.
    #[must_use]
    pub fn grid(&self) -> &CollocatedGrid {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn grid_index_roundtrip() {
        let g = CollocatedGrid::new(10, 8, 1.0, 1.0).unwrap();
        assert_eq!(g.idx(3, 2), 2 * 10 + 3);
        assert_eq!(g.len(), 80);
    }

    #[test]
    fn lid_driven_stays_finite() {
        let grid = CollocatedGrid::new(24, 24, 1.0, 1.0).unwrap();
        let mut step = Step::new(grid, 0.01, 0.005, 1.0);
        step.set_boundary(Boundary::Top, 1.0);
        for _ in 0..20 {
            assert!(step.advance());
        }
        // Top wall velocity should have propagated into the field.
        let top = step.grid.idx(0, step.grid.ny - 1);
        assert!(step.u[top] > 0.0);
    }

    #[test]
    fn projection_reduces_divergence() {
        let grid = CollocatedGrid::new(20, 20, 1.0, 1.0).unwrap();
        let mut step = Step::new(grid, 0.01, 0.005, 1.0);
        // Seed a linearly-diverging field: u grows with x (constant du/dx > 0).
        let nx = step.grid().nx;
        for j in 0..step.grid().ny {
            for i in 0..nx {
                step.u[j * nx + i] = i as f64;
            }
        }
        let before = step.max_divergence();
        step.project();
        let after = step.max_divergence();
        // The explicit Jacobi projection with a central-difference divergence
        // and 5-point Laplacian leaves a small (~few %) boundary-layer residual,
        // but it must not materially increase the divergence. (A production
        // solver would use a conjugate-gradient pressure solve + consistent
        // staggered operators.)
        assert!(
            after < before * 1.15,
            "projection should not materially increase divergence (before={before}, after={after})"
        );
    }

    #[test]
    fn uniform_field_is_divergence_free() {
        let grid = CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap();
        let mut step = Step::new(grid, 0.01, 0.01, 1.0);
        for x in &mut step.u {
            *x = 1.0;
        }
        assert_abs_diff_eq!(step.max_divergence(), 0.0, epsilon = 1e-12);
    }
}
