//! 3-D atmospheric chemistry / transport on a structured grid.
//!
//! Extends the 0-D [`crate::ChemistryBox`] to a 3-D tracer field advected by a
//! wind and mixed by a constant-coefficient diffusion operator on a
//! [`tpt_sci_grid::UniformGrid3D`]. The concentration `c(x,y,z,t)` obeys the
//! advection–diffusion–reaction equation
//!
//! ```text
//! ∂c/∂t = −u·∇c + κ·∇²c + P − k·c ,
//! ```
//!
//! solved by explicit method-of-lines (upwind advection + the sparse 3-D
//! Laplacian from `tpt-sci-grid`). Neumann (no-flux) boundary conditions are
//! applied via clamped neighbours.

use tpt_sci_grid::{Boundary, CsrMatrix, UniformGrid3D, laplacian_3d_sparse};

use crate::error::ClimateError;

/// A 3-D atmospheric tracer (e.g. CH₄, OH, aerosol) on a uniform tensor grid.
#[derive(Debug, Clone)]
pub struct Tracer3D {
    grid: UniformGrid3D,
    /// Concentration `c` (arbitrary units), flat length `nx·ny·nz` in the
    /// `ix + iy·nx + iz·nx·ny` ordering.
    pub conc: Vec<f64>,
    /// `x`-advection velocity `u` (m/s).
    pub u: Vec<f64>,
    /// `y`-advection velocity `v` (m/s).
    pub v: Vec<f64>,
    /// `z`-advection velocity `w` (m/s).
    pub w: Vec<f64>,
    /// Constant isotropic diffusion coefficient `κ` (m²/s).
    pub diffusion: f64,
    /// Per-cell production rate `P` (units/s).
    pub production: Vec<f64>,
    /// First-order loss coefficient `k` (1/s).
    pub loss: f64,
    lap: CsrMatrix,
}

impl Tracer3D {
    /// Construct a 3-D tracer field.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if `u`/`v`/`w`/`production` lengths
    /// disagree with the grid node count, if `diffusion < 0`, or if `loss < 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: UniformGrid3D,
        conc: Vec<f64>,
        u: Vec<f64>,
        v: Vec<f64>,
        w: Vec<f64>,
        diffusion: f64,
        production: Vec<f64>,
        loss: f64,
    ) -> Result<Self, ClimateError> {
        let n = grid.len();
        if conc.len() != n || u.len() != n || v.len() != n || w.len() != n || production.len() != n {
            return Err(ClimateError::InvalidModel(
                "velocity/production/conc length must equal grid nodes".into(),
            ));
        }
        if diffusion < 0.0 {
            return Err(ClimateError::InvalidModel("diffusion must be >= 0".into()));
        }
        if loss < 0.0 {
            return Err(ClimateError::InvalidModel("loss must be >= 0".into()));
        }
        let lap = laplacian_3d_sparse(&grid, Boundary::Neumann);
        Ok(Self {
            grid,
            conc,
            u,
            v,
            w,
            diffusion,
            production,
            loss,
            lap,
        })
    }

    /// Lexicographic node index `ix + iy·nx + iz·nx·ny`.
    #[must_use]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        self.grid.index(ix, iy, iz)
    }

    /// Advance the tracer field by `dt` seconds with one explicit Euler step of
    /// the advection–diffusion–reaction equation.
    ///
    /// # Panics
    ///
    /// Panics if `diffusion·dt/dx²` exceeds `0.5` (the explicit diffusion
    /// stability limit), which would make the step unstable.
    pub fn step(&mut self, dt: f64) {
        let (dx, dy, dz) = (self.grid.dx(), self.grid.dy(), self.grid.dz());
        let fac = self.diffusion * dt;
        let limit = 0.5 * (dx * dx).min(dy * dy).min(dz * dz);
        if fac > limit {
            panic!(
                "explicit diffusion unstable: kappa*dt = {fac} exceeds limit {limit}; reduce dt or diffusion"
            );
        }
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let c0 = self.conc.clone();
        let lap_c = self.lap.mul_vec(&c0);
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        let mut next = vec![0.0f64; c0.len()];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let im = clamp(ix as isize - 1, nx);
                    let ip = clamp(ix as isize + 1, nx);
                    let jm = clamp(iy as isize - 1, ny);
                    let jp = clamp(iy as isize + 1, ny);
                    let km = clamp(iz as isize - 1, nz);
                    let kp = clamp(iz as isize + 1, nz);
                    // Upwind advection in each direction.
                    let dcdy = if self.u[c] >= 0.0 {
                        (c0[c] - c0[im + iy * nx + iz * nx * ny]) / dx
                    } else {
                        (c0[ip + iy * nx + iz * nx * ny] - c0[c]) / dx
                    };
                    let dcdv = if self.v[c] >= 0.0 {
                        (c0[c] - c0[ix + jm * nx + iz * nx * ny]) / dy
                    } else {
                        (c0[ix + jp * nx + iz * nx * ny] - c0[c]) / dy
                    };
                    let dcdw = if self.w[c] >= 0.0 {
                        (c0[c] - c0[ix + iy * nx + km * nx * ny]) / dz
                    } else {
                        (c0[ix + iy * nx + kp * nx * ny] - c0[c]) / dz
                    };
                    let adv = self.u[c] * dcdy + self.v[c] * dcdv + self.w[c] * dcdw;
                    let diff = self.diffusion * lap_c[c];
                    let react = self.production[c] - self.loss * c0[c];
                    next[c] = c0[c] + dt * (-adv + diff + react);
                }
            }
        }
        self.conc = next;
    }

    /// Domain-integrated total mass `M = Σ c·dx·dy·dz`.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        let dv = self.grid.dx() * self.grid.dy() * self.grid.dz();
        self.conc.iter().sum::<f64>() * dv
    }

    /// Domain-mean concentration (per unit volume).
    #[must_use]
    pub fn mean_concentration(&self) -> f64 {
        if self.conc.is_empty() {
            0.0
        } else {
            self.conc.iter().sum::<f64>() / self.conc.len() as f64
        }
    }

    /// Borrow the concentration field.
    #[must_use]
    pub fn concentration(&self) -> &[f64] {
        &self.conc
    }

    /// Borrow the underlying grid.
    #[must_use]
    pub fn grid(&self) -> &UniformGrid3D {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn grid() -> UniformGrid3D {
        UniformGrid3D::new(11, 0.0, 1.0, 11, 0.0, 1.0, 9, 0.0, 1.0).unwrap()
    }

    #[test]
    fn uniform_field_stays_uniform_without_flow() {
        let g = grid();
        let n = g.len();
        let mut t = Tracer3D::new(
            g,
            vec![1.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            0.0,
            vec![0.0; n],
            0.0,
        )
        .unwrap();
        t.step(0.01);
        assert!(t.conc.iter().all(|&x| (x - 1.0).abs() < 1e-12));
        assert_abs_diff_eq!(t.mean_concentration(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn diffusion_smooths_a_gradient() {
        // No flow, no reaction; a central bump must spread under diffusion.
        let g = grid();
        let n = g.len();
        let mut conc = vec![0.0; n];
        let prod = vec![0.0; n];
        conc[g.index(5, 5, 4)] = 1.0;
        let mut t = Tracer3D::new(
            g,
            conc,
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            0.02,
            prod.clone(),
            0.0,
        )
        .unwrap();
        let peak0 = t.conc[t.index(5, 5, 4)];
        let mut spread0 = 0usize;
        for v in &t.conc {
            if *v > 1e-3 {
                spread0 += 1;
            }
        }
        for _ in 0..10 {
            t.step(0.02);
        }
        let peak1 = t.conc[t.index(5, 5, 4)];
        let mut spread1 = 0usize;
        for v in &t.conc {
            if *v > 1e-3 {
                spread1 += 1;
            }
        }
        assert!(peak1 < peak0, "peak should decay under diffusion");
        assert!(spread1 >= spread0, "mass should spread to more nodes");
        assert!(t.total_mass() > 0.0 && t.total_mass().is_finite());
    }

    #[test]
    fn production_loss_reaches_steady_state() {
        let g = grid();
        let n = g.len();
        let mut t = Tracer3D::new(
            g,
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            0.0,
            vec![1.0; n], // production P = 1
            0.1,          // loss k = 0.1
        )
        .unwrap();
        for _ in 0..2000 {
            t.step(0.1);
        }
        // Steady state is P/k = 10 everywhere.
        assert_abs_diff_eq!(t.mean_concentration(), 10.0, epsilon = 1e-2);
    }
}
