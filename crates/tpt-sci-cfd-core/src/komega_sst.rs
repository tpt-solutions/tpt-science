//! Two-equation `k`-`ω` SST turbulence closure (Menter, 2003).
//!
//! This implements the Menter shear-stress-transport (SST) model: transport
//! equations for the turbulent kinetic energy `k` and the specific dissipation
//! rate `ω`, with the SST blend of the `k-ω` (near-wall) and `k-ε` (far-field)
//! branches via the `F1`/`F2` blending functions, the `a1` production
//! limiter, and the `k-ω`/`k-ε` cross-diffusion term in the `ω` equation. It
//! is self-contained and runs *alongside* (not replacing) the algebraic
//! [`crate::turbulence::eddy_viscosity`] already in the crate.
//!
//! The model is advanced explicitly on a [`crate::CollocatedGrid`]; it needs
//! the mean velocity (`u`,`v`) to evaluate the strain-rate and vorticity
//! tensors and a per-cell wall distance for the blending functions.

use crate::CollocatedGrid;

#[derive(Debug, Clone)]
pub struct KOmegaSst {
    grid: CollocatedGrid,
    /// Turbulent kinetic energy `k` (per cell).
    pub k: Vec<f64>,
    /// Specific dissipation rate `ω` (per cell).
    pub omega: Vec<f64>,
    /// Mean `x`-velocity (drives production, advection, vorticity).
    pub u: Vec<f64>,
    /// Mean `y`-velocity.
    pub v: Vec<f64>,
    wall_distance: Vec<f64>,
    nu: f64,
    // --- Menter (2003) closure constants ---
    beta_star: f64,
    a1: f64,
    sigma_k1: f64,
    sigma_k2: f64,
    sigma_omega1: f64,
    sigma_omega2: f64,
    beta1: f64,
    beta2: f64,
    alpha1: f64,
    alpha2: f64,
}

impl KOmegaSst {
    /// Construct the model on `grid` with molecular viscosity `ν`, initialising
    /// `k = 1e-3` and `ω = 1` everywhere and computing the wall distance to the
    /// nearest domain boundary.
    #[must_use]
    pub fn new(grid: CollocatedGrid, nu: f64) -> Self {
        let n = grid.len();
        let kappa = 0.41_f64;
        let sqrt_beta_star = 0.3_f64; // sqrt(0.09)
        let mut s = Self {
            grid,
            k: vec![1e-3; n],
            omega: vec![1.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            wall_distance: vec![1.0; n],
            nu,
            beta_star: 0.09,
            a1: 0.31,
            sigma_k1: 0.85,
            sigma_k2: 1.0,
            sigma_omega1: 0.5,
            sigma_omega2: 0.856,
            beta1: 0.075,
            beta2: 0.0828,
            alpha1: 0.075_f64 / 0.09 - 0.5 * kappa * kappa / sqrt_beta_star,
            alpha2: 0.0828_f64 / 0.09 - 0.856 * kappa * kappa / sqrt_beta_star,
        };
        s.compute_wall_distance();
        s
    }

    fn clamp(i: isize, n: usize) -> usize {
        i.clamp(0, n as isize - 1) as usize
    }

    /// Recompute the per-cell distance to the nearest domain wall (used by the
    /// `F1`/`F2` blending functions).
    pub fn compute_wall_distance(&mut self) {
        let g = &self.grid;
        let (dx, dy, lx, ly) = (g.dx, g.dy, g.lx, g.ly);
        for j in 0..g.ny {
            for i in 0..g.nx {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dy;
                let d = x.min(lx - x).min(y).min(ly - y);
                self.wall_distance[g.idx(i, j)] = d;
            }
        }
    }

    /// Central-difference gradient `(∂f/∂x, ∂f/∂y)` with one-sided clamping at
    /// the boundary.
    fn grad(&self, f: &[f64], i: usize, j: usize) -> (f64, f64) {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let im = g.idx(Self::clamp(i as isize - 1, g.nx), j);
        let ip = g.idx(Self::clamp(i as isize + 1, g.nx), j);
        let jm = g.idx(i, Self::clamp(j as isize - 1, g.ny));
        let jp = g.idx(i, Self::clamp(j as isize + 1, g.ny));
        ((f[ip] - f[im]) / (2.0 * dx), (f[jp] - f[jm]) / (2.0 * dy))
    }

    /// Central-difference Laplacian with one-sided clamping at the boundary.
    fn laplacian(&self, f: &[f64], i: usize, j: usize) -> f64 {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let c = g.idx(i, j);
        let im = g.idx(Self::clamp(i as isize - 1, g.nx), j);
        let ip = g.idx(Self::clamp(i as isize + 1, g.nx), j);
        let jm = g.idx(i, Self::clamp(j as isize - 1, g.ny));
        let jp = g.idx(i, Self::clamp(j as isize + 1, g.ny));
        (f[ip] - 2.0 * f[c] + f[im]) / (dx * dx) + (f[jp] - 2.0 * f[c] + f[jm]) / (dy * dy)
    }

    /// The `CD_{kω}` term used in the `F1` blend:
    /// `max(2 σ_{ω2} (1/ω) ∇k·∇ω, 1e-10)`.
    fn cross_term(&self, k: &[f64], w: &[f64], i: usize, j: usize) -> f64 {
        let (dkdx, dkdy) = self.grad(k, i, j);
        let (dwdx, dwdy) = self.grad(w, i, j);
        let wc = w[self.grid.idx(i, j)].max(1e-12);
        let val = 2.0 * self.sigma_omega2 * (dkdx * dwdx + dkdy * dwdy) / wc;
        val.max(1e-10)
    }

    /// Eddy viscosity `ν_t = a1 k / max(a1 ω, F2·Ω)` at cell `(i, j)`, using the
    /// `F2` blending of the vorticity magnitude `Ω`. Returns `0` when the
    /// denominator is non-positive (e.g. laminar / zero-turbulence limit).
    #[must_use]
    pub fn eddy_viscosity_at(&self, i: usize, j: usize) -> f64 {
        let c = self.grid.idx(i, j);
        let k = self.k[c].max(1e-12);
        let w = self.omega[c].max(1e-12);
        let y = self.wall_distance[c];
        let (_dudx, dudy) = self.grad(&self.u, i, j);
        let (dvdx, _dvdy) = self.grad(&self.v, i, j);
        let vort = (dvdx - dudy).abs();
        let arg2 = (2.0 * k.sqrt() / (self.beta_star * w * y)).max(500.0 * self.nu / (y * y * w));
        let f2 = arg2.tanh().powi(2);
        let denom = (self.a1 * w).max(f2 * vort);
        if denom <= 0.0 {
            0.0
        } else {
            self.a1 * k / denom
        }
    }

    /// Eddy viscosity field `ν_t` at every cell.
    #[must_use]
    pub fn eddy_viscosity_field(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.grid.len()];
        for j in 0..self.grid.ny {
            for i in 0..self.grid.nx {
                out[self.grid.idx(i, j)] = self.eddy_viscosity_at(i, j);
            }
        }
        out
    }

    /// Advance `k` and `ω` by one explicit Euler step of the SST transport
    /// equations with timestep `dt`. Values are floored at `1e-12` to stay
    /// positive and finite.
    pub fn step(&mut self, dt: f64) {
        let g = &self.grid;
        let (nx, ny) = (g.nx, g.ny);
        let k0 = self.k.clone();
        let w0 = self.omega.clone();
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let mut k1 = k0.clone();
        let mut w1 = w0.clone();

        for j in 0..ny {
            for i in 0..nx {
                let c = g.idx(i, j);
                let y = self.wall_distance[c];
                let kc = k0[c].max(1e-12);
                let wc = w0[c].max(1e-12);

                // Mean velocity strain-rate and vorticity.
                let (dudx, dudy) = self.grad(&u0, i, j);
                let (dvdx, dvdy) = self.grad(&v0, i, j);
                let sxx = dudx;
                let syy = dvdy;
                let sxy = 0.5 * (dudy + dvdx);
                let s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy; // S:S
                let vort = (dvdx - dudy).abs();

                // Eddy viscosity and production (with the a1 limiter).
                let arg2 = (2.0 * kc.sqrt() / (self.beta_star * wc * y))
                    .max(500.0 * self.nu / (y * y * wc));
                let f2 = arg2.tanh().powi(2);
                let nu_t = if (self.a1 * wc).max(f2 * vort) <= 0.0 {
                    0.0
                } else {
                    self.a1 * kc / (self.a1 * wc).max(f2 * vort)
                };
                let pk = (2.0 * nu_t * s2).min(2.0 * self.beta_star * kc * wc);

                // F1 blending of coefficients.
                let arg1 = (kc.sqrt() / (self.beta_star * wc * y))
                    .max(500.0 * self.nu / (y * y * wc))
                    .max(4.0 * self.nu * self.sigma_omega2 * kc / self.cross_term(&k0, &w0, i, j));
                let f1 = arg1.tanh().powi(4);
                let sigma_k = f1 * self.sigma_k1 + (1.0 - f1) * self.sigma_k2;
                let sigma_w = f1 * self.sigma_omega1 + (1.0 - f1) * self.sigma_omega2;
                let beta = f1 * self.beta1 + (1.0 - f1) * self.beta2;
                let alpha = f1 * self.alpha1 + (1.0 - f1) * self.alpha2;

                // Diffusion (frozen-coefficient Laplacian) and advection.
                let (dkdx, dkdy) = self.grad(&k0, i, j);
                let (dwdx, dwdy) = self.grad(&w0, i, j);
                let diff_k = (self.nu + sigma_k * nu_t) * self.laplacian(&k0, i, j);
                let diff_w = (self.nu + sigma_w * nu_t) * self.laplacian(&w0, i, j);
                let cross = (1.0 - f1) * 2.0 * self.sigma_omega2 * (dkdx * dwdx + dkdy * dwdy) / wc;
                let conv_k = u0[c] * dkdx + v0[c] * dkdy;
                let conv_w = u0[c] * dwdx + v0[c] * dwdy;

                let dk = -conv_k + pk - self.beta_star * kc * wc + diff_k;
                let dw = -conv_w + alpha * nu_t.recip() * pk - beta * wc * wc + diff_w + cross;

                k1[c] = (kc + dt * dk).max(1e-12);
                w1[c] = (wc + dt * dw).max(1e-12);
            }
        }

        self.k = k1;
        self.omega = w1;
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
    fn decaying_turbulence_stays_finite_and_positive() {
        let grid = CollocatedGrid::new(12, 12, 1.0, 1.0).unwrap();
        let mut model = KOmegaSst::new(grid, 1e-3);
        let k0_mean: f64 = model.k.iter().sum::<f64>() / model.k.len() as f64;
        // No mean flow -> no production/advection: k and ω decay.
        for _ in 0..50 {
            model.step(1e-3);
        }
        let k1_mean: f64 = model.k.iter().sum::<f64>() / model.k.len() as f64;
        assert!(model.k.iter().all(|&x| x.is_finite() && x >= 1e-12));
        assert!(model.omega.iter().all(|&x| x.is_finite() && x >= 1e-12));
        assert!(k1_mean < k0_mean, "k should decay without production");
    }

    #[test]
    fn laminar_limit_gives_zero_eddy_viscosity() {
        let grid = CollocatedGrid::new(8, 8, 1.0, 1.0).unwrap();
        let mut model = KOmegaSst::new(grid, 1e-6);
        // Suppress turbulence so the eddy viscosity must collapse to ~0.
        model.k.fill(1e-12);
        model.omega.fill(1.0);
        let mut max_nu_t = 0.0_f64;
        for j in 0..model.grid().ny {
            for i in 0..model.grid().nx {
                max_nu_t = max_nu_t.max(model.eddy_viscosity_at(i, j));
            }
        }
        assert!(
            max_nu_t < 1e-6,
            "eddy viscosity should vanish in laminar limit"
        );
    }

    #[test]
    fn shear_produces_positive_eddy_viscosity() {
        let grid = CollocatedGrid::new(8, 8, 1.0, 1.0).unwrap();
        let mut model = KOmegaSst::new(grid.clone(), 1e-3);
        // Linear shear u = y gives constant strain and vorticity.
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                model.u[grid.idx(i, j)] = (j as f64 + 0.5) * grid.dy;
            }
        }
        let mut max_nu_t = 0.0_f64;
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                max_nu_t = max_nu_t.max(model.eddy_viscosity_at(i, j));
            }
        }
        assert!(max_nu_t > 0.0, "shear flow should produce eddy viscosity");
        assert!(max_nu_t.is_finite());
    }

    #[test]
    fn turbulent_kinetic_energy_is_non_negative() {
        let grid = CollocatedGrid::new(10, 10, 1.0, 1.0).unwrap();
        let mut model = KOmegaSst::new(grid, 1e-3);
        for _ in 0..20 {
            model.step(5e-4);
        }
        assert!(model.k.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn wall_distance_is_positive_interior() {
        let grid = CollocatedGrid::new(10, 10, 2.0, 2.0).unwrap();
        let model = KOmegaSst::new(grid.clone(), 1e-3);
        assert!(model.wall_distance.iter().all(|&d| d > 0.0));
        // Centre cell is farthest from any wall.
        let c = grid.idx(grid.nx / 2, grid.ny / 2);
        let g = model.grid();
        let cx = ((c % g.nx) as f64 + 0.5) * g.dx;
        let cy = ((c / g.nx) as f64 + 0.5) * g.dy;
        let expected = cx.min(g.lx - cx).min(cy).min(g.ly - cy);
        assert_abs_diff_eq!(model.wall_distance[c], expected, epsilon = 1e-9);
    }
}
