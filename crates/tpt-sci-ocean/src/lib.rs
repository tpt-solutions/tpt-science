//! # tpt-sci-ocean
//!
//! **Ocean circulation** modelling for the `tpt-science` pillar, built on
//! [`tpt_sci_cfd_core`]. It provides a 2-D **shallow-water** model
//! ([`ShallowWater`]) — the primitive prototype for large-scale circulation:
//!
//! * Height `h` and depth-averaged velocities `u`, `v` on a uniform grid.
//! * The shallow-water equations `∂h/∂t + ∇·(h·u) = 0`,
//!   `∂u/∂t + (u·∇)u = −g·∇h + f·v` (with Coriolis `f` for geostrophic balance).
//! * A finite-volume update reusing [`tpt_sci_cfd_core::CollocatedGrid`] for the
//!   discretization, so the same grid/convergence machinery carries over.
//!
//! This is a reduced-order circulation primitive — not a full 3-D primitive
//! equation ocean GCM.
//!
//! # Example
//!
//! ```
//! use tpt_sci_ocean::ShallowWater;
//!
//! // 64×64 basin, gravity g = 9.81, Coriolis f = 1e-4.
//! let mut sw = ShallowWater::new(64, 64, 1.0, 1.0, 9.81, 1e-4, 0.001);
//! // Perturb the free surface; it should evolve without blowing up.
//! sw.perturb_center(1.0);
//! for _ in 0..10 {
//!     sw.step(0.001);
//! }
//! assert!(sw.max_speed().is_finite());
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::OceanError;

use tpt_sci_cfd_core::CollocatedGrid;

/// A 2-D shallow-water ocean model on a uniform grid.
#[derive(Debug, Clone)]
pub struct ShallowWater {
    grid: CollocatedGrid,
    /// Free-surface height `h` (m), length `nx·ny`.
    pub h: Vec<f64>,
    /// Depth-averaged `x`-velocity `u`.
    pub u: Vec<f64>,
    /// Depth-averaged `y`-velocity `v`.
    pub v: Vec<f64>,
    /// Rest height `H0` (m).
    pub h0: f64,
    /// Gravity `g` (m/s²).
    pub g: f64,
    /// Coriolis parameter `f` (1/s).
    pub f: f64,
    /// Linear bottom friction `k` (1/s).
    pub friction: f64,
}

impl ShallowWater {
    /// Construct a model with rest height `h0` everywhere.
    ///
    /// # Panics
    ///
    /// Panics if the underlying grid construction fails (non-positive extents or
    /// zero cell counts).
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64, g: f64, f: f64, _dt: f64) -> Self {
        let grid = CollocatedGrid::new(nx, ny, lx, ly).unwrap();
        let n = grid.len();
        Self {
            h: vec![10.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            h0: 10.0,
            g,
            f,
            friction: 1e-4,
            grid,
        }
    }

    /// Add a Gaussian bump of amplitude `amp` at the domain center (a
    /// gravity-wave seed).
    pub fn perturb_center(&mut self, amp: f64) {
        let (cx, cy) = (self.grid.nx as f64 / 2.0, self.grid.ny as f64 / 2.0);
        for j in 0..self.grid.ny {
            for i in 0..self.grid.nx {
                let d2 = (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2);
                self.h[self.grid.idx(i, j)] += amp * (-d2 / 20.0).exp();
            }
        }
    }

    /// Advance by one explicit step of the shallow-water equations.
    pub fn step(&mut self, dt: f64) {
        let g = &self.grid;
        let (dx, dy) = (g.dx, g.dy);
        let idx = |i: usize, j: usize| g.idx(i, j);
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        let (h0, grav, f, k) = (self.h0, self.g, self.f, self.friction);

        let h = self.h.clone();
        let u = self.u.clone();
        let v = self.v.clone();
        let mut nh = h.clone();
        let mut nu = u.clone();
        let mut nv = v.clone();

        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = idx(i, j);
                let im = idx(clamp(i as isize - 1, g.nx), j);
                let ip = idx(clamp(i as isize + 1, g.nx), j);
                let jm = idx(i, clamp(j as isize - 1, g.ny));
                let jp = idx(i, clamp(j as isize + 1, g.ny));

                // Continuity: dh/dt = -∇·(h u).
                let div = (h[c] * u[c] - h[im] * u[im]) / dx + (h[c] * v[c] - h[jm] * v[jm]) / dy;
                nh[c] = (h[c] - dt * div).max(1e-3);

                // Momentum (x): du/dt = -(u·∇)u - g·∂η/∂x + f·v - k·u.
                let dudx = (u[ip] - u[im]) / (2.0 * dx);
                let dudy = (u[jp] - u[jm]) / (2.0 * dy);
                let deta = (nh[ip] - nh[im]) / (2.0 * dx);
                nu[c] =
                    u[c] + dt * (-(u[c] * dudx + v[c] * dudy) - grav * deta + f * v[c] - k * u[c]);

                // Momentum (y): dv/dt = -(v·∇)v - g·∂η/∂y - f·u - k·v.
                let dvdx = (v[ip] - v[im]) / (2.0 * dx);
                let dvdy = (v[jp] - v[jm]) / (2.0 * dy);
                let detay = (nh[jp] - nh[jm]) / (2.0 * dy);
                nv[c] =
                    v[c] + dt * (-(v[c] * dvdx + v[c] * dvdy) - grav * detay - f * u[c] - k * v[c]);
            }
        }
        self.h = nh;
        self.u = nu;
        self.v = nv;
        let _ = h0;
    }

    /// Maximum horizontal speed `max(|u|, |v|)` (a stability indicator).
    #[must_use]
    pub fn max_speed(&self) -> f64 {
        self.u
            .iter()
            .zip(self.v.iter())
            .map(|(&uu, &vv)| (uu.abs()).max(vv.abs()))
            .fold(0.0, f64::max)
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
    fn still_water_stays_still() {
        let mut sw = ShallowWater::new(32, 32, 1.0, 1.0, 9.81, 0.0, 0.001);
        for _ in 0..20 {
            sw.step(0.001);
        }
        assert_abs_diff_eq!(sw.max_speed(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn bump_propagates_finitely() {
        // dt chosen inside the explicit-scheme CFL limit: c·dt/dx < 1 with
        // c = sqrt(g·h0) = sqrt(9.81·10) ≈ 9.9 and dx = 1/48.
        let mut sw = ShallowWater::new(48, 48, 1.0, 1.0, 9.81, 1e-4, 0.001);
        sw.perturb_center(1.0);
        for _ in 0..30 {
            sw.step(0.001);
        }
        assert!(sw.max_speed().is_finite());
        // Height stays positive and bounded.
        assert!(sw.h.iter().all(|&x| x > 0.0 && x.is_finite()));
    }

    #[test]
    fn coriolis_deflects_flow() {
        let mut sw = ShallowWater::new(32, 32, 1.0, 1.0, 9.81, 1e-3, 0.002);
        sw.u[sw.grid.idx(16, 16)] = 1.0; // initial zonal velocity
        sw.step(0.002);
        // Coriolis f·v term should have spawned a meridional (v) component.
        assert!(sw.v[sw.grid.idx(16, 16)].abs() > 0.0);
    }
}
