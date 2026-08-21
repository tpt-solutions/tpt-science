//! A genuine **primitive-equation atmospheric GCM dynamical core**.
//!
//! This is structurally analogous to the ocean's 3-D z-level core
//! ([`tpt_sci_ocean::Ocean3D`]): a 3-D field of horizontal winds `u`, `v` and
//! temperature `T` on a uniform tensor grid, advanced with the hydrostatic
//! primitive equations. The horizontal pressure-gradient force comes from the
//! hydrostatic pressure `p = g·∫ρ dz` (with a Boussinesq density `ρ = ρ0·(1 −
//! α·(T − T0))`), the Coriolis force uses a beta-plane `f(y)`, and temperature is
//! advected by the flow and relaxed toward a radiative-equilibrium profile.
//!
//! Two evolution paths mirror the ocean:
//!
//! * [`AtmosphereGcm::step`] — the **hydrostatic** primitive-equation step.
//! * [`AtmosphereGcm::step_nonhydrostatic`] — the hydrostatic step followed by an
//!   optional non-hydrostatic pressure-correction that projects the provisional
//!   velocity to be (interior) divergence-free via a 3-D Poisson solve with
//!   conjugate gradients (reusing `tpt-sci-grid`'s sparse 3-D Laplacian).
//!
//! The core couples to [`crate::EnergyBalanceModel`] through
//! [`AtmosphereGcm::couple_to_ebm`], which relaxes the global-mean temperature to
//! the EBM's equilibrium — closing the loop between the fast dynamical core and
//! the slow energy-balance climate state.

use tpt_sci_grid::sparse::{conjugate_gradient, laplacian_3d_sparse};
use tpt_sci_grid::{Boundary, UniformGrid3D};

use crate::EnergyBalanceModel;
use crate::error::ClimateError;

/// A 3-D primitive-equation atmospheric dynamical core on a uniform grid.
#[derive(Debug, Clone)]
pub struct AtmosphereGcm {
    /// Structured grid (`x` fastest, `z` slowest, surface at `iz = nz - 1`).
    pub grid: UniformGrid3D,
    /// `x`-velocity `u` (m/s), flat length `nx·ny·nz`.
    pub u: Vec<f64>,
    /// `y`-velocity `v` (m/s).
    pub v: Vec<f64>,
    /// `z`-velocity `w` (m/s) — diagnostic / correction field.
    pub w: Vec<f64>,
    /// Temperature `T` (K).
    pub t: Vec<f64>,
    /// Reference density `ρ0` (kg/m³).
    pub rho0: f64,
    /// Thermal expansion coefficient `α` (1/K): warmer air is lighter.
    pub alpha: f64,
    /// Reference temperature `T0` (K).
    pub t_ref: f64,
    /// Gravitational acceleration `g` (m/s²).
    pub g: f64,
    /// Base Coriolis parameter `f0` (1/s).
    pub f0: f64,
    /// Beta parameter `β = df/dy` (1/(s·m)).
    pub beta: f64,
    /// Horizontal mixing coefficient `K_h` (m²/s).
    pub kh: f64,
    /// Linear friction `k` (1/s).
    pub friction: f64,
    /// Relaxation rate toward radiative equilibrium (1/s).
    pub relax_rate: f64,
    /// Radiative-equilibrium temperature target `T_eq` (K) set by the EBM.
    pub t_eq: f64,
    /// Per-node Coriolis parameter `f(x,y)` (varies with `y` only).
    f: Vec<f64>,
}

impl AtmosphereGcm {
    /// Construct a quiescent, isothermal atmosphere of `nx × ny × nz` cells.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if any physical constant is
    /// non-finite or negative where positivity is required (`rho0`, `alpha`, `g`),
    /// or if the grid assembler rejects the extents.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        lx: f64,
        ly: f64,
        lz: f64,
        rho0: f64,
        alpha: f64,
        t_ref: f64,
        g: f64,
        f0: f64,
        beta: f64,
        kh: f64,
    ) -> Result<Self, ClimateError> {
        if !(rho0.is_finite() && rho0 > 0.0) {
            return Err(ClimateError::InvalidModel("rho0 must be positive".into()));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(ClimateError::InvalidModel("alpha must be positive".into()));
        }
        if !(g.is_finite() && g > 0.0) {
            return Err(ClimateError::InvalidModel("g must be positive".into()));
        }
        let grid =
            UniformGrid3D::new(nx, 0.0, lx, ny, 0.0, ly, nz, 0.0, lz).map_err(|e| {
                ClimateError::InvalidModel(e.to_string())
            })?;
        let n = grid.len();
        let y_mid = ly / 2.0;
        let yc = grid.y_coordinates();
        let f = yc
            .iter()
            .map(|&y| f0 + beta * (y - y_mid))
            .collect::<Vec<_>>();
        // Broadcast the 1-D (in y) Coriolis profile across the full 3-D field.
        let mut f3 = vec![0.0f64; n];
        for iz in 0..nz {
            for ix in 0..nx {
                for iy in 0..ny {
                    f3[ix + iy * nx + iz * nx * ny] = f[iy];
                }
            }
        }
        Ok(Self {
            grid,
            u: vec![0.0; n],
            v: vec![0.0; n],
            w: vec![0.0; n],
            t: vec![t_ref; n],
            rho0,
            alpha,
            t_ref,
            g,
            f0,
            beta,
            kh,
            friction: 1e-6,
            relax_rate: 1e-6,
            t_eq: t_ref,
            f: f3,
        })
    }

    /// Boussinesq density `ρ = ρ0·(1 − α·(T − T0))` (lighter where warmer).
    #[must_use]
    pub fn density(&self) -> Vec<f64> {
        self.t
            .iter()
            .map(|&t| self.rho0 * (1.0 - self.alpha * (t - self.t_ref)))
            .collect()
    }

    /// Hydrostatic pressure integrated *downward* from the (low-pressure) top:
    /// `p[k] = g·dz·Σ_{m=0}^{k-1} ρ_m`, satisfying discrete hydrostatic balance
    /// `(p[k+1] − p[k])/dz ≈ ρ_k·g` with `z` increasing upward.
    #[must_use]
    pub fn hydrostatic_pressure(&self) -> Vec<f64> {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let dz = self.grid.dz();
        let rho = self.density();
        let mut col_sum = vec![0.0f64; nx * ny];
        let mut p = vec![0.0f64; nx * ny * nz];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let col = ix + iy * nx;
                    col_sum[col] += rho[c] * dz;
                    p[c] = self.g * col_sum[col];
                }
            }
        }
        p
    }

    /// Lexicographic node index `ix + iy·nx + iz·nx·ny`.
    #[must_use]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        self.grid.index(ix, iy, iz)
    }

    /// Central-difference horizontal pressure-gradient force `(−∂p/∂x, −∂p/∂y)`
    /// (per unit density) at every node.
    #[must_use]
    fn baroclinic_force(&self) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy) = (self.grid.dx(), self.grid.dy());
        let p = self.hydrostatic_pressure();
        let rho = self.density();
        let mut fx = vec![0.0f64; p.len()];
        let mut fy = vec![0.0f64; p.len()];
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let im = clamp(ix as isize - 1, nx);
                    let ip = clamp(ix as isize + 1, nx);
                    let jm = clamp(iy as isize - 1, ny);
                    let jp = clamp(iy as isize + 1, ny);
                    let dpx = (p[ip + iy * nx + iz * nx * ny] - p[im + iy * nx + iz * nx * ny])
                        / (2.0 * dx);
                    let dpy = (p[ix + jp * nx + iz * nx * ny] - p[ix + jm * nx + iz * nx * ny])
                        / (2.0 * dy);
                    let inv = 1.0 / rho[c].max(1e-3);
                    fx[c] = -inv * dpx;
                    fy[c] = -inv * dpy;
                }
            }
        }
        (fx, fy)
    }

    /// One hydrostatic primitive-equation step: momentum gets the baroclinic
    /// pressure-gradient force, the beta-plane Coriolis force `f·v` / `−f·u`,
    /// linear friction and (if `kh > 0`) horizontal diffusion; `T` is advected by
    /// the horizontal flow (upwind), relaxed toward `t_eq`, and diffused.
    pub fn step(&mut self, dt: f64) {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy) = (self.grid.dx(), self.grid.dy());
        let (fx, fy) = self.baroclinic_force();
        let rho = self.density();
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let t0 = self.t.clone();
        let mut nu = u0.clone();
        let mut nv = v0.clone();

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let fc = self.f[c];
                    let inv = 1.0 / rho[c].max(1e-3);
                    let mut lap_u = 0.0;
                    let mut lap_v = 0.0;
                    if self.kh > 0.0 {
                        let im = clamp(ix as isize - 1, nx);
                        let ip = clamp(ix as isize + 1, nx);
                        let jm = clamp(iy as isize - 1, ny);
                        let jp = clamp(iy as isize + 1, ny);
                        let cu = |i: usize, j: usize| u0[i + j * nx + iz * nx * ny];
                        let cv = |i: usize, j: usize| v0[i + j * nx + iz * nx * ny];
                        lap_u = (cu(ip, iy) - 2.0 * u0[c] + cu(im, iy)) / (dx * dx)
                            + (cu(ix, jp) - 2.0 * u0[c] + cu(ix, jm)) / (dy * dy);
                        lap_v = (cv(ip, iy) - 2.0 * v0[c] + cv(im, iy)) / (dx * dx)
                            + (cv(ix, jp) - 2.0 * v0[c] + cv(ix, jm)) / (dy * dy);
                    }
                    nu[c] = u0[c]
                        + dt * (inv * fx[c] + fc * v0[c] - self.friction * u0[c]
                            + self.kh * lap_u);
                    nv[c] = v0[c]
                        + dt * (inv * fy[c] - fc * u0[c] - self.friction * v0[c]
                            + self.kh * lap_v);

                    // Upwind horizontal advection of temperature.
                    let im = clamp(ix as isize - 1, nx);
                    let ip = clamp(ix as isize + 1, nx);
                    let jm = clamp(iy as isize - 1, ny);
                    let jp = clamp(iy as isize + 1, ny);
                    let ct = |i: usize, j: usize| t0[i + j * nx + iz * nx * ny];
                    let dtdx = if u0[c] >= 0.0 {
                        (ct(ix, iy) - ct(im, iy)) / dx
                    } else {
                        (ct(ip, iy) - ct(ix, iy)) / dx
                    };
                    let dtdy = if v0[c] >= 0.0 {
                        (ct(ix, iy) - ct(ix, jm)) / dy
                    } else {
                        (ct(ix, jp) - ct(ix, iy)) / dy
                    };
                    let dtdt = if self.kh > 0.0 {
                        (ct(ip, iy) - 2.0 * t0[c] + ct(im, iy)) / (dx * dx)
                            + (ct(ix, jp) - 2.0 * t0[c] + ct(ix, jm)) / (dy * dy)
                    } else {
                        0.0
                    };
                    self.t[c] = t0[c]
                        + dt * (-(u0[c] * dtdx + v0[c] * dtdy)
                            + self.kh * dtdt
                            + self.relax_rate * (self.t_eq - t0[c]));
                }
            }
        }
        self.u = nu;
        self.v = nv;
    }

    /// 3-D velocity divergence `∇·u` (backward differences; consistent with the
    /// forward-difference gradient in [`AtmosphereGcm::nonhydrostatic_correct`]).
    #[must_use]
    pub fn divergence(&self) -> Vec<f64> {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy, dz) = (self.grid.dx(), self.grid.dy(), self.grid.dz());
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        let mut d = vec![0.0f64; self.u.len()];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let im = clamp(ix as isize - 1, nx);
                    let jm = clamp(iy as isize - 1, ny);
                    let km = clamp(iz as isize - 1, nz);
                    let du = (self.u[c] - self.u[im + iy * nx + iz * nx * ny]) / dx;
                    let dv = (self.v[c] - self.v[ix + jm * nx + iz * nx * ny]) / dy;
                    let dw = (self.w[c] - self.w[ix + iy * nx + km * nx * ny]) / dz;
                    d[c] = du + dv + dw;
                }
            }
        }
        d
    }

    /// Maximum absolute interior divergence (boundary stencil artefacts excluded).
    #[must_use]
    pub fn max_divergence(&self) -> f64 {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let d = self.divergence();
        let mut max = 0.0f64;
        for iz in 1..nz.saturating_sub(1) {
            for iy in 1..ny.saturating_sub(1) {
                for ix in 1..nx.saturating_sub(1) {
                    let c = ix + iy * nx + iz * nx * ny;
                    max = max.max(d[c].abs());
                }
            }
        }
        max
    }

    /// Solve the non-hydrostatic pressure-Poisson equation `∇²φ = (∇·u*)/dt` with
    /// conjugate gradients and correct the provisional velocity: `u ⟵ u* − ∇φ`.
    ///
    /// # Panics
    ///
    /// Panics if the sparse solve returns a non-finite vector (should not happen
    /// for a well-posed SPD Laplacian).
    pub fn nonhydrostatic_correct(&mut self, dt: f64, tol: f64) {
        let div = self.divergence();
        let rhs: Vec<f64> = div.iter().map(|&d| d / dt).collect();
        let lap = laplacian_3d_sparse(&self.grid, Boundary::Dirichlet);
        let phi = conjugate_gradient(&lap, &rhs, None, tol, lap.nrows() * 4 + 100);
        assert!(
            phi.iter().all(|&x| x.is_finite()),
            "non-hydrostatic Poisson solve produced a non-finite pressure"
        );
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy, dz) = (self.grid.dx(), self.grid.dy(), self.grid.dz());
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let w0 = self.w.clone();
        let mut nu = u0.clone();
        let mut nv = v0.clone();
        let mut nw = w0.clone();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let ip = clamp(ix as isize + 1, nx);
                    let jp = clamp(iy as isize + 1, ny);
                    let kp = clamp(iz as isize + 1, nz);
                    let dpx = (phi[ip + iy * nx + iz * nx * ny] - phi[c]) / dx;
                    let dpy = (phi[ix + jp * nx + iz * nx * ny] - phi[c]) / dy;
                    let dpz = (phi[ix + iy * nx + kp * nx * ny] - phi[c]) / dz;
                    nu[c] = u0[c] - dpx;
                    nv[c] = v0[c] - dpy;
                    nw[c] = w0[c] - dpz;
                }
            }
        }
        self.u = nu;
        self.v = nv;
        self.w = nw;
    }

    /// Hydrostatic step followed by the optional non-hydrostatic pressure
    /// correction.
    pub fn step_nonhydrostatic(&mut self, dt: f64, tol: f64) {
        self.step(dt);
        self.nonhydrostatic_correct(dt, tol);
    }

    /// Domain-mean temperature (K) — the quantity coupled to the EBM.
    #[must_use]
    pub fn mean_temperature(&self) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t.iter().sum::<f64>() / self.t.len() as f64
        }
    }

    /// Maximum horizontal wind speed `max(|u|, |v|)` (m/s), a stability indicator.
    #[must_use]
    pub fn max_wind(&self) -> f64 {
        self.u
            .iter()
            .zip(self.v.iter())
            .map(|(&uu, &vv)| uu.abs().max(vv.abs()))
            .fold(0.0, f64::max)
    }

    /// Couple the dynamical core to an [`EnergyBalanceModel`]: set the radiative
    /// relaxation target `t_eq` to the EBM's equilibrium temperature. After
    /// calling this the [`AtmosphereGcm::step`] temperature relaxation pulls the
    /// global-mean atmospheric temperature toward the EBM state. Returns the EBM
    /// equilibrium temperature (the relaxation target) for logging/inspection.
    pub fn couple_to_ebm(&mut self, ebm: &EnergyBalanceModel) -> f64 {
        self.t_eq = ebm.equilibrium_temperature();
        self.t_eq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn model() -> AtmosphereGcm {
        AtmosphereGcm::new(
            9, 3, 5, 100.0, 30.0, 50.0, 1.2, 1.0 / 300.0, 250.0, 9.81, 1e-4, 1.6e-11, 0.0,
        )
        .unwrap()
    }

    #[test]
    fn hydrostatic_vertical_balance() {
        let mut a = model();
        let nz = a.grid.nz();
        // Warmer (lighter) toward the surface.
        for iz in 0..nz {
            let tval = a.t_ref - (iz as f64) * 0.5;
            for iy in 0..a.grid.ny() {
                for ix in 0..a.grid.nx() {
                    let c = a.index(ix, iy, iz);
                    a.t[c] = tval;
                }
            }
        }
        let p = a.hydrostatic_pressure();
        let dz = a.grid.dz();
        let rho = a.density();
        let c0 = a.index(4, 1, 0);
        let c1 = a.index(4, 1, 1);
        let dp_dz = (p[c1] - p[c0]) / dz;
        // Pressure increases downward (deeper = higher pressure under more air).
        assert!(p[c1] > p[c0]);
        // The layer-to-layer pressure jump carries the density of the deeper cell.
        assert_abs_diff_eq!(dp_dz, rho[c1] * a.g, epsilon = 1e-6);
    }

    #[test]
    fn warm_anomaly_drives_convergence() {
        // A warm surface anomaly makes the column lighter, lowering the pressure
        // at depth; the baroclinic force drives flow toward the anomaly on both
        // sides (convergence).
        let mut a = model();
        let (nx, ny, nz) = (a.grid.nx(), a.grid.ny(), a.grid.nz());
        let cx = nx / 2;
        let cy = ny / 2;
        for iz in (nz - 2)..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = (ix as f64 - cx as f64).powi(2) + (iy as f64 - cy as f64).powi(2);
                    let c = a.index(ix, iy, iz);
                    a.t[c] += 8.0 * (-d2 / 4.0).exp();
                }
            }
        }
        let (fx, _) = a.baroclinic_force();
        let bottom = nz - 1;
        let left = a.index(cx - 2, cy, bottom);
        let right = a.index(cx + 2, cy, bottom);
        assert!(fx[left] > 0.0, "left-of-centre flow should head toward the anomaly");
        assert!(fx[right] < 0.0, "right-of-centre flow should head toward the anomaly");
        for _ in 0..5 {
            a.step(0.5);
        }
        assert!(a.u.iter().chain(a.v.iter()).all(|&x| x.is_finite()));
        assert!(a.u[left] > 0.0);
        assert!(a.u[right] < 0.0);
        assert!(a.t.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn relaxation_pulls_mean_toward_ebm() {
        let mut a = model();
        let ebm = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 280.0).unwrap();
        let teq = a.couple_to_ebm(&ebm);
        assert!(teq.is_finite() && teq > 250.0);
        // Warm the initial atmosphere away from the EBM target, then relax.
        for v in &mut a.t {
            *v += 20.0;
        }
        a.relax_rate = 5e-2;
        for _ in 0..200 {
            a.step(0.5);
        }
        assert!(
            (a.mean_temperature() - a.t_eq).abs() < 5.0,
            "mean T should relax toward the EBM equilibrium target"
        );
    }

    #[test]
    fn nonhydrostatic_removes_divergence() {
        let mut a = model();
        let (nx, ny, nz) = (a.grid.nx(), a.grid.ny(), a.grid.nz());
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = a.index(ix, iy, iz);
                    a.u[c] = ix as f64 * 0.01;
                }
            }
        }
        let before = a.max_divergence();
        assert!(before > 0.0);
        a.nonhydrostatic_correct(1.0, 1e-10);
        let after = a.max_divergence();
        assert!(
            after < before * 0.05,
            "projection should reduce interior divergence (before={before}, after={after})"
        );
    }
}
