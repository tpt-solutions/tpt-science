//! 3-D z-level (layered) ocean dynamical core.
//!
//! [`Ocean3D`] extends the 2-D [`crate::ShallowWater`] primitive by stacking
//! `nz` z-level layers. Density is diagnosed from prognostic temperature `T`
//! and salinity `S` through a linear equation of state, hydrostatic pressure is
//! integrated from that density, and tracers are advanced with horizontal
//! advection plus constant-coefficient vertical mixing.
//!
//! Two evolution paths are provided:
//!
//! * [`Ocean3D::step_3d`] — the **hydrostatic** primitive-equation step
//!   (pressure-gradient force from the integrated hydrostatic pressure, plus
//!   Coriolis and bottom friction). The vertical velocity is not prognosed.
//! * [`Ocean3D::step_3d_nonhydrostatic`] — the hydrostatic step followed by an
//!   **optional non-hydrostatic pressure correction**: a 3-D pressure-Poisson
//!   equation is solved with conjugate gradients
//!   ([`tpt_sci_grid::sparse::conjugate_gradient`]) on the structured grid to
//!   remove the divergence of the provisional velocity.
//!
//! All fields use the lexicographic node ordering
//! `index = ix + iy·nx + iz·nx·ny` shared by [`tpt_sci_grid::UniformGrid3D`]
//! (so `z` is the slowest-varying axis and the surface is `iz = nz - 1`).

use tpt_sci_grid::sparse::{conjugate_gradient, laplacian_3d_sparse};
use tpt_sci_grid::{Boundary, UniformGrid3D};

use crate::OceanError;

/// A 3-D z-level ocean model on a uniform tensor-product grid.
///
/// Fields (`u`, `v`, `w`, `t`, `s`) are stored flat with `nx·ny·nz` entries in
/// the ordering `ix + iy·nx + iz·nx·ny` (`z` slowest). `t` is temperature (K or
/// °C as long as the reference is consistent) and `s` is salinity (psu).
#[derive(Debug, Clone)]
pub struct Ocean3D {
    /// The structured grid the model is discretised on.
    pub grid: UniformGrid3D,
    /// `x`-velocity `u` (m/s), flat length `nx·ny·nz`.
    pub u: Vec<f64>,
    /// `y`-velocity `v` (m/s).
    pub v: Vec<f64>,
    /// `z`-velocity `w` (m/s) — diagnostic in hydrostatic mode.
    pub w: Vec<f64>,
    /// Temperature `T` (active tracer).
    pub t: Vec<f64>,
    /// Salinity `S` (active tracer).
    pub s: Vec<f64>,
    /// Reference density `ρ0` (kg/m³).
    pub rho0: f64,
    /// Thermal expansion coefficient `α` (kg/m³ per unit T).
    pub alpha: f64,
    /// Haline contraction coefficient `β` (kg/m³ per unit S).
    pub beta: f64,
    /// Reference temperature `T0`.
    pub t_ref: f64,
    /// Reference salinity `S0`.
    pub s_ref: f64,
    /// Gravitational acceleration `g` (m/s²).
    pub g: f64,
    /// Coriolis parameter `f` (1/s).
    pub f: f64,
    /// Vertical mixing coefficient `K_v` (m²/s, constant).
    pub kv: f64,
    /// Horizontal mixing coefficient `K_h` (m²/s, constant).
    pub kh: f64,
    /// Linear bottom/background friction `k` (1/s).
    pub friction: f64,
}

impl Ocean3D {
    /// Construct a quiescent, uniformly-stratified ocean of `nx × ny × nz` cells.
    ///
    /// The domain spans `[0, lx] × [0, ly] × [0, lz]` with `z` increasing
    /// upward and the surface at `z = lz`. Temperature and salinity are
    /// initialised to the reference values `t_ref`/`s_ref` (so the initial
    /// density is `rho0` everywhere and the ocean is initially motionless).
    ///
    /// # Errors
    ///
    /// Returns [`OceanError::InvalidModel`] if any extent is non-positive or any
    /// cell count is below 2 (the grid assembler rejects those), or if the
    /// equation-of-state / physical constants are non-finite or negative where a
    /// positive value is required (`rho0`, `alpha`, `beta`, `g`).
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
        beta: f64,
        t_ref: f64,
        s_ref: f64,
        g: f64,
        f: f64,
        kv: f64,
    ) -> Result<Self, OceanError> {
        if !(rho0.is_finite() && rho0 > 0.0) {
            return Err(OceanError::InvalidModel("rho0 must be positive".into()));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(OceanError::InvalidModel("alpha must be positive".into()));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(OceanError::InvalidModel("beta must be positive".into()));
        }
        if !(g.is_finite() && g > 0.0) {
            return Err(OceanError::InvalidModel("g must be positive".into()));
        }
        let grid = UniformGrid3D::new(nx, 0.0, lx, ny, 0.0, ly, nz, 0.0, lz)
            .map_err(|e| OceanError::InvalidModel(e.to_string()))?;
        let n = grid.len();
        Ok(Self {
            grid,
            u: vec![0.0; n],
            v: vec![0.0; n],
            w: vec![0.0; n],
            t: vec![t_ref; n],
            s: vec![s_ref; n],
            rho0,
            alpha,
            beta,
            t_ref,
            s_ref,
            g,
            f,
            kv,
            kh: 0.0,
            friction: 1e-5,
        })
    }

    /// Linearised equation of state: `ρ = ρ0 − α·(T − T0) + β·(S − S0)`.
    ///
    /// Warmer water is lighter and saltier water is denser, matching the
    /// Boussinesq sign convention.
    #[must_use]
    pub fn density(&self) -> Vec<f64> {
        self.t
            .iter()
            .zip(&self.s)
            .map(|(&t, &s)| {
                self.rho0 - self.alpha * (t - self.t_ref) + self.beta * (s - self.s_ref)
            })
            .collect()
    }

    /// Hydrostatic pressure at each cell centre, integrated downward from the
    /// (rigid-lid) surface: `p[k] = g·dz·Σ_{m=k}^{nz-1} ρ_m`.
    ///
    /// With `z` increasing upward, `z = k` cell centre, this is `p = ∫_k^surf ρ g dz`
    /// and therefore satisfies the discrete hydrostatic balance
    /// `(p[k+1] − p[k]) / dz ≈ −ρ_k·g`.
    #[must_use]
    pub fn hydrostatic_pressure(&self) -> Vec<f64> {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let dz = self.grid.dz();
        let rho = self.density();
        // Column-integrated weight from the surface down to each level.
        let mut col_sum = vec![0.0f64; nx * ny];
        let mut p = vec![0.0f64; nx * ny * nz];
        for iz in (0..nz).rev() {
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
    /// (per unit density) at every node, from the hydrostatic pressure.
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
                    // Guard the (near-)uniform density case so 1/ρ stays finite.
                    let inv = 1.0 / rho[c].max(1e-3);
                    fx[c] = -inv * dpx;
                    fy[c] = -inv * dpy;
                }
            }
        }
        (fx, fy)
    }

    /// Apply one explicit, constant-coefficient vertical-mixing step to the
    /// temperature and salinity fields (no-flux / Neumann at the top and bottom).
    pub fn mix_vertical(&mut self, dt: f64) {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let dz2 = self.grid.dz() * self.grid.dz();
        let factor = self.kv * dt / dz2;
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        for field in [&mut self.t, &mut self.s] {
            let src = field.clone();
            for iz in 0..nz {
                let km = clamp(iz as isize - 1, nz);
                let kp = clamp(iz as isize + 1, nz);
                for iy in 0..ny {
                    for ix in 0..nx {
                        let c = ix + iy * nx + iz * nx * ny;
                        let cm = ix + iy * nx + km * nx * ny;
                        let cp = ix + iy * nx + kp * nx * ny;
                        field[c] = src[c] + factor * (src[cp] - 2.0 * src[c] + src[cm]);
                    }
                }
            }
        }
    }

    /// One hydrostatic primitive-equation step:
    ///
    /// * momentum gets the baroclinic pressure-gradient force, Coriolis `f`,
    ///   linear friction, and (if [`Ocean3D::kh`] > 0) horizontal diffusion;
    /// * `T` and `S` are advected by the horizontal flow (upwind) and then
    ///   relaxed by [`Ocean3D::mix_vertical`].
    ///
    /// The vertical velocity `w` is left at zero (hydrostatic balance is
    /// assumed), so this step alone is not divergence-free in 3-D; use
    /// [`Ocean3D::step_3d_nonhydrostatic`] for that.
    pub fn step_3d(&mut self, dt: f64) {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy) = (self.grid.dx(), self.grid.dy());
        let (fx, fy) = self.baroclinic_force();
        let rho = self.density();
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let t0 = self.t.clone();
        let s0 = self.s.clone();
        let mut nu = u0.clone();
        let mut nv = v0.clone();

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = ix + iy * nx + iz * nx * ny;
                    let inv = 1.0 / rho[c].max(1e-3);
                    // Horizontal diffusion of momentum (optional, for stability).
                    let (mut lap_u, mut lap_v) = (0.0, 0.0);
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
                        + dt * (inv * fx[c] + self.f * v0[c] - self.friction * u0[c]
                            + self.kh * lap_u);
                    nv[c] = v0[c]
                        + dt * (inv * fy[c] - self.f * u0[c] - self.friction * v0[c]
                            + self.kh * lap_v);

                    // Upwind advection of tracers by the horizontal flow.
                    let im = clamp(ix as isize - 1, nx);
                    let ip = clamp(ix as isize + 1, nx);
                    let jm = clamp(iy as isize - 1, ny);
                    let jp = clamp(iy as isize + 1, ny);
                    let ct = |i: usize, j: usize| t0[i + j * nx + iz * nx * ny];
                    let cs = |i: usize, j: usize| s0[i + j * nx + iz * nx * ny];
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
                    let dsdx = if u0[c] >= 0.0 {
                        (cs(ix, iy) - cs(im, iy)) / dx
                    } else {
                        (cs(ip, iy) - cs(ix, iy)) / dx
                    };
                    let dsdy = if v0[c] >= 0.0 {
                        (cs(ix, iy) - cs(ix, jm)) / dy
                    } else {
                        (cs(ix, jp) - cs(ix, iy)) / dy
                    };
                    self.t[c] = t0[c] - dt * (u0[c] * dtdx + v0[c] * dtdy);
                    self.s[c] = s0[c] - dt * (u0[c] * dsdx + v0[c] * dsdy);
                }
            }
        }
        self.u = nu;
        self.v = nv;
        self.mix_vertical(dt);
    }

    /// 3-D velocity divergence `∇·u` at every node.
    ///
    /// Uses backward differences `(u_i − u_{i-1})/dx` (with `u_{i-1} = u_i` at the
    /// boundary), which are the discrete adjoint of the forward-difference
    /// gradient used in [`Ocean3D::nonhydrostatic_correct`] and therefore
    /// consistent with the second-order Laplacian `∇²φ = ∇·(∇φ)` solved there.
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

    /// Maximum absolute interior divergence (boundary stencil artefacts excluded,
    /// mirroring [`tpt_sci_cfd_core::Step::max_divergence`]).
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

    /// Solve the non-hydrostatic pressure-Poisson equation
    /// `∇²φ = (∇·u*)/dt` with conjugate gradients on the structured grid and
    /// correct the provisional velocity: `u ⟵ u* − ∇φ`.
    ///
    /// This is the pressure-projection (fractional-step) correction that enforces
    /// `∇·u = 0` after the hydrostatic step. Dirichlet boundary conditions are
    /// applied to `φ` (so the corrected interior is divergence-free; boundary
    /// stencil artefacts are excluded by [`Ocean3D::max_divergence`]).
    ///
    /// # Panics
    ///
    /// Panics if the sparse linear solve returns a non-finite vector (should not
    /// happen for a well-posed SPD Laplacian; the guard exists so a NaN cannot
    /// silently propagate through the velocity field).
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
                    // Forward-difference gradient (consistent with the backward
                    // divergence in `divergence`, so ∇·∇φ reproduces the
                    // second-order Laplacian and the projection is exact in the
                    // interior).
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

    /// Hydrostatic step followed by the optional non-hydrostatic
    /// pressure-correction projection (see [`Ocean3D::nonhydrostatic_correct`]).
    pub fn step_3d_nonhydrostatic(&mut self, dt: f64, tol: f64) {
        self.step_3d(dt);
        self.nonhydrostatic_correct(dt, tol);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn model() -> Ocean3D {
        Ocean3D::new(
            9, 3, 5, 100.0, 30.0, 50.0, 1025.0, 0.2, 0.8, 15.0, 35.0, 9.81, 0.0, 0.01,
        )
        .unwrap()
    }

    #[test]
    fn hydrostatic_vertical_balance() {
        // A horizontally uniform, vertically varying density must reproduce the
        // discrete hydrostatic relation (p[k+1]-p[k])/dz ≈ -ρ_k·g.
        let mut o = model();
        let nz = o.grid.nz();
        // Stratify: warmer (lighter) toward the surface.
        for iz in 0..nz {
            let tval = o.t_ref + (iz as f64) * 0.5;
            for iy in 0..o.grid.ny() {
                for ix in 0..o.grid.nx() {
                    let c = o.index(ix, iy, iz);
                    o.t[c] = tval;
                }
            }
        }
        let p = o.hydrostatic_pressure();
        let dz = o.grid.dz();
        let rho = o.density();
        // Check at an interior column.
        let c0 = o.index(4, 1, 0);
        let c1 = o.index(4, 1, 1);
        let dp_dz = (p[c1] - p[c0]) / dz;
        assert_abs_diff_eq!(dp_dz, -rho[c0] * o.g, epsilon = 1e-6);
        // Pressure increases downward (deeper = higher pressure under more water).
        assert!(p[c0] > p[c1]);
    }

    #[test]
    fn motionless_stratified_is_hydrostatic() {
        // With no horizontal density variation the horizontal pressure gradient
        // must vanish, so a motionless state feels no spurious baroclinic force.
        let mut o = model();
        // Vertical-only stratification (function of z only).
        let nz = o.grid.nz();
        for iz in 0..nz {
            let tval = o.t_ref - (iz as f64) * 0.3;
            for iy in 0..o.grid.ny() {
                for ix in 0..o.grid.nx() {
                    let c = o.index(ix, iy, iz);
                    o.t[c] = tval;
                }
            }
        }
        let (fx, fy) = o.baroclinic_force();
        let max_fx = fx.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let max_fy = fy.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert_abs_diff_eq!(max_fx, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(max_fy, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn warm_anomaly_drives_flow() {
        // A warm surface anomaly makes the local column lighter, lowering the
        // hydrostatic pressure beneath it at every depth. The resulting
        // baroclinic pressure gradient therefore drives flow toward the anomaly
        // (convergence) on either side.
        let mut o = model();
        o.f = 0.0;
        let (nx, ny, nz) = (o.grid.nx(), o.grid.ny(), o.grid.nz());
        let cx = nx / 2;
        let cy = ny / 2;
        for iz in (nz - 2)..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = (ix as f64 - cx as f64).powi(2) + (iy as f64 - cy as f64).powi(2);
                    let c = o.index(ix, iy, iz);
                    o.t[c] += 4.0 * (-d2 / 4.0).exp();
                }
            }
        }
        // Confirm the anomaly creates a horizontal pressure gradient at depth.
        let (fx, _) = o.baroclinic_force();
        let bottom = nz - 1;
        let left = o.index(cx - 2, cy, bottom);
        let right = o.index(cx + 2, cy, bottom);
        // Pressure is lowest under the anomaly, so the force points toward it:
        // positive (toward +x) on the left, negative (toward -x) on the right.
        assert!(
            fx[left] > 0.0,
            "left-of-centre flow should head toward the anomaly"
        );
        assert!(
            fx[right] < 0.0,
            "right-of-centre flow should head toward the anomaly"
        );

        // After a few steps the velocities should have developed in that sense
        // and stayed finite.
        for _ in 0..5 {
            o.step_3d(0.5);
        }
        assert!(o.u.iter().chain(o.v.iter()).all(|&x| x.is_finite()));
        let u_left = o.u[left];
        let u_right = o.u[right];
        assert!(
            u_left > 0.0,
            "left-of-centre u should be positive (toward centre)"
        );
        assert!(
            u_right < 0.0,
            "right-of-centre u should be negative (toward centre)"
        );
    }

    #[test]
    fn vertical_mixing_smooths_tracers() {
        // Smaller vertical domain so the explicit step is effective and stable.
        let mut o = Ocean3D::new(
            9, 3, 6, 100.0, 30.0, 6.0, 1025.0, 0.2, 0.8, 15.0, 35.0, 9.81, 0.0, 2.0,
        )
        .unwrap();
        o.f = 0.0;
        let (nx, ny, nz) = (o.grid.nx(), o.grid.ny(), o.grid.nz());
        // A sharp vertical step in temperature at the mid-layer.
        for iz in 0..nz {
            let tval = if iz < nz / 2 { 10.0 } else { 20.0 };
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = o.index(ix, iy, iz);
                    o.t[c] = tval;
                }
            }
        }
        let grad0: f64 = (1..nz)
            .map(|iz| (o.t[o.index(0, 0, iz)] - o.t[o.index(0, 0, iz - 1)]).abs())
            .sum();
        // Several small stable explicit steps (Kv·dt/dz² = 0.28 < 0.5).
        for _ in 0..6 {
            o.mix_vertical(0.2);
        }
        let grad1: f64 = (1..nz)
            .map(|iz| (o.t[o.index(0, 0, iz)] - o.t[o.index(0, 0, iz - 1)]).abs())
            .sum();
        assert!(
            grad1 < grad0,
            "vertical mixing should reduce the tracer gradient"
        );
    }

    #[test]
    fn nonhydrostatic_removes_divergence() {
        // Start from a divergent provisional field (u grows linearly with x) and
        // confirm the projection yields an (interior) divergence-free field.
        let mut o = model();
        let (nx, ny, nz) = (o.grid.nx(), o.grid.ny(), o.grid.nz());
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = o.index(ix, iy, iz);
                    o.u[c] = ix as f64 * 0.01; // du/dx = 0.01 > 0 everywhere
                    o.v[c] = 0.0;
                    o.w[c] = 0.0;
                }
            }
        }
        let before = o.max_divergence();
        assert!(before > 0.0, "initial field must be divergent");
        o.nonhydrostatic_correct(1.0, 1e-10);
        let after = o.max_divergence();
        assert!(
            after < before * 0.05,
            "projection should reduce interior divergence (before={before}, after={after})"
        );
    }
}
