//! From-scratch classical density-functional theory (DFT) using a
//! square-gradient (van der Waals / Cahn–Hilliard-style) local free-energy
//! functional.
//!
//! This module implements a self-contained classical DFT *alongside* the
//! `feos`-wrap path ([`crate::ClassicalDft`]) so the crate no longer needs an
//! external functional to study a simple inhomogeneous fluid. The grand
//! potential density is
//!
//! ```text
//! Ω[n] = ∫ [ f_bulk(n(r)) + κ/2 (∇n)² + V_ext(r) n(r) - μ n(r) ] d r ,
//! ```
//!
//! with a van der Waals bulk free-energy density
//! `f_bulk(n) = n(ln n − 1) + (1/b)[(1−b n) ln(1−b n) + b n] − a n²/2`
//! (in units of `k_B T`, so the chemical potential is
//! `μ(n) = ln(n/(1−b n)) − a n`). Minimising `Ω` gives the Euler–Lagrange
//! equation
//!
//! ```text
//! μ = ∂f_bulk/∂n − κ ∇²n + V_ext(r) ,
//! ```
//!
//! which is iterated to a constant chemical potential by gradient relaxation.
//! The same functional drives both the 1-D planar solve ([`SquareGradientDft::solve_1d`])
//! and its 3-D generalisation ([`SquareGradientDft::solve_3d`]), where the
//! Laplacian is applied through `tpt-sci-grid`'s sparse 3-D operator.

use tpt_sci_grid::{
    Boundary, CsrMatrix, UniformGrid1D, UniformGrid3D, laplacian_1d_sparse, laplacian_3d_sparse,
};

use crate::error::DftError;

/// Parameters of the van der Waals square-gradient functional (all in units of
/// `k_B T`, i.e. `β = 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VdWParams {
    /// Excluded-volume parameter `b`; close packing occurs at `n = 1/b`.
    pub b: f64,
    /// Attractive strength `a` (dimensionless, equal to `β·a_phys`).
    pub a: f64,
    /// Square-gradient coefficient `κ` (sets the interfacial width / capillary
    /// length `ξ = sqrt(κ / f''(n_b))`).
    pub kappa: f64,
}

impl VdWParams {
    /// Construct from the excluded volume `b`, attraction `a`, and gradient
    /// coefficient `κ`.
    #[must_use]
    pub fn new(b: f64, a: f64, kappa: f64) -> Self {
        Self { b, a, kappa }
    }

    /// Bulk Helmholtz free-energy density `f_bulk(n)` (units of `k_B T` per
    /// volume).
    #[must_use]
    pub fn free_energy_density(&self, n: f64) -> f64 {
        let bn = self.b * n;
        n * (n.ln() - 1.0) + (1.0 / self.b) * ((1.0 - bn) * (1.0 - bn).ln() + bn)
            - 0.5 * self.a * n * n
    }

    /// Local chemical potential `μ(n) = ∂f_bulk/∂n`.
    #[must_use]
    pub fn chemical_potential(&self, n: f64) -> f64 {
        (n / (1.0 - self.b * n)).ln() - self.a * n
    }

    /// `∂μ/∂n = ∂²f_bulk/∂n²`, the curvature of the bulk free energy.
    #[must_use]
    pub fn dmu_dn(&self, n: f64) -> f64 {
        1.0 / n + self.b / (1.0 - self.b * n) - self.a
    }

    /// Bulk correlation (capillary) length `ξ = sqrt(κ / f''(n_b))`.
    #[must_use]
    pub fn correlation_length(&self, n: f64) -> f64 {
        (self.kappa / self.dmu_dn(n)).sqrt()
    }

    /// Bulk density `n` that solves `μ(n) = mu` (the homogeneous equilibrium at
    /// the prescribed chemical potential).
    ///
    /// # Errors
    ///
    /// Returns [`DftError::Functional`] if the parameters admit no physical
    /// density root (e.g. `b <= 0`) or the root lies outside `(0, 1/b)`.
    pub fn bulk_density(&self, mu: f64) -> Result<f64, DftError> {
        if self.b <= 0.0 {
            return Err(DftError::Functional(
                "excluded-volume b must be positive".into(),
            ));
        }
        let lo = 1e-9;
        let hi = 1.0 / self.b - 1e-9;
        let g = |n: f64| self.chemical_potential(n) - mu;
        if g(lo).is_nan() || g(hi).is_nan() || g(lo) >= 0.0 || g(hi) <= 0.0 {
            return Err(DftError::Functional(
                "no bulk density root in (0, 1/b) for the given chemical potential".into(),
            ));
        }
        let mut a = lo;
        let mut b = hi;
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if g(mid) > 0.0 {
                b = mid;
            } else {
                a = mid;
            }
        }
        Ok(0.5 * (a + b))
    }
}

/// Iteration statistics returned alongside a solved density field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveStats {
    /// Number of relaxation iterations actually performed.
    pub iterations: usize,
    /// Final maximum magnitude of the Euler–Lagrange residual `|δΩ/δn|`.
    pub residual: f64,
}

/// A from-scratch square-gradient / local density-functional solver.
///
/// Unlike [`crate::ClassicalDft`] (which delegates to `feos`), this builds the
/// functional itself and minimises it by gradient relaxation on a real-space
/// grid.
#[derive(Debug, Clone, Copy)]
pub struct SquareGradientDft {
    /// Functional parameters in units of `k_B T`.
    pub params: VdWParams,
}

impl SquareGradientDft {
    /// Construct a solver for the given [`VdWParams`].
    #[must_use]
    pub fn new(params: VdWParams) -> Self {
        Self { params }
    }

    /// Bulk density in equilibrium with chemical potential `mu` (delegates to
    /// [`VdWParams::bulk_density`]).
    ///
    /// # Errors
    ///
    /// Propagates [`DftError::Functional`] when no physical root exists.
    pub fn bulk_density(&self, mu: f64) -> Result<f64, DftError> {
        self.params.bulk_density(mu)
    }

    /// Solve the 1-D planar square-gradient Euler–Lagrange equation on a uniform
    /// grid.
    ///
    /// # Errors
    ///
    /// Returns [`DftError::Profile`] if `initial` / `external_potential` lengths
    /// do not match the grid node count.
    pub fn solve_1d(
        &self,
        grid: &UniformGrid1D,
        cfg: &PlanarSolve,
    ) -> Result<DftSolution1D, DftError> {
        let n = grid.n();
        if cfg.initial.len() != n {
            return Err(DftError::Profile(format!(
                "initial length {} != grid nodes {}",
                cfg.initial.len(),
                n
            )));
        }
        if let Some(v) = &cfg.external_potential {
            if v.len() != n {
                return Err(DftError::Profile(
                    "external_potential length must equal grid nodes".into(),
                ));
            }
        }
        let lap = laplacian_1d_sparse(grid, Boundary::Neumann);
        let mut fixed = vec![None; n];
        if cfg.boundary == Boundary::Dirichlet {
            fixed[0] = Some(cfg.initial[0]);
            fixed[n - 1] = Some(cfg.initial[n - 1]);
        }
        let v_ext = cfg
            .external_potential
            .clone()
            .unwrap_or_else(|| vec![0.0; n]);
        let lmax = 2.0 / (grid.dx() * grid.dx());
        let ctx = RelaxCtx {
            v_ext,
            fixed,
            mu: cfg.mu,
            tol: cfg.tol,
            max_iter: cfg.max_iter,
            lmax,
        };
        let (profile, stats) = self.relax(cfg.initial.clone(), &lap, &ctx);
        Ok(DftSolution1D { profile, stats })
    }

    /// Solve the 3-D square-gradient Euler–Lagrange equation on a uniform tensor
    /// grid, using `tpt-sci-grid`'s sparse 3-D Laplacian for the gradient term.
    ///
    /// # Errors
    ///
    /// Returns [`DftError::Profile`] if `initial`, `external_potential`, or
    /// `fixed` lengths do not match the grid node count.
    pub fn solve_3d(
        &self,
        grid: &UniformGrid3D,
        cfg: &VolumetricSolve,
    ) -> Result<DftSolution3D, DftError> {
        let n = grid.len();
        if cfg.initial.len() != n {
            return Err(DftError::Profile(format!(
                "initial length {} != grid nodes {}",
                cfg.initial.len(),
                n
            )));
        }
        if cfg.fixed.len() != n {
            return Err(DftError::Profile(
                "fixed length must equal grid nodes".into(),
            ));
        }
        if let Some(v) = &cfg.external_potential {
            if v.len() != n {
                return Err(DftError::Profile(
                    "external_potential length must equal grid nodes".into(),
                ));
            }
        }
        let lap = laplacian_3d_sparse(grid, cfg.boundary);
        let mut fixed = cfg.fixed.clone();
        if cfg.boundary == Boundary::Dirichlet {
            let nx = grid.nx();
            let ny = grid.ny();
            let nz = grid.nz();
            let idx = |ix: usize, iy: usize, iz: usize| ix + iy * nx + iz * nx * ny;
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        if ix == 0
                            || ix == nx - 1
                            || iy == 0
                            || iy == ny - 1
                            || iz == 0
                            || iz == nz - 1
                        {
                            fixed[idx(ix, iy, iz)] = Some(cfg.initial[idx(ix, iy, iz)]);
                        }
                    }
                }
            }
        }
        let v_ext = cfg
            .external_potential
            .clone()
            .unwrap_or_else(|| vec![0.0; n]);
        let lmax = 2.0 / (grid.dx() * grid.dx())
            + 2.0 / (grid.dy() * grid.dy())
            + 2.0 / (grid.dz() * grid.dz());
        let ctx = RelaxCtx {
            v_ext,
            fixed,
            mu: cfg.mu,
            tol: cfg.tol,
            max_iter: cfg.max_iter,
            lmax,
        };
        let (field, stats) = self.relax(cfg.initial.clone(), &lap, &ctx);
        Ok(DftSolution3D { field, stats })
    }

    /// Planar (1-D) interfacial / wall surface tension
    /// `γ = ∫ κ (dn/dx)² dx`, the capillary contribution of a 1-D profile.
    #[must_use]
    pub fn surface_tension_1d(&self, grid: &UniformGrid1D, profile: &[f64]) -> f64 {
        let dx = grid.dx();
        let n = grid.n();
        let mut gamma = 0.0_f64;
        for i in 0..n {
            let dndx = if i == 0 {
                (profile[1] - profile[0]) / dx
            } else if i == n - 1 {
                (profile[i] - profile[i - 1]) / dx
            } else {
                (profile[i + 1] - profile[i - 1]) / (2.0 * dx)
            };
            gamma += self.params.kappa * dndx * dndx * dx;
        }
        gamma
    }

    /// Excess (gradient / surface) free energy of a 3-D field,
    /// `F_ex = ∫ κ |∇n|² dV`.
    #[must_use]
    pub fn excess_free_energy_3d(&self, grid: &UniformGrid3D, field: &[f64]) -> f64 {
        let dx = grid.dx();
        let dy = grid.dy();
        let dz = grid.dz();
        let nx = grid.nx();
        let ny = grid.ny();
        let nz = grid.nz();
        let idx = |ix: usize, iy: usize, iz: usize| ix + iy * nx + iz * nx * ny;
        let dnx = |ix: usize, iy: usize, iz: usize| -> f64 {
            if ix == 0 {
                (field[idx(1, iy, iz)] - field[idx(0, iy, iz)]) / dx
            } else if ix == nx - 1 {
                (field[idx(nx - 1, iy, iz)] - field[idx(nx - 2, iy, iz)]) / dx
            } else {
                (field[idx(ix + 1, iy, iz)] - field[idx(ix - 1, iy, iz)]) / (2.0 * dx)
            }
        };
        let dny = |ix: usize, iy: usize, iz: usize| -> f64 {
            if iy == 0 {
                (field[idx(ix, 1, iz)] - field[idx(ix, 0, iz)]) / dy
            } else if iy == ny - 1 {
                (field[idx(ix, ny - 1, iz)] - field[idx(ix, ny - 2, iz)]) / dy
            } else {
                (field[idx(ix, iy + 1, iz)] - field[idx(ix, iy - 1, iz)]) / (2.0 * dy)
            }
        };
        let dnz = |ix: usize, iy: usize, iz: usize| -> f64 {
            if iz == 0 {
                (field[idx(ix, iy, 1)] - field[idx(ix, iy, 0)]) / dz
            } else if iz == nz - 1 {
                (field[idx(ix, iy, nz - 1)] - field[idx(ix, iy, nz - 2)]) / dz
            } else {
                (field[idx(ix, iy, iz + 1)] - field[idx(ix, iy, iz - 1)]) / (2.0 * dz)
            }
        };
        let mut ex = 0.0_f64;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let g2 =
                        dnx(ix, iy, iz).powi(2) + dny(ix, iy, iz).powi(2) + dnz(ix, iy, iz).powi(2);
                    ex += self.params.kappa * g2 * dx * dy * dz;
                }
            }
        }
        ex
    }

    fn relax(&self, n: Vec<f64>, lap: &CsrMatrix, ctx: &RelaxCtx) -> (Vec<f64>, SolveStats) {
        let mut n = n;
        let mut residual = f64::INFINITY;
        let mut iterations = 0;
        for k in 0..ctx.max_iter {
            let lap_n = lap.mul_vec(&n);
            let mut n_min = f64::INFINITY;
            let mut n_max = 0.0_f64;
            for (i, v) in n.iter().enumerate() {
                if ctx.fixed[i].is_some() {
                    continue;
                }
                if *v < n_min {
                    n_min = *v;
                }
                if *v > n_max {
                    n_max = *v;
                }
            }
            if !n_min.is_finite() {
                n_min = 1e-9;
            }
            let dmu = self
                .params
                .dmu_dn(n_min)
                .abs()
                .max(self.params.dmu_dn(n_max).abs());
            let alpha = 0.8 / (self.params.kappa * ctx.lmax + dmu);
            let mut res_max = 0.0_f64;
            for i in 0..n.len() {
                if let Some(v) = ctx.fixed[i] {
                    n[i] = v;
                    continue;
                }
                let r = self.params.chemical_potential(n[i]) - self.params.kappa * lap_n[i]
                    + ctx.v_ext[i]
                    - ctx.mu;
                let mut nv = n[i] - alpha * r;
                if nv < 1e-12 {
                    nv = 1e-12;
                }
                n[i] = nv;
                let ra = r.abs();
                if ra > res_max {
                    res_max = ra;
                }
            }
            residual = res_max;
            iterations = k + 1;
            if res_max < ctx.tol {
                break;
            }
        }
        (
            n,
            SolveStats {
                iterations,
                residual,
            },
        )
    }
}

/// Bundled scalar inputs for the gradient-relaxation inner loop.
struct RelaxCtx {
    v_ext: Vec<f64>,
    fixed: Vec<Option<f64>>,
    mu: f64,
    tol: f64,
    max_iter: usize,
    lmax: f64,
}

/// Inputs for a 1-D planar [`SquareGradientDft::solve_1d`] solve.
#[derive(Debug, Clone)]
pub struct PlanarSolve {
    /// Target (constant) chemical potential `μ`.
    pub mu: f64,
    /// Initial guess, one value per grid node.
    pub initial: Vec<f64>,
    /// Outer boundary condition applied at both grid ends.
    pub boundary: Boundary,
    /// Optional soft external potential `V_ext(r)` per node (defaults to zero).
    pub external_potential: Option<Vec<f64>>,
    /// Convergence tolerance on the maximum Euler–Lagrange residual.
    pub tol: f64,
    /// Maximum number of relaxation iterations.
    pub max_iter: usize,
}

/// Output of a 1-D planar solve: the converged density profile and stats.
#[derive(Debug, Clone)]
pub struct DftSolution1D {
    /// Converged density `n(x)` (length = grid node count).
    pub profile: Vec<f64>,
    /// Iteration statistics.
    pub stats: SolveStats,
}

/// Inputs for a 3-D [`SquareGradientDft::solve_3d`] solve.
#[derive(Debug, Clone)]
pub struct VolumetricSolve {
    /// Target (constant) chemical potential `μ`.
    pub mu: f64,
    /// Initial guess, one value per grid node (node order matches
    /// [`UniformGrid3D::index`]).
    pub initial: Vec<f64>,
    /// Outer (box-face) boundary condition.
    pub boundary: Boundary,
    /// Optional soft external potential `V_ext(r)` per node.
    pub external_potential: Option<Vec<f64>>,
    /// Per-node Dirichlet pin: `Some(value)` holds the node fixed at `value`
    /// (e.g. an impenetrable spherical core), `None` leaves it free.
    pub fixed: Vec<Option<f64>>,
    /// Convergence tolerance on the maximum Euler–Lagrange residual.
    pub tol: f64,
    /// Maximum number of relaxation iterations.
    pub max_iter: usize,
}

/// Output of a 3-D solve: the converged density field and stats.
#[derive(Debug, Clone)]
pub struct DftSolution3D {
    /// Converged density `n(r)` (length = grid node count).
    pub field: Vec<f64>,
    /// Iteration statistics.
    pub stats: SolveStats,
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn params() -> VdWParams {
        VdWParams::new(0.5, 1.0, 0.5)
    }

    fn bulk() -> f64 {
        0.1
    }

    #[test]
    fn bulk_density_is_analytic_root() {
        let dft = SquareGradientDft::new(params());
        let nb = bulk();
        let mu = dft.params.chemical_potential(nb);
        let solved = dft.bulk_density(mu).unwrap();
        assert_abs_diff_eq!(solved, nb, epsilon = 1e-9);
    }

    #[test]
    fn flat_bulk_minimizes_to_analytic_density() {
        let dft = SquareGradientDft::new(params());
        let nb = bulk();
        let mu = dft.params.chemical_potential(nb);
        let grid = UniformGrid1D::new(81, 0.0, 4.0).unwrap();
        let initial: Vec<f64> = (0..grid.n()).map(|_| 1.5 * nb).collect();
        let cfg = PlanarSolve {
            mu,
            initial,
            boundary: Boundary::Neumann,
            external_potential: None,
            tol: 1e-4,
            max_iter: 20000,
        };
        let sol = dft.solve_1d(&grid, &cfg).unwrap();
        let max_dev = sol
            .profile
            .iter()
            .map(|v| (v - nb).abs())
            .fold(0.0, f64::max);
        assert!(max_dev < 1e-2, "profile deviated by {max_dev}");
    }

    #[test]
    fn hard_wall_profile_is_sensible_and_capillary_scales() {
        let nb = bulk();
        let mu = VdWParams::new(0.5, 1.0, 0.5).chemical_potential(nb);

        let solve = |kappa: f64| -> (Vec<f64>, f64) {
            let dft = SquareGradientDft::new(VdWParams::new(0.5, 1.0, kappa));
            let grid = UniformGrid1D::new(81, 0.0, 4.0).unwrap();
            let mut initial = vec![nb; grid.n()];
            initial[0] = 0.0;
            let cfg = PlanarSolve {
                mu,
                initial,
                boundary: Boundary::Dirichlet,
                external_potential: None,
                tol: 1e-4,
                max_iter: 20000,
            };
            let sol = dft.solve_1d(&grid, &cfg).unwrap();
            let gamma = dft.surface_tension_1d(&grid, &sol.profile);
            (sol.profile, gamma)
        };

        let width_of = |profile: &[f64], grid: &UniformGrid1D| -> f64 {
            let xs = grid.coordinates();
            let target_lo = 0.1 * nb;
            let target_hi = 0.9 * nb;
            let mut x10 = None;
            let mut x90 = None;
            for i in 0..profile.len() {
                if profile[i] >= target_lo && x10.is_none() {
                    x10 = Some(xs[i]);
                }
                if profile[i] >= target_hi && x90.is_none() {
                    x90 = Some(xs[i]);
                }
            }
            x90.unwrap() - x10.unwrap()
        };

        let grid = UniformGrid1D::new(81, 0.0, 4.0).unwrap();
        let (p1, g1) = solve(0.5);
        let (p4, g4) = solve(2.0);

        assert!(p1[1] < 0.5 * nb, "density should rise from the wall");
        assert!(
            (p1[p1.len() - 2] - nb).abs() < 0.02,
            "far field must be bulk"
        );
        assert!(
            p1[p1.len() - 2] > p1[1] * 2.0,
            "profile must increase toward bulk"
        );

        let w1 = width_of(&p1, &grid);
        let w4 = width_of(&p4, &grid);
        let width_ratio = w4 / w1;
        let kappa_ratio = (2.0_f64 / 0.5_f64).sqrt();
        assert!(
            (width_ratio / kappa_ratio - 1.0).abs() < 0.25,
            "interface width ratio {width_ratio} should match sqrt(kappa) {kappa_ratio}"
        );

        let gamma_ratio = g4 / g1;
        assert!(
            (gamma_ratio / kappa_ratio - 1.0).abs() < 0.3,
            "surface tension {g4} should scale as sqrt(kappa) (got {g1})"
        );
    }

    fn center(grid: &UniformGrid3D) -> (usize, usize, usize) {
        (grid.nx() / 2, grid.ny() / 2, grid.nz() / 2)
    }

    #[test]
    fn spherical_core_is_radially_monotonic_and_equals_1d_bulk() {
        let dft = SquareGradientDft::new(params());
        let nb = bulk();
        let mu = dft.params.chemical_potential(nb);
        let grid = UniformGrid3D::new(20, 0.0, 2.0, 20, 0.0, 2.0, 20, 0.0, 2.0).unwrap();
        let (cx, cy, cz) = center(&grid);
        let r_core = 0.4;
        let idx =
            |ix: usize, iy: usize, iz: usize| ix + iy * grid.nx() + iz * grid.nx() * grid.ny();
        let mut initial = vec![nb; grid.len()];
        let mut fixed = vec![None; grid.len()];
        for iz in 0..grid.nz() {
            for iy in 0..grid.ny() {
                for ix in 0..grid.nx() {
                    let dx = grid.x_coordinates()[ix] - grid.x_coordinates()[cx];
                    let dy = grid.y_coordinates()[iy] - grid.y_coordinates()[cy];
                    let dz = grid.z_coordinates()[iz] - grid.z_coordinates()[cz];
                    if (dx * dx + dy * dy + dz * dz).sqrt() < r_core {
                        fixed[idx(ix, iy, iz)] = Some(0.0);
                        initial[idx(ix, iy, iz)] = 0.0;
                    }
                }
            }
        }
        let cfg = VolumetricSolve {
            mu,
            initial,
            boundary: Boundary::Dirichlet,
            external_potential: None,
            fixed,
            tol: 2e-3,
            max_iter: 15000,
        };
        let sol = dft.solve_3d(&grid, &cfg).unwrap();

        let bulk_from_1d = dft.bulk_density(mu).unwrap();
        let far = sol.field[idx(cx, cy, grid.nz() - 2)];
        assert!(
            (far - bulk_from_1d).abs() < 0.05,
            "far field {far} should equal 1-D bulk {bulk_from_1d}"
        );

        let mut prev = -1.0;
        for ix in (cx + 4)..(grid.nx() - 1) {
            let v = sol.field[idx(ix, cy, cz)];
            assert!(
                v >= prev - 1e-3,
                "radial profile must be non-decreasing outward (got {v} after {prev})"
            );
            prev = v;
        }
        assert!(
            sol.field[idx(grid.nx() - 2, cy, cz)] > sol.field[idx(cx + 4, cy, cz)] * 1.5,
            "clear rise from core surface to bulk"
        );
    }

    #[test]
    fn excess_free_energy_scales_with_core_surface_area() {
        let nb = bulk();
        let mu = VdWParams::new(0.5, 1.0, 0.5).chemical_potential(nb);
        let run = |r_core: f64, kappa: f64| -> f64 {
            let dft = SquareGradientDft::new(VdWParams::new(0.5, 1.0, kappa));
            let grid = UniformGrid3D::new(22, 0.0, 2.0, 22, 0.0, 2.0, 22, 0.0, 2.0).unwrap();
            let (cx, cy, cz) = center(&grid);
            let xs = grid.x_coordinates();
            let ys = grid.y_coordinates();
            let zs = grid.z_coordinates();
            let idx =
                |ix: usize, iy: usize, iz: usize| ix + iy * grid.nx() + iz * grid.nx() * grid.ny();
            let mut initial = vec![nb; grid.len()];
            let mut fixed = vec![None; grid.len()];
            for iz in 0..grid.nz() {
                for iy in 0..grid.ny() {
                    for ix in 0..grid.nx() {
                        let dx = xs[ix] - xs[cx];
                        let dy = ys[iy] - ys[cy];
                        let dz = zs[iz] - zs[cz];
                        if (dx * dx + dy * dy + dz * dz).sqrt() < r_core {
                            fixed[idx(ix, iy, iz)] = Some(0.0);
                            initial[idx(ix, iy, iz)] = 0.0;
                        }
                    }
                }
            }
            let cfg = VolumetricSolve {
                mu,
                initial,
                boundary: Boundary::Dirichlet,
                external_potential: None,
                fixed,
                tol: 2e-3,
                max_iter: 15000,
            };
            let sol = dft.solve_3d(&grid, &cfg).unwrap();
            dft.excess_free_energy_3d(&grid, &sol.field)
        };

        let ex_small = run(0.4, 0.5);
        let ex_large = run(0.8, 0.5);
        let area_ratio = (0.8_f64 / 0.4_f64).powi(2);
        let ratio = ex_large / ex_small;
        assert!(
            (ratio / area_ratio - 1.0).abs() < 0.6,
            "excess free energy ratio {ratio} should track surface area ratio {area_ratio}"
        );
    }
}
