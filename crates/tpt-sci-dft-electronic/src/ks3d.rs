//! 3-D real-space-grid Kohn–Sham electronic-structure solver.
//!
//! Generalizes the 1-D [`crate::KohnSham`] LDA solver to three dimensions using
//! the [`tpt_sci_grid`] 3-D (sparse) Laplacian. The Hamiltonian
//! `H = −½∇² + V_eff` is diagonalized with a Lanczos eigensolver (the grid is
//! far too large for a dense diagonalization), and the effective potential is
//! built self-consistently from the Hartree term (3-D Poisson, solved by
//! conjugate gradient) and the chosen exchange-correlation [`XcFunctional`]
//! (LDA or PBE/GGA).
//!
//! The kinetic operator is built on the **interior** nodes only (the box walls
//! are infinite Dirichlet boundaries, so the wavefunction vanishes there), which
//! removes the spurious boundary eigenvalue that a full-grid Dirichlet
//! discretization would introduce. A non-interacting ("bare") solve is also
//! provided: it diagonalizes `H = −½∇² + V_ext` once with no Hartree/XC double
//! counting, which is what the analytic box-eigenvalue tests rely on.

use std::f64::consts::PI;

use tpt_sci_grid::sparse::conjugate_gradient;
use tpt_sci_grid::{laplacian_3d_sparse, Boundary, CsrMatrix, UniformGrid3D};

use crate::eigen::lanczos_lowest;
use crate::xc::XcFunctional;
use crate::DftError;

/// Result of a 3-D Kohn–Sham solve.
#[derive(Debug, Clone)]
pub struct KohnSham3DResult {
    /// Total Kohn–Sham ground-state energy (Hartree).
    pub total_energy: f64,
    /// Occupied orbital energies (Hartree), lowest first.
    pub orbital_energies: Vec<f64>,
    /// Self-consistent density `ρ(r)` (electrons per unit volume), addressed by
    /// `grid.index(ix, iy, iz)` (zero on the box boundary).
    pub density: Vec<f64>,
    /// Occupied orbital coefficients (discrete-normalized so `Σ ψ² = 1`),
    /// addressed by `grid.index(ix, iy, iz)` (zero on the box boundary), lowest
    /// energy first.
    pub orbitals: Vec<Vec<f64>>,
}

/// A 3-D real-space Kohn–Sham self-consistent solver.
pub struct KohnSham3D {
    grid: UniformGrid3D,
    /// Full-grid 3-D Laplacian (used for the Hartree Poisson solve).
    lap: CsrMatrix,
    /// Interior-node indices (full-grid addresses that are not on a boundary).
    interior: Vec<usize>,
    /// Interior Dirichlet Laplacian row lists (`n_int × n_int`), the kinetic
    /// operator, stored as `(column, value)` per row.
    lap_int_rows: Vec<Vec<(usize, f64)>>,
    n_int: usize,
    dv: f64,
    v_ext: Vec<f64>,
    nelect: usize,
    n_bands: usize,
    mixing: f64,
    with_hartree: bool,
    xc: Box<dyn XcFunctional>,
    rho: Vec<f64>,
}

impl KohnSham3D {
    /// Construct a 3-D solver from a grid, a static external/local potential
    /// `v_ext` (e.g. a confining well or a [`crate::Pseudopotential`]), an
    /// electron count, and an exchange-correlation functional.
    ///
    /// # Errors
    ///
    /// Returns [`DftError::InvalidSetup`] if `nelect == 0` or `v_ext.len()` does
    /// not match `grid.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `v_ext` addresses `grid` out of range (it will not, for a valid
    /// [`UniformGrid3D`]).
    pub fn new(
        grid: UniformGrid3D,
        v_ext: Vec<f64>,
        nelect: usize,
        xc: Box<dyn XcFunctional>,
    ) -> Result<Self, DftError> {
        if nelect == 0 {
            return Err(DftError::InvalidSetup("need > 0 electrons".into()));
        }
        if v_ext.len() != grid.len() {
            return Err(DftError::InvalidSetup("v_ext length mismatch".into()));
        }
        let lap = laplacian_3d_sparse(&grid, Boundary::Dirichlet);
        let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
        let is_b = |ix: usize, iy: usize, iz: usize| -> bool {
            ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1
        };
        let interior: Vec<usize> = (0..grid.len())
            .filter(|&k| {
                let ix = k % nx;
                let iy = (k / nx) % ny;
                let iz = k / (nx * ny);
                !is_b(ix, iy, iz)
            })
            .collect();
        // Build the interior Dirichlet Laplacian: drop every matrix entry whose
        // column lies on the (ψ = 0) boundary, leaving the standard 7-point
        // interior stencil (negative-definite, no spurious modes).
        let n_int = interior.len();
        let mut lap_int_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_int);
        for &full_i in &interior {
            let start = lap.row_ptr[full_i];
            let end = lap.row_ptr[full_i + 1];
            let mut row = Vec::new();
            for p in start..end {
                let c_full = lap.col_ind[p];
                if is_b(
                    c_full % nx,
                    (c_full / nx) % ny,
                    c_full / (nx * ny),
                ) {
                    continue;
                }
                let c_local = interior.iter().position(|&x| x == c_full).unwrap();
                row.push((c_local, lap.values[p]));
            }
            lap_int_rows.push(row);
        }

        let dv = grid.dx() * grid.dy() * grid.dz();
        let n_bands = (nelect + 1).div_ceil(2);
        let rho = vec![0.0; grid.len()];
        Ok(Self {
            grid,
            lap,
            interior,
            lap_int_rows,
            n_int,
            dv,
            v_ext,
            nelect,
            n_bands,
            mixing: 0.5,
            with_hartree: true,
            xc,
            rho,
        })
    }

    /// Enable/disable the Hartree (electron–electron) self-interaction term.
    #[must_use]
    pub fn with_hartree(mut self, on: bool) -> Self {
        self.with_hartree = on;
        self
    }

    /// Set the density-mixing factor `α ∈ (0, 1]` for the fixed-point SCF loop.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is outside `(0, 1]`.
    #[must_use]
    pub fn with_mixing(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "mixing must be in (0, 1]");
        self.mixing = alpha;
        self
    }

    /// Add an analytic local potential (e.g. a [`crate::Pseudopotential`]) into
    /// the static external potential.
    ///
    /// # Panics
    ///
    /// Panics if `v.len()` does not match the grid size.
    pub fn add_local_potential(&mut self, v: &[f64]) {
        assert_eq!(v.len(), self.v_ext.len(), "potential length mismatch");
        for (e, &vi) in self.v_ext.iter_mut().zip(v) {
            *e += vi;
        }
    }

    /// Grid volume element `dV = dx·dy·dz`.
    #[must_use]
    pub fn volume_element(&self) -> f64 {
        self.dv
    }

    /// Current self-consistent density.
    #[must_use]
    pub fn density(&self) -> &[f64] {
        &self.rho
    }

    /// Build the Hartree potential `V_H` solving `∇²V_H = −4πρ` by conjugate
    /// gradient on the sparse (Dirichlet) Poisson operator. Returns a zero
    /// vector when Hartree is disabled.
    fn hartree_potential(&self, rho: &[f64]) -> Vec<f64> {
        let n = self.grid.len();
        if !self.with_hartree {
            return vec![0.0; n];
        }
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let is_b = |ix: usize, iy: usize, iz: usize| -> bool {
            ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1
        };
        // M = -Lap, with boundary rows pinned to identity (V_H = 0 at the box
        // wall) and all interior↔boundary couplings zeroed, so M is SPD.
        let mut m = self.lap.clone();
        for v in &mut m.values {
            *v = -*v;
        }
        let idx = |ix: usize, iy: usize, iz: usize| ix + iy * nx + iz * nx * ny;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let k = idx(ix, iy, iz);
                    let start = m.row_ptr[k];
                    let end = m.row_ptr[k + 1];
                    if is_b(ix, iy, iz) {
                        for p in start..end {
                            if m.col_ind[p] == k {
                                m.values[p] = 1.0;
                            } else {
                                m.values[p] = 0.0;
                            }
                        }
                    } else {
                        for p in start..end {
                            let c = m.col_ind[p];
                            let cix = c % nx;
                            let ciy = (c / nx) % ny;
                            let ciz = c / (nx * ny);
                            if is_b(cix, ciy, ciz) {
                                m.values[p] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        let rhs: Vec<f64> = rho.iter().map(|&r| 4.0 * PI * r).collect();
        conjugate_gradient(&m, &rhs, None, 1e-9, 5000)
    }

    /// Build the exchange-correlation potential `v_xc` from the density, using
    /// the functional's per-electron energy density and its partial derivatives
    /// (the correct GGA form `v_xc = ∂(ρ ε)/∂ρ − ∇·(∂(ρ ε)/∂∇ρ)`).
    fn xc_potential(&self, rho: &[f64], gr: &[f64]) -> Vec<f64> {
        let n = rho.len();
        let mut vxc = vec![0.0; n];
        let mut gx = vec![0.0; n];
        let mut gy = vec![0.0; n];
        let mut gz = vec![0.0; n];
        for i in 0..n {
            let r = rho[i];
            let g = gr[i];
            if r <= 0.0 {
                continue;
            }
            let eps = self.xc.energy_density(r, g);
            let deps_dr = self.xc.deriv_rho(r, g);
            let deps_dg = self.xc.deriv_gr(r, g);
            // ∂(ρ ε)/∂ρ = ε + ρ ∂ε/∂ρ.
            vxc[i] = eps + r * deps_dr;
            // Divergence term field: ρ (∂ε/∂|∇ρ|) (∇ρ/|∇ρ|).
            if g > 0.0 && deps_dg != 0.0 {
                let (rx, ry, rz) = self.gradient_at(rho, i);
                let scale = r * deps_dg / g;
                gx[i] = scale * rx;
                gy[i] = scale * ry;
                gz[i] = scale * rz;
            }
        }
        // Subtract ∇·(field) from the potential.
        for i in 0..n {
            if gx[i] != 0.0 || gy[i] != 0.0 || gz[i] != 0.0 {
                vxc[i] -= self.divergence_at(&gx, &gy, &gz, i);
            }
        }
        vxc
    }

    /// Gradient of the scalar field `f` at node `i` via central differences
    /// (one-sided at the grid boundary).
    fn gradient_at(&self, f: &[f64], i: usize) -> (f64, f64, f64) {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy, dz) = (self.grid.dx(), self.grid.dy(), self.grid.dz());
        let ix = i % nx;
        let iy = (i / nx) % ny;
        let iz = i / (nx * ny);
        let idx = |a: usize, b: usize, c: usize| a + b * nx + c * nx * ny;
        let gx = if ix > 0 && ix + 1 < nx {
            (f[idx(ix + 1, iy, iz)] - f[idx(ix - 1, iy, iz)]) / (2.0 * dx)
        } else if ix == 0 {
            (f[idx(1, iy, iz)] - f[idx(0, iy, iz)]) / dx
        } else {
            (f[idx(nx - 1, iy, iz)] - f[idx(nx - 2, iy, iz)]) / dx
        };
        let gy = if iy > 0 && iy + 1 < ny {
            (f[idx(ix, iy + 1, iz)] - f[idx(ix, iy - 1, iz)]) / (2.0 * dy)
        } else if iy == 0 {
            (f[idx(ix, 1, iz)] - f[idx(ix, 0, iz)]) / dy
        } else {
            (f[idx(ix, ny - 1, iz)] - f[idx(ix, ny - 2, iz)]) / dy
        };
        let gz = if iz > 0 && iz + 1 < nz {
            (f[idx(ix, iy, iz + 1)] - f[idx(ix, iy, iz - 1)]) / (2.0 * dz)
        } else if iz == 0 {
            (f[idx(ix, iy, 1)] - f[idx(ix, iy, 0)]) / dz
        } else {
            (f[idx(ix, iy, nz - 1)] - f[idx(ix, iy, nz - 2)]) / dz
        };
        (gx, gy, gz)
    }

    /// Divergence of the vector field `(fx, fy, fz)` at node `i`.
    fn divergence_at(&self, fx: &[f64], fy: &[f64], fz: &[f64], i: usize) -> f64 {
        let (nx, ny, nz) = (self.grid.nx(), self.grid.ny(), self.grid.nz());
        let (dx, dy, dz) = (self.grid.dx(), self.grid.dy(), self.grid.dz());
        let ix = i % nx;
        let iy = (i / nx) % ny;
        let iz = i / (nx * ny);
        let idx = |a: usize, b: usize, c: usize| a + b * nx + c * nx * ny;
        let dxc = if ix > 0 && ix + 1 < nx {
            (fx[idx(ix + 1, iy, iz)] - fx[idx(ix - 1, iy, iz)]) / (2.0 * dx)
        } else if ix == 0 {
            (fx[idx(1, iy, iz)] - fx[idx(0, iy, iz)]) / dx
        } else {
            (fx[idx(nx - 1, iy, iz)] - fx[idx(nx - 2, iy, iz)]) / dx
        };
        let dyc = if iy > 0 && iy + 1 < ny {
            (fy[idx(ix, iy + 1, iz)] - fy[idx(ix, iy - 1, iz)]) / (2.0 * dy)
        } else if iy == 0 {
            (fy[idx(ix, 1, iz)] - fy[idx(ix, 0, iz)]) / dy
        } else {
            (fy[idx(ix, ny - 1, iz)] - fy[idx(ix, ny - 2, iz)]) / dy
        };
        let dzc = if iz > 0 && iz + 1 < nz {
            (fz[idx(ix, iy, iz + 1)] - fz[idx(ix, iy, iz - 1)]) / (2.0 * dz)
        } else if iz == 0 {
            (fz[idx(ix, iy, 1)] - fz[idx(ix, iy, 0)]) / dz
        } else {
            (fz[idx(ix, iy, nz - 1)] - fz[idx(ix, iy, nz - 2)]) / dz
        };
        dxc + dyc + dzc
    }

    /// Gradient magnitude `|∇ρ|` at every node.
    #[allow(clippy::needless_range_loop)]
    fn gradient_magnitude(&self, rho: &[f64]) -> Vec<f64> {
        let n = rho.len();
        let mut gr = vec![0.0; n];
        for i in 0..n {
            let (gx, gy, gz) = self.gradient_at(rho, i);
            gr[i] = (gx * gx + gy * gy + gz * gz).sqrt();
        }
        gr
    }

    /// Restrict a full-grid vector to the interior nodes.
    fn to_interior(&self, v: &[f64]) -> Vec<f64> {
        self.interior.iter().map(|&i| v[i]).collect()
    }

    /// Matrix–vector product `y = L_int · x` for the interior Laplacian, stored
    /// as row lists.
    fn matvec_int(&self, x: &[f64]) -> Vec<f64> {
        self.lap_int_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(c, v)| v * x[c])
                    .sum::<f64>()
            })
            .collect()
    }

    /// Scatter an interior vector back to the full grid (boundary entries 0).
    fn to_full(&self, v_int: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; self.grid.len()];
        for (k, &val) in self.interior.iter().zip(v_int) {
            v[*k] = val;
        }
        v
    }

    /// Build the occupied density `ρ` (full grid) and occupation list from
    /// interior orbitals.
    fn density_from_orbitals(&self, orbitals: &[Vec<f64>]) -> (Vec<f64>, Vec<usize>) {
        let mut rho = vec![0.0; self.grid.len()];
        let mut occ = Vec::new();
        let mut filled = 0_usize;
        for orb in orbitals {
            let o = if filled + 2 <= self.nelect {
                2
            } else if filled < self.nelect {
                1
            } else {
                0
            };
            if o == 0 {
                break;
            }
            filled += o;
            occ.push(o);
            for (k, &c) in self.interior.iter().zip(orb) {
                rho[*k] += o as f64 * c * c / self.dv;
            }
        }
        (rho, occ)
    }

    /// Diagonalize `H = −½∇² + V_ext` once (no Hartree/XC), returning the
    /// occupied orbital energies, orbitals, and the bare density. Used for
    /// non-interacting analytic checks.
    ///
    /// # Panics
    ///
    /// Panics if the internal Lanczos matvec returns a wrongly-sized vector
    /// (it will not, since the Laplacian and potential share the grid size).
    pub fn solve_bare(&mut self) -> KohnSham3DResult {
        let vext_int = self.to_interior(&self.v_ext);
        let matvec = |x: &[f64]| -> Vec<f64> {
            let t = self.matvec_int(x);
            x.iter()
                .zip(&vext_int)
                .zip(&t)
                .map(|((&xi, &v), &ti)| -0.5 * ti + v * xi)
                .collect()
        };
        let (eigvals, eigvecs) = lanczos_lowest(self.n_int, self.n_bands, self.n_int.min(200), matvec);
        let (rho, occ) = self.density_from_orbitals(&eigvecs);
        let energy: f64 = eigvals.iter().zip(&occ).map(|(&e, &o)| e * o as f64).sum();
        let orbitals = eigvecs.iter().map(|v| self.to_full(v)).collect();
        self.rho = rho.clone();
        KohnSham3DResult {
            total_energy: energy,
            orbital_energies: eigvals,
            density: rho,
            orbitals,
        }
    }

    /// Solve the Kohn–Sham equations self-consistently (Hartree + XC), up to
    /// `max_iter` outer SCF loops. Returns the [`KohnSham3DResult`].
    ///
    /// # Panics
    ///
    /// Panics if the internal Lanczos matvec returns a wrongly-sized vector.
    pub fn solve(&mut self, max_iter: usize) -> KohnSham3DResult {
        let mixing = self.mixing;
        let mut eigvals = Vec::new();
        let mut eigvecs = Vec::new();
        let mut occ = Vec::new();
        for _ in 0..max_iter {
            let gr = self.gradient_magnitude(&self.rho);
            let vh = self.hartree_potential(&self.rho);
            let vxc = self.xc_potential(&self.rho, &gr);
            let veff_full: Vec<f64> = self
                .v_ext
                .iter()
                .zip(&vh)
                .zip(&vxc)
                .map(|((&ve, &vh), &vxc)| ve + vh + vxc)
                .collect();
            let veff_int = self.to_interior(&veff_full);
            let matvec = |x: &[f64]| -> Vec<f64> {
                let t = self.matvec_int(x);
                x.iter()
                    .zip(&veff_int)
                    .zip(&t)
                    .map(|((&xi, &v), &ti)| -0.5 * ti + v * xi)
                    .collect()
            };
            let (ev, orbs) = lanczos_lowest(self.n_int, self.n_bands, self.n_int.min(200), matvec);
            let (new_rho, new_occ) = self.density_from_orbitals(&orbs);
            for (r, &nr) in self.rho.iter_mut().zip(&new_rho) {
                *r = (1.0 - mixing) * *r + mixing * nr;
            }
            eigvals = ev;
            eigvecs = orbs;
            occ = new_occ;
        }
        // Final potentials for the total-energy decomposition.
        let gr = self.gradient_magnitude(&self.rho);
        let vh = self.hartree_potential(&self.rho);
        let vxc = self.xc_potential(&self.rho, &gr);
        let e_h = 0.5
            * self
                .rho
                .iter()
                .zip(&vh)
                .map(|(&r, &v)| r * v)
                .sum::<f64>()
            * self.dv;
        let e_int_rho_vxc = self
            .rho
            .iter()
            .zip(&vxc)
            .map(|(&r, &v)| r * v)
            .sum::<f64>()
            * self.dv;
        let e_xc = self.xc.total_energy(&self.rho, &gr, self.dv);
        let kinetic_plus_ext: f64 = eigvals.iter().zip(&occ).map(|(&e, &o)| e * o as f64).sum();
        let total = kinetic_plus_ext - e_h - e_int_rho_vxc + e_xc;

        let orbitals = eigvecs.iter().map(|v| self.to_full(v)).collect();
        KohnSham3DResult {
            total_energy: total,
            orbital_energies: eigvals,
            density: self.rho.clone(),
            orbitals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_sci_grid::UniformGrid3D;

    use crate::xc::Lda;

    #[test]
    fn bare_box_ground_state_is_kinetic_only_and_normalized() {
        // Cubic box [0,1]³, zero external potential: the bare solve diagonalizes
        // H = −½∇² with Dirichlet walls, so every eigenvalue is a positive
        // kinetic energy and the density integrates to the electron count.
        let grid = UniformGrid3D::new(13, 0.0, 1.0, 13, 0.0, 1.0, 13, 0.0, 1.0).unwrap();
        let v_ext = vec![0.0; grid.len()];
        let mut ks = KohnSham3D::new(grid, v_ext, 2, Box::new(Lda)).unwrap();
        let res = ks.solve_bare();
        assert!(res.total_energy.is_finite());
        assert!(res.orbital_energies[0] > 0.0, "kinetic energy must be positive");
        let dv = ks.volume_element();
        let integral: f64 = res.density.iter().sum::<f64>() * dv;
        assert!(
            (integral - 2.0).abs() < 0.15,
            "density should integrate to 2 electrons, got {integral}"
        );
    }

    #[test]
    fn full_solve_is_finite_and_conserves_electron_count() {
        let grid = UniformGrid3D::new(11, 0.0, 1.0, 11, 0.0, 1.0, 11, 0.0, 1.0).unwrap();
        let (cx, cy, cz) = (0.5, 0.5, 0.5);
        let v_ext: Vec<f64> = (0..grid.len())
            .map(|k| {
                let ix = k % grid.nx();
                let iy = (k / grid.nx()) % grid.ny();
                let iz = k / (grid.nx() * grid.ny());
                let x = grid.x0() + ix as f64 * grid.dx();
                let y = grid.y0() + iy as f64 * grid.dy();
                let z = grid.z0() + iz as f64 * grid.dz();
                0.5 * ((x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2))
            })
            .collect();
        let mut ks = KohnSham3D::new(grid, v_ext, 2, Box::new(Lda)).unwrap();
        let res = ks.solve(40);
        assert!(res.total_energy.is_finite());
        let dv = ks.volume_element();
        let integral: f64 = res.density.iter().sum::<f64>() * dv;
        assert!(
            (integral - 2.0).abs() < 0.25,
            "self-consistent density should integrate to 2 electrons, got {integral}"
        );
    }
}
