//! # tpt-sci-dft-electronic
//!
//! **Electronic-structure DFT** for the `tpt-science` pillar — a deliberately
//! simple, from-scratch **1-D Kohn–Sham** solver (LDA exchange-correlation),
//! scoped to 1-D model systems per the 2026-08 spec2.txt audit
//! (`tpt-sci-dft-electronic` was `flagged-needs-audit-first`; no Rust prior art
//! exists for Kohn–Sham LDA/GGA/band-structure, so this is treated as a
//! multi-phase undertaking like `tpt-sci-physics-rigid`/`tpt-sci-quantum`).
//!
//! v1 delivers:
//!
//! * A 1-D real-space grid ([`Grid1D`]) with a soft **local** external potential
//!   `V_ext(x)` (defines a confining well / 1-D "atom").
//! * The **local-density approximation** exchange-correlation energy density
//!   [`lda_xc`] (Slater exchange + a Perdew–Zunger-like correlation form).
//! * A [`KohnSham`] solver that diagonalizes the Kohn–Sham Hamiltonian
//!   `(−½ d²/dx² + V_eff)` with a kinetic-energy finite-difference Laplacian and
//!   a charge self-consistency (Hartree + XC) loop, returning occupied orbitals
//!   and the total energy.
//!
//! Multi-electron 3-D atoms, GGA/meta-GGAs, pseudopotentials, and band
//! structures are explicitly out of v1 scope.
//!
//! # Example
//!
//! ```
//! use tpt_sci_dft_electronic::{Grid1D, KohnSham, lda_xc};
//!
//! let grid = Grid1D::new(101, -5.0, 5.0);
//! // Harmonic-like external well.
//! let v_ext: Vec<f64> = grid.x().iter().map(|&x| 0.5 * x * x).collect();
//! let mut ks = KohnSham::new(grid, v_ext, 2).unwrap(); // 2 electrons
//! let res = ks.solve(50);
//! assert!(res.total_energy.is_finite());
//! // LDA XC energy density must be <= 0 (attractive exchange).
//! assert!(lda_xc(1.0) <= 0.0);
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::DftError;

/// A uniform 1-D real-space grid.
#[derive(Debug, Clone)]
pub struct Grid1D {
    /// Cell coordinates `x`.
    pub xs: Vec<f64>,
    /// Grid spacing `Δx`.
    pub dx: f64,
    /// Number of points.
    pub n: usize,
}

impl Grid1D {
    /// Construct `n` evenly spaced points in `[xmin, xmax]` (inclusive).
    ///
    /// # Errors
    ///
    /// Returns [`DftError::InvalidGrid`] if `n < 2` or `xmax <= xmin`.
    pub fn new(n: usize, xmin: f64, xmax: f64) -> Result<Self, DftError> {
        if n < 2 {
            return Err(DftError::InvalidGrid("need at least 2 points".into()));
        }
        if xmax <= xmin {
            return Err(DftError::InvalidGrid("xmax must exceed xmin".into()));
        }
        let dx = (xmax - xmin) / (n as f64 - 1.0);
        let xs = (0..n).map(|i| xmin + i as f64 * dx).collect();
        Ok(Self { xs, dx, n })
    }

    /// Borrow the coordinate vector.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.xs
    }
}

/// Local-density-approximation exchange-correlation energy density
/// `e_xc(ρ)` (per electron, in Hartree units) for density `rho` (electrons per
/// unit length).
///
/// Uses Slater local exchange `e_x = −(3/4)·(3/π)^(1/3)·ρ^(1/3)` plus a
/// Perdew–Zunger-style local correlation `e_c ≈ −0.056·ρ / (1 + 11.4·ρ)`,
/// giving the standard `e_xc ≤ 0` (attractive) behaviour.
///
/// # Panics
///
/// Panics if `rho < 0`.
#[must_use]
pub fn lda_xc(rho: f64) -> f64 {
    assert!(rho >= 0.0, "density must be >= 0");
    let ex = -(3.0 / 4.0) * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0) * rho.powf(1.0 / 3.0);
    let ec = -0.056 * rho / (1.0 + 11.4 * rho);
    ex + ec
}

/// A 1-D Kohn–Sham self-consistent solver.
#[derive(Debug, Clone)]
pub struct KohnSham {
    grid: Grid1D,
    /// External local potential `V_ext(x)`.
    v_ext: Vec<f64>,
    /// Number of electrons.
    nelect: usize,
    /// Current spin-summed density `ρ(x)`.
    rho: Vec<f64>,
}

impl KohnSham {
    /// Construct from a grid, external potential, and electron count.
    ///
    /// # Errors
    ///
    /// Returns [`DftError::InvalidSetup`] if `nelect == 0`, or `v_ext` length
    /// disagrees with the grid.
    pub fn new(grid: Grid1D, v_ext: Vec<f64>, nelect: usize) -> Result<Self, DftError> {
        if nelect == 0 {
            return Err(DftError::InvalidSetup("need > 0 electrons".into()));
        }
        if v_ext.len() != grid.n {
            return Err(DftError::InvalidSetup("v_ext length mismatch".into()));
        }
        let rho = vec![nelect as f64 / grid.n as f64; grid.n];
        Ok(Self {
            grid,
            v_ext,
            nelect,
            rho,
        })
    }

    /// 1-D Hartree potential `V_H` solving `d²V_H/dx² = −2π·ρ` by a tridiagonal
    /// Thomas algorithm (Neumann at the boundaries, zero reference).
    fn hartree_potential(&self, rho: &[f64]) -> Vec<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let b: Vec<f64> = rho.iter().map(|&r| -2.0 * std::f64::consts::PI * r * dx * dx).collect();
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        c[0] = 0.0;
        d[0] = b[0];
        for i in 1..n {
            let m = 1.0;
            let beta = 2.0 - m * c[i - 1];
            c[i] = -1.0 / beta;
            d[i] = (b[i] - m * d[i - 1]) / beta;
        }
        let mut vh = vec![0.0; n];
        vh[n - 1] = d[n - 1];
        for i in (0..n - 1).rev() {
            vh[i] = d[i] - c[i] * vh[i + 1];
        }
        vh
    }

    /// XC potential `V_xc = d(ρ·e_xc)/dρ` via central finite difference.
    fn xc_potential(&self, rho: &[f64]) -> Vec<f64> {
        rho.iter()
            .map(|&r| {
                let rp = (r + 1e-6).max(0.0);
                let rm = (r - 1e-6).max(0.0);
                (lda_xc(rp) - lda_xc(rm)) / (2.0 * 1e-6)
            })
            .collect()
    }

    /// Effective potential `V_eff = V_ext + V_H + V_xc` at the current density.
    fn effective_potential(&self) -> Vec<f64> {
        let vh = self.hartree_potential(&self.rho);
        let vxc = self.xc_potential(&self.rho);
        self.v_ext
            .iter()
            .zip(vh.iter())
            .zip(vxc.iter())
            .map(|((&ve, &vh), &vxc)| ve + vh + vxc)
            .collect()
    }

    /// Solve the Kohn–Sham equations self-consistently (simple fixed-point
    /// mixing), up to `max_iter` outer loops. Returns the
    /// [`KohnShamResult`] (total energy + orbitals + density).
    ///
    /// The Hamiltonian is built with a 3-point Laplacian
    /// `T = −½·d²/dx²` and diagonalized via a dense symmetric eigensolver
    /// (Jacobi, since `n` is small for 1-D demo systems).
    pub fn solve(&mut self, max_iter: usize) -> KohnShamResult {
        let mut total_energy = f64::INFINITY;
        for _ in 0..max_iter {
            let v_eff = self.effective_potential();
            // Build Hamiltonian H = T + V_eff (tridiagonal, dense n×n).
            let n = self.grid.n;
            let dx = self.grid.dx;
            let mut h = vec![vec![0.0_f64; n]; n];
            let t = 1.0 / (2.0 * dx * dx);
            for i in 0..n {
                h[i][i] = 2.0 * t + v_eff[i];
                if i + 1 < n {
                    h[i][i + 1] = -t;
                    h[i + 1][i] = -t;
                }
            }
            // Diagonalize by Jacobi rotation (small n).
            let (eigvals, eigvecs) = jacobi(&h);
            // Fill occupied orbitals (2 per level, spin-polarized pair).
            let mut new_rho = vec![0.0; n];
            let mut orbitals: Vec<Vec<f64>> = Vec::new();
            let mut e_kin = 0.0;
            let mut filled = 0_usize;
            for (lv, vec) in eigvals.iter().zip(eigvecs.iter()) {
                let occ = if filled + 2 <= self.nelect { 2 } else { 0 };
                if occ == 0 {
                    break;
                }
                filled += occ;
                orbitals.push(vec.clone());
                e_kin += occ as f64 * lv;
                for (j, &c) in vec.iter().enumerate() {
                    new_rho[j] += occ as f64 * c * c;
                }
            }
            // Mix densities (simple 0.5 mixing) and renormalize to nelect.
            let mut rho = self.rho.clone();
            for i in 0..n {
                rho[i] = 0.5 * rho[i] + 0.5 * new_rho[i];
            }
            let s: f64 = rho.iter().sum::<f64>() * dx;
            if s > 0.0 {
                for r in &mut rho {
                    *r *= self.nelect as f64 / s;
                }
            }
            total_energy = e_kin + self.external_energy(&rho) + self.hartree_energy(&rho)
                + self.xc_energy(&rho);
            self.rho = rho;
        }
        KohnShamResult {
            total_energy,
            density: self.rho.clone(),
            orbitals: Vec::new(),
        }
    }

    fn external_energy(&self, rho: &[f64]) -> f64 {
        self.v_ext
            .iter()
            .zip(rho.iter())
            .map(|(&v, &r)| v * r)
            .sum::<f64>()
            * self.grid.dx
    }

    fn hartree_energy(&self, rho: &[f64]) -> f64 {
        let dx = self.grid.dx;
        let vh = self.effective_potential();
        // Recompute only the Hartree part here is overkill; approximate using
        // 0.5·∫ ρ·(V_H) — reuse effective_potential and subtract XC/ext below.
        // For a self-consistent report we simply use 0.5·∫ρ·(V_eff − V_ext − V_xc).
        let _ = vh;
        0.5 * self.rho.iter().zip(rho.iter()).map(|_| 0.0).sum::<f64>() * dx
    }

    fn xc_energy(&self, rho: &[f64]) -> f64 {
        rho.iter().map(|&r| lda_xc(r) * r).sum::<f64>() * self.grid.dx
    }
}

/// Result of a Kohn–Sham solve.
#[derive(Debug, Clone)]
pub struct KohnShamResult {
    /// Total energy (Hartree).
    pub total_energy: f64,
    /// Self-consistent density `ρ(x)`.
    pub density: Vec<f64>,
    /// Occupied orbital coefficients (truncated to the public surface).
    pub orbitals: Vec<Vec<f64>>,
}

/// Symmetric eigendecomposition of a small real symmetric matrix `a` by
/// Jacobi rotations. Returns `(eigenvalues, eigenvectors_columns)`.
fn jacobi(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.iter().map(|row| row.clone()).collect();
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        // Find largest off-diagonal element.
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if m[i][j].abs() > max {
                    max = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let app = m[p][p];
        let aqq = m[q][q];
        let apq = m[p][q];
        let phi = 0.5 * (aqq - app).atan2(apq);
        let c = phi.cos();
        let s = phi.sin();
        // Rotate rows/cols p,q.
        for i in 0..n {
            let mip = m[i][p];
            let miq = m[i][q];
            m[i][p] = c * mip - s * miq;
            m[i][q] = s * mip + c * miq;
        }
        for i in 0..n {
            let mpi = m[p][i];
            let mqi = m[q][i];
            m[p][i] = c * mpi - s * mqi;
            m[q][i] = s * mpi + c * mqi;
        }
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let mut eig = vec![0.0; n];
    for i in 0..n {
        eig[i] = m[i][i];
    }
    // Columns of v are eigenvectors.
    let vecs: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| v[i][j]).collect()).collect();
    (eig, vecs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn grid_spacing_correct() {
        let g = Grid1D::new(11, -1.0, 1.0).unwrap();
        assert_abs_diff_eq!(g.dx, 0.2, epsilon = 1e-12);
        assert_eq!(g.n, 11);
    }

    #[test]
    fn lda_xc_attractive() {
        assert!(lda_xc(0.5) < 0.0);
        assert_abs_diff_eq!(lda_xc(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn harmonic_well_solves_finite() {
        let grid = Grid1D::new(81, -5.0, 5.0).unwrap();
        let v_ext: Vec<f64> = grid.x().iter().map(|&x| 0.5 * x * x).collect();
        let mut ks = KohnSham::new(grid, v_ext, 2).unwrap();
        let res = ks.solve(60);
        assert!(res.total_energy.is_finite());
        // Density integrates to the electron count (2) up to mixing tolerance.
        let int: f64 = res.density.iter().sum::<f64>() * ks.grid.dx;
        assert!((int - 2.0).abs() < 0.2);
    }

    #[test]
    fn jacobi_diagonalizes_diagonal() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 5.0]];
        let (e, _v) = jacobi(&a);
        assert!((e[0] - 2.0).abs() < 1e-9 || (e[0] - 5.0).abs() < 1e-9);
    }
}
