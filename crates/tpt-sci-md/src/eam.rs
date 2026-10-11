//! Finnis–Sinclair-style embedded-atom (EAM) potential.

use super::*;

/// Finnis-Sinclair-style embedded-atom-method (EAM) potential parameters.
///
/// Total energy `V = Σ_{i<j} φ(r_ij) + Σ_i F(ρ_i)` with local electron density
/// `ρ_i = Σ_{j≠i} f(r_ij)`. This crate uses the classic Finnis-Sinclair forms
///
/// * pairwise repulsion `φ(r) = (c0 + c1·r + c2·r²)·(r - cutoff_phi)²` for
///   `r < cutoff_phi`, else `0`;
/// * density contribution `f(r) = (r - cutoff_rho)²` for `r < cutoff_rho`, else `0`;
/// * embedding function `F(ρ) = -embed_a·√ρ` (square-root embedding, the
///   standard Finnis-Sinclair choice, which makes the many-body term more
///   negative as local density grows — i.e. attractive metallic bonding).
///
/// [`EamParams::generic`] provides parameters chosen only to be physically
/// sound (short-range repulsive, longer-range attractive via the embedding
/// term) — they are **not** fit to reproduce any specific real metal.
#[derive(Debug, Clone)]
pub struct EamParams {
    /// Cut-off for the pairwise repulsive term `φ`.
    pub cutoff_phi: f64,
    /// Cut-off for the density contribution `f`.
    pub cutoff_rho: f64,
    /// Constant coefficient of the `φ` polynomial prefactor.
    pub phi_c0: f64,
    /// Linear coefficient of the `φ` polynomial prefactor.
    pub phi_c1: f64,
    /// Quadratic coefficient of the `φ` polynomial prefactor.
    pub phi_c2: f64,
    /// Embedding-function strength `A` in `F(ρ) = -A·√ρ`.
    pub embed_a: f64,
}

impl EamParams {
    /// Construct validated EAM parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEam`] if `cutoff_phi <= 0`, `cutoff_rho <= 0`,
    /// or `embed_a <= 0`.
    pub fn new(
        cutoff_phi: f64,
        cutoff_rho: f64,
        phi_c0: f64,
        phi_c1: f64,
        phi_c2: f64,
        embed_a: f64,
    ) -> Result<Self, MdError> {
        if cutoff_phi <= 0.0 {
            return Err(MdError::InvalidEam(format!(
                "cutoff_phi must be > 0, got {cutoff_phi}"
            )));
        }
        if cutoff_rho <= 0.0 {
            return Err(MdError::InvalidEam(format!(
                "cutoff_rho must be > 0, got {cutoff_rho}"
            )));
        }
        if embed_a <= 0.0 {
            return Err(MdError::InvalidEam(format!(
                "embed_a must be > 0, got {embed_a}"
            )));
        }
        Ok(Self {
            cutoff_phi,
            cutoff_rho,
            phi_c0,
            phi_c1,
            phi_c2,
            embed_a,
        })
    }

    /// Generic, physically-sound (but not element-specific) parameters: short
    /// range repulsive, attractive at intermediate range.
    #[must_use]
    pub fn generic() -> Self {
        Self {
            cutoff_phi: 3.0,
            cutoff_rho: 3.0,
            phi_c0: 1.0,
            phi_c1: 0.0,
            phi_c2: 0.0,
            embed_a: 2.0,
        }
    }

    fn phi(&self, r: f64) -> f64 {
        if r >= self.cutoff_phi {
            return 0.0;
        }
        let p = self.phi_c0 + self.phi_c1 * r + self.phi_c2 * r * r;
        p * (r - self.cutoff_phi).powi(2)
    }

    fn dphi_dr(&self, r: f64) -> f64 {
        if r >= self.cutoff_phi {
            return 0.0;
        }
        let p = self.phi_c0 + self.phi_c1 * r + self.phi_c2 * r * r;
        let dp = self.phi_c1 + 2.0 * self.phi_c2 * r;
        let c = self.cutoff_phi;
        dp * (r - c).powi(2) + p * 2.0 * (r - c)
    }

    fn f_rho(&self, r: f64) -> f64 {
        if r >= self.cutoff_rho {
            return 0.0;
        }
        (r - self.cutoff_rho).powi(2)
    }

    fn df_dr(&self, r: f64) -> f64 {
        if r >= self.cutoff_rho {
            return 0.0;
        }
        2.0 * (r - self.cutoff_rho)
    }

    fn embed(&self, rho: f64) -> f64 {
        -self.embed_a * rho.max(0.0).sqrt()
    }

    fn embed_prime(&self, rho: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        -self.embed_a / (2.0 * rho.sqrt())
    }
}

/// Compute EAM forces and total potential energy without mutating `particles`.
pub(crate) fn eam_forces_energy(
    particles: &[Particle],
    box_len: f64,
    params: &EamParams,
) -> (Vec<DVector<f64>>, f64) {
    let n = particles.len();
    let dim = particles.first().map_or(3, |p| p.position.len());
    if n < 2 {
        return (vec![DVector::zeros(dim); n], 0.0);
    }
    let rcut = params.cutoff_phi.max(params.cutoff_rho);
    let rcut2 = rcut * rcut;

    let wrap = |x: f64| -> f64 {
        if !box_len.is_finite() {
            return x;
        }
        let half = box_len * 0.5;
        let mut y = x - box_len * (x / box_len).floor();
        if y > half {
            y -= box_len;
        }
        y
    };

    let mut pos = Vec::with_capacity(n * dim);
    for p in particles {
        pos.extend(p.position.iter().copied());
    }

    // Pass 1: pairwise distances and local densities ρ_i. Pair data is kept in
    // flat buffers (`pair_ij`, `pair_r`, and `pair_dr` with `dim` entries per pair).
    let mut rho = vec![0.0_f64; n];
    let mut pair_ij: Vec<(usize, usize)> = Vec::new();
    let mut pair_r: Vec<f64> = Vec::new();
    let mut pair_dr: Vec<f64> = Vec::new();
    let mut dr = vec![0.0; dim];
    for i in 0..n {
        let pi = &pos[i * dim..(i + 1) * dim];
        for j in (i + 1)..n {
            let pj = &pos[j * dim..(j + 1) * dim];
            let mut r2 = 0.0;
            for k in 0..dim {
                dr[k] = wrap(pj[k] - pi[k]);
                r2 += dr[k] * dr[k];
            }
            if r2 >= rcut2 || r2 <= 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let f = params.f_rho(r);
            rho[i] += f;
            rho[j] += f;
            pair_ij.push((i, j));
            pair_r.push(r);
            pair_dr.extend_from_slice(&dr);
        }
    }

    let fprime: Vec<f64> = rho.iter().map(|&rh| params.embed_prime(rh)).collect();
    let mut energy: f64 = rho.iter().map(|&rh| params.embed(rh)).sum();

    // Pass 2: pairwise force, including the embedding chain-rule term
    // (F'(ρ_i) + F'(ρ_j))·f'(r_ij), the standard EAM force expression.
    let mut forces = vec![0.0_f64; n * dim];
    for (p, (&(i, j), &r)) in pair_ij.iter().zip(&pair_r).enumerate() {
        energy += params.phi(r);
        let dudr = params.dphi_dr(r) + (fprime[i] + fprime[j]) * params.df_dr(r);
        let f_mag = dudr / r;
        for k in 0..dim {
            let f = pair_dr[p * dim + k] * f_mag;
            forces[i * dim + k] += f;
            forces[j * dim + k] -= f;
        }
    }

    let forces = forces
        .chunks_exact(dim)
        .map(DVector::from_row_slice)
        .collect();
    (forces, energy)
}

/// Compute the force on every particle from a Finnis-Sinclair-style EAM
/// potential (see [`EamParams`]). Analogous in shape to [`lennard_jones`].
#[must_use]
pub fn eam_forces(particles: &[Particle], box_len: f64, params: &EamParams) -> Vec<DVector<f64>> {
    eam_forces_energy(particles, box_len, params).0
}
