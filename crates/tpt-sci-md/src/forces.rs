//! Pairwise forces: Lennard-Jones (direct and cell-list), and EAM dispatch.

use super::*;

/// Lennard-Jones constants and pair kernel shared by every force routine.
///
/// All interactions use `epsilon = 1`; the potential is truncated at
/// `rcut = 2.5·sigma` and shifted by `u_cut` so it vanishes at the cut-off.
struct Lj {
    box_len: f64,
    rcut2: f64,
    sig6: f64,
    sig12: f64,
    u_cut: f64,
}

impl Lj {
    fn new(box_len: f64, sigma: f64) -> Self {
        let rcut = 2.5 * sigma;
        let rcut2 = rcut * rcut;
        let sig6 = sigma.powi(6);
        let sig12 = sig6 * sig6;
        let u_cut = 4.0 * (sig12 / rcut2.powi(6) - sig6 / rcut2.powi(3));
        Self {
            box_len,
            rcut2,
            sig6,
            sig12,
            u_cut,
        }
    }

    /// Minimum-image convention into `[-box_len/2, box_len/2)`.
    #[inline]
    fn wrap(&self, x: f64) -> f64 {
        if !self.box_len.is_finite() {
            return x;
        }
        let half = self.box_len * 0.5;
        let mut y = x - self.box_len * (x / self.box_len).floor();
        if y > half {
            y -= self.box_len;
        }
        y
    }

    /// Evaluate the pair `(pi, pj)`. Writes the minimum-image displacement
    /// `pj - pi` into `dr` and returns `(energy, f_mag)` when the pair lies inside
    /// the cut-off, where the force on `i` is `dr * f_mag` (and its negation on `j`).
    ///
    /// `f_mag` is the radial LJ force magnitude `F_r = -dU/dr`, divided by `r`:
    /// `-24·ε·(2·(σ/r)^12 - (σ/r)^6)/r²`.
    #[inline]
    fn pair(&self, pi: &[f64], pj: &[f64], dr: &mut [f64]) -> Option<(f64, f64)> {
        let mut r2 = 0.0;
        for k in 0..dr.len() {
            let d = self.wrap(pj[k] - pi[k]);
            dr[k] = d;
            r2 += d * d;
        }
        if r2 >= self.rcut2 || r2 <= 0.0 {
            return None;
        }
        let inv_r2 = 1.0 / r2;
        let inv_r6 = inv_r2 * inv_r2 * inv_r2;
        let inv_r12 = inv_r6 * inv_r6;
        let u = 4.0 * (self.sig12 * inv_r12 - self.sig6 * inv_r6) - self.u_cut;
        let f_mag = -24.0 * (2.0 * self.sig12 * inv_r12 - self.sig6 * inv_r6) * inv_r2;
        Some((u, f_mag))
    }
}

/// Copy all positions into one contiguous `n * dim` buffer.
fn flatten_positions(particles: &[Particle], dim: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(particles.len() * dim);
    for p in particles {
        flat.extend(p.position.iter().copied());
    }
    flat
}

/// Accumulate the force of every pair in `pairs` into `forces` (`n * dim`),
/// returning the summed shifted pair energy.
fn accumulate_pairs(
    lj: &Lj,
    pos: &[f64],
    dim: usize,
    pairs: impl Iterator<Item = (usize, usize)>,
    forces: &mut [f64],
) -> f64 {
    let mut dr = vec![0.0; dim];
    let mut energy = 0.0;
    for (i, j) in pairs {
        let (pi, pj) = (&pos[i * dim..(i + 1) * dim], &pos[j * dim..(j + 1) * dim]);
        if let Some((u, f_mag)) = lj.pair(pi, pj, &mut dr) {
            energy += u;
            for k in 0..dim {
                let f = dr[k] * f_mag;
                forces[i * dim + k] += f;
                forces[j * dim + k] -= f;
            }
        }
    }
    energy
}

/// All unordered pairs `(i, j)`, `i < j`, of `n` particles.
fn all_pairs(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |i| ((i + 1)..n).map(move |j| (i, j)))
}

/// Compute the force on every particle from a pairwise Lennard-Jones 12-6
/// potential `U(r) = 4·ε·[(σ/r)^12 - (σ/r)^6]`, truncated at `rcut` and shifted
/// so the force vanishes continuously at the cut-off.
///
/// All interactions use the same `epsilon`/`sigma`; `box_len` is the cubic
/// periodic-box side length (use `f64::INFINITY` for free space / no wrapping).
#[must_use]
pub fn lennard_jones(particles: &[Particle], box_len: f64, sigma: f64) -> Vec<DVector<f64>> {
    let n = particles.len();
    let dim = particles.first().map_or(3, |p| p.position.len());
    if n < 2 {
        return vec![DVector::zeros(dim); n];
    }
    let lj = Lj::new(box_len, sigma);
    let pos = flatten_positions(particles, dim);
    let mut forces = vec![0.0; n * dim];
    let _ = accumulate_pairs(&lj, &pos, dim, all_pairs(n), &mut forces);
    forces
        .chunks_exact(dim)
        .map(DVector::from_row_slice)
        .collect()
}

/// Stateless helpers over a set of [`Particle`]s.
pub struct Forces;

/// Overwrite each particle's `force` from a flat `n * dim` buffer.
fn store_forces(particles: &mut [Particle], dim: usize, forces: &[f64]) {
    for (p, f) in particles.iter_mut().zip(forces.chunks_exact(dim)) {
        p.force = DVector::from_row_slice(f);
    }
}

impl Forces {
    /// Compute Lennard-Jones forces (see [`lennard_jones`]) and write them into
    /// each particle's `force` field, returning the total potential energy.
    #[must_use]
    pub fn lennard_jones(particles: &mut [Particle], box_len: f64, sigma: f64) -> f64 {
        let n = particles.len();
        let dim = particles.first().map_or(3, |p| p.position.len());
        if n < 2 {
            for p in particles.iter_mut() {
                p.force = DVector::zeros(dim);
            }
            return 0.0;
        }
        let lj = Lj::new(box_len, sigma);
        let pos = flatten_positions(particles, dim);
        let mut forces = vec![0.0; n * dim];
        let energy = accumulate_pairs(&lj, &pos, dim, all_pairs(n), &mut forces);
        store_forces(particles, dim, &forces);
        energy
    }
}

impl Forces {
    /// Compute EAM forces (see [`eam_forces`]) and write them into each
    /// particle's `force` field, returning the total potential energy.
    #[must_use]
    pub fn eam(particles: &mut [Particle], box_len: f64, params: &EamParams) -> f64 {
        let (forces, energy) = eam_forces_energy(particles, box_len, params);
        for (p, f) in particles.iter_mut().zip(forces) {
            p.force = f;
        }
        energy
    }
}

impl Forces {
    /// Compute Lennard-Jones forces exactly as [`Forces::lennard_jones`], but
    /// using a [`CellList`] to find interacting pairs instead of the `O(n²)`
    /// scan — an alternative, faster evaluation path for larger systems.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidCellList`] under the same conditions as
    /// [`CellList::build`] (in particular, `box_len` must be finite and
    /// `2.5·sigma <= box_len/2`).
    pub fn lennard_jones_cells(
        particles: &mut [Particle],
        box_len: f64,
        sigma: f64,
    ) -> Result<f64, MdError> {
        let dim = particles.first().map_or(3, |p| p.position.len());
        let n = particles.len();
        if n < 2 {
            for p in particles.iter_mut() {
                p.force = DVector::zeros(dim);
            }
            return Ok(0.0);
        }
        let lj = Lj::new(box_len, sigma);
        let cell_list = CellList::build(particles, box_len, 2.5 * sigma)?;
        let pos = flatten_positions(particles, dim);
        let mut forces = vec![0.0; n * dim];
        let pairs = cell_list.neighbor_pairs(particles);
        let energy = accumulate_pairs(&lj, &pos, dim, pairs.into_iter(), &mut forces);
        store_forces(particles, dim, &forces);
        Ok(energy)
    }
}

#[cfg(feature = "rayon")]
impl Forces {
    /// Parallel version of [`Forces::lennard_jones`]: each particle's force is
    /// summed independently over all of its neighbours, so the work spreads
    /// across threads. Forces are written into each particle's `force` field and
    /// the total potential energy is returned.
    ///
    /// The result equals the serial routine up to floating-point summation
    /// order. The serial routine visits each pair once; this one visits each pair
    /// twice (once from each end) to avoid write conflicts between threads.
    #[must_use]
    pub fn lennard_jones_parallel(particles: &mut [Particle], box_len: f64, sigma: f64) -> f64 {
        use rayon::prelude::*;

        let n = particles.len();
        let dim = particles.first().map_or(3, |p| p.position.len());
        if n < 2 {
            for p in particles.iter_mut() {
                p.force = DVector::zeros(dim);
            }
            return 0.0;
        }
        let lj = Lj::new(box_len, sigma);

        // Snapshot the positions so every particle can read all of them while its
        // own force is written.
        let pos = flatten_positions(particles, dim);
        let per_particle: Vec<f64> = particles
            .par_iter_mut()
            .enumerate()
            .map(|(i, p)| {
                let mut force = vec![0.0; dim];
                let mut dr = vec![0.0; dim];
                let pi = &pos[i * dim..(i + 1) * dim];
                let mut u_i = 0.0;
                for (j, pj) in pos.chunks_exact(dim).enumerate() {
                    if j == i {
                        continue;
                    }
                    if let Some((u, f_mag)) = lj.pair(pi, pj, &mut dr) {
                        u_i += u;
                        for k in 0..dim {
                            force[k] += dr[k] * f_mag;
                        }
                    }
                }
                p.force = DVector::from_row_slice(&force);
                u_i
            })
            .collect();
        // Each pair contributes to both ends, so the total is half the sum.
        0.5 * per_particle.iter().sum::<f64>()
    }
}
