//! Pairwise forces: Lennard-Jones (direct and cell-list), and EAM dispatch.

use super::*;

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
    let mut forces = vec![DVector::zeros(dim); n];
    if n < 2 {
        return forces;
    }
    let epsilon = 1.0;
    let rcut = 2.5 * sigma;
    let rcut2 = rcut * rcut;
    let sig6 = sigma.powi(6);
    let sig12 = sig6 * sig6;

    let wrap = |x: f64| -> f64 {
        if !box_len.is_finite() {
            return x;
        }
        // Minimum-image convention into [-box_len/2, box_len/2).
        let half = box_len * 0.5;
        let mut y = x - box_len * (x / box_len).floor();
        if y > half {
            y -= box_len;
        }
        y
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r2 = dr.dot(&dr);
            if r2 >= rcut2 || r2 <= 0.0 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r12 = inv_r6 * inv_r6;
            // Radial LJ force magnitude F_r = -dU/dr = -24·ε·(2·(σ/r)^12 - (σ/r)^6)/r.
            let f_mag = -24.0 * epsilon * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2;
            let fvec = dr * f_mag;
            forces[i] = forces[i].clone() + fvec.clone();
            forces[j] = forces[j].clone() - fvec;
        }
    }
    forces
}

/// Stateless helpers over a set of [`Particle`]s.
pub struct Forces;

impl Forces {
    /// Compute Lennard-Jones forces (see [`lennard_jones`]) and write them into
    /// each particle's `force` field, returning the total potential energy.
    #[must_use]
    pub fn lennard_jones(particles: &mut [Particle], box_len: f64, sigma: f64) -> f64 {
        let n = particles.len();
        let dim = particles.first().map_or(3, |p| p.position.len());
        for p in particles.iter_mut() {
            p.force = DVector::zeros(dim);
        }
        if n < 2 {
            return 0.0;
        }
        let epsilon = 1.0;
        let rcut = 2.5 * sigma;
        let rcut2 = rcut * rcut;
        let sig6 = sigma.powi(6);
        let sig12 = sig6 * sig6;
        let u_cut = 4.0 * epsilon * (sig12 / rcut2.powi(6) - sig6 / rcut2.powi(3));

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

        let mut energy = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let raw = particles[j].position.clone() - particles[i].position.clone();
                let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
                let r2 = dr.dot(&dr);
                if r2 >= rcut2 || r2 <= 0.0 {
                    continue;
                }
                let inv_r2 = 1.0 / r2;
                let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                let inv_r12 = inv_r6 * inv_r6;
                let u = 4.0 * epsilon * (sig12 * inv_r12 - sig6 * inv_r6) - u_cut;
                energy += u;
                let f_mag = -24.0 * epsilon * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2;
                let fvec = dr * f_mag;
                particles[i].force = particles[i].force.clone() + fvec.clone();
                particles[j].force = particles[j].force.clone() - fvec;
            }
        }
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
        for p in particles.iter_mut() {
            p.force = DVector::zeros(dim);
        }
        let n = particles.len();
        if n < 2 {
            return Ok(0.0);
        }
        let epsilon = 1.0;
        let rcut = 2.5 * sigma;
        let sig6 = sigma.powi(6);
        let sig12 = sig6 * sig6;
        let rcut2 = rcut * rcut;
        let u_cut = 4.0 * epsilon * (sig12 / rcut2.powi(6) - sig6 / rcut2.powi(3));

        let wrap = |x: f64| -> f64 {
            let half = box_len * 0.5;
            let mut y = x - box_len * (x / box_len).floor();
            if y > half {
                y -= box_len;
            }
            y
        };

        let cell_list = CellList::build(particles, box_len, rcut)?;
        let mut energy = 0.0;
        for (i, j) in cell_list.neighbor_pairs(particles) {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let dim = raw.len();
            let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r2 = dr.dot(&dr);
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r12 = inv_r6 * inv_r6;
            let u = 4.0 * epsilon * (sig12 * inv_r12 - sig6 * inv_r6) - u_cut;
            energy += u;
            let f_mag = -24.0 * epsilon * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2;
            let fvec = dr * f_mag;
            particles[i].force = particles[i].force.clone() + fvec.clone();
            particles[j].force = particles[j].force.clone() - fvec;
        }
        Ok(energy)
    }
}
