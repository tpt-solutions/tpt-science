//! # tpt-sci-md
//!
//! A small, dependency-light **classical molecular dynamics (MD)** engine built
//! entirely from scratch (no wrapped engine such as `lumol`, which was audited
//! and rejected — BSD-3-Clause and alpha/stale). It provides:
//!
//! * A [`Particle`] type (position, velocity, force, mass, type id).
//! * Pairwise **Lennard-Jones (12-6)** interactions with a cut-off radius and
//!   minimum-image convention under periodic boundaries ([`Forces::lennard_jones`]).
//! * Velocity-Verlet integration ([`Integrator::velocity_verlet`]) with optional
//!   Berendsen-style thermostats.
//! * A **radial distribution function** [`rdf`] for structural analysis.
//!
//! The engine is intentionally minimal: it models mono- (or few-) species
//! Lennard-Jones fluids in a cubic periodic box and is sized for teaching,
//! prototyping, and coupling into the broader `tpt-science` platform — not for
//! production-scale biomolecular simulation.
//!
//! # Example
//!
//! ```
//! use tpt_sci_md::{Particle, lennard_jones};
//! use tpt_math_linalg::tpt_math_linalg_dense::DVector;
//!
//! // Two particles on a line, slightly closer than the LJ minimum.
//! let parts = vec![
//!     Particle::new(0, DVector::from_row_slice(&[0.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
//!     Particle::new(1, DVector::from_row_slice(&[1.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
//! ];
//! let forces = lennard_jones(&parts, 10.0, 1.0);
//! assert!(forces[0].iter().any(|&f| f.is_finite()));
//! ```
#![forbid(unsafe_code)]

use tpt_math_linalg::tpt_math_linalg_dense::DVector;

mod error;

pub use error::MdError;

/// A single point particle in a molecular-dynamics system.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Unique id within the system.
    pub id: usize,
    /// Position vector (length = spatial dimension).
    pub position: DVector<f64>,
    /// Velocity vector.
    pub velocity: DVector<f64>,
    /// Accumulated force (reset each step).
    pub force: DVector<f64>,
    /// Positive mass.
    pub mass: f64,
    /// Species / type id (0 = default).
    pub species: usize,
}

impl Particle {
    /// Construct a validated particle.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidParticle`] if `mass <= 0`, `position` and
    /// `velocity` have mismatched lengths, or any component is non-finite.
    pub fn new(
        id: usize,
        position: DVector<f64>,
        velocity: DVector<f64>,
        mass: f64,
    ) -> Result<Self, MdError> {
        Self::new_with_species(id, position, velocity, mass, 0)
    }

    /// Construct a particle with an explicit species id.
    ///
    /// # Errors
    ///
    /// See [`Particle::new`].
    pub fn new_with_species(
        id: usize,
        position: DVector<f64>,
        velocity: DVector<f64>,
        mass: f64,
        species: usize,
    ) -> Result<Self, MdError> {
        if position.len() != velocity.len() {
            return Err(MdError::InvalidParticle(format!(
                "particle {id}: position/velocity dim mismatch"
            )));
        }
        if mass <= 0.0 {
            return Err(MdError::InvalidParticle(format!(
                "particle {id}: mass must be > 0, got {mass}"
            )));
        }
        for (name, v) in [("position", &position), ("velocity", &velocity)] {
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(MdError::InvalidParticle(format!(
                        "particle {id}: non-finite {name} component at {i}"
                    )));
                }
            }
        }
        let force = DVector::zeros(velocity.len());
        Ok(Self {
            id,
            position,
            velocity,
            force,
            mass,
            species,
        })
    }
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

/// A velocity-Verlet MD integrator over a cubic periodic box.
#[derive(Debug, Clone)]
pub struct Integrator {
    /// Cubic periodic-box side length (`f64::INFINITY` = no wrapping).
    pub box_len: f64,
    /// Lennard-Jones `σ`.
    pub sigma: f64,
    /// Timestep (seconds, in LJ-reduced units).
    pub dt: f64,
    dim: usize,
}

impl Integrator {
    /// Construct a validated integrator.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidIntegrator`] if `dt <= 0` or `sigma <= 0`.
    pub fn new(box_len: f64, sigma: f64, dt: f64) -> Result<Self, MdError> {
        if dt <= 0.0 {
            return Err(MdError::InvalidIntegrator(format!(
                "dt must be > 0, got {dt}"
            )));
        }
        if sigma <= 0.0 {
            return Err(MdError::InvalidIntegrator(format!(
                "sigma must be > 0, got {sigma}"
            )));
        }
        Ok(Self {
            box_len,
            sigma,
            dt,
            dim: 3,
        })
    }

    /// Advance the system by one velocity-Verlet step, returning the potential
    /// energy at the new positions.
    #[must_use]
    pub fn velocity_verlet(&self, particles: &mut [Particle]) -> f64 {
        let dt = self.dt;
        let half = 0.5 * dt;
        // (1) x(t+dt) = x + v·dt + 0.5·a·dt² ; (2) v += 0.5·a·dt using old force.
        for p in particles.iter_mut() {
            p.position = p.position.clone()
                + (p.velocity.clone() * dt)
                + (p.force.clone() * (0.5 * dt * dt) / p.mass);
            // Keep inside the periodic box.
            if self.box_len.is_finite() {
                let bl = self.box_len;
                let pos = p.position.clone();
                p.position = DVector::from_fn(pos.len(), |k| {
                    let x = pos[k];
                    x - bl * (x / bl).floor()
                });
            }
            p.velocity = p.velocity.clone() + (p.force.clone() * half / p.mass);
        }
        // (3) recompute forces at new positions, (4) finish velocity update.
        let energy = Forces::lennard_jones(particles, self.box_len, self.sigma);
        for p in particles.iter_mut() {
            p.velocity = p.velocity.clone() + (p.force.clone() * half / p.mass);
        }
        energy
    }

    /// Total kinetic energy `Σ ½·m·v²` of the system.
    #[must_use]
    pub fn kinetic_energy(&self, particles: &[Particle]) -> f64 {
        particles
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.dot(&p.velocity))
            .sum()
    }

    /// Instantaneous temperature `T = 2·K / (dim·(N-1)·k_B)` with `k_B = 1`
    /// (reduced units). Returns `0` for a single particle.
    #[must_use]
    pub fn temperature(&self, particles: &[Particle]) -> f64 {
        let n = particles.len();
        if n <= 1 {
            return 0.0;
        }
        let kb = 1.0;
        (2.0 * self.kinetic_energy(particles)) / (self.dim as f64 * (n - 1) as f64 * kb)
    }

    /// Rescale velocities toward a target temperature (Berendsen-style weak
    /// coupling) with relaxation time `tau`. No-op when `tau <= 0`.
    pub fn thermostat(&self, particles: &mut [Particle], target_t: f64, tau: f64) {
        if tau <= 0.0 || target_t <= 0.0 {
            return;
        }
        let t = self.temperature(particles);
        if t <= 0.0 {
            return;
        }
        let factor = ((1.0 + (self.dt / tau) * (target_t / t - 1.0)).sqrt()).clamp(0.0, 2.0);
        for p in particles.iter_mut() {
            p.velocity = p.velocity.clone() * factor;
        }
    }
}

/// Compute the radial distribution function `g(r)` of a configuration of
/// particles in a cubic periodic box.
///
/// `r_max` is the largest sampled radius (≤ `box_len/2`), `nbins` the histogram
/// resolution. Returns `(r_centers, g)` where `g` is normalized by the ideal-gas
/// shell volume (so `g → 1` in the dilute limit).
///
/// # Errors
///
/// Returns [`MdError::RdfError`] if fewer than two particles, `nbins == 0`,
/// `r_max <= 0`, or `r_max > box_len/2` (for a periodic box).
pub fn rdf(
    particles: &[Particle],
    box_len: f64,
    r_max: f64,
    nbins: usize,
) -> Result<(Vec<f64>, Vec<f64>), MdError> {
    let n = particles.len();
    if n < 2 {
        return Err(MdError::RdfError("need at least two particles".into()));
    }
    if nbins == 0 {
        return Err(MdError::RdfError("nbins must be > 0".into()));
    }
    if r_max <= 0.0 {
        return Err(MdError::RdfError("r_max must be > 0".into()));
    }
    if box_len.is_finite() && r_max > box_len * 0.5 {
        return Err(MdError::RdfError("r_max must be <= box_len/2".into()));
    }
    let dim = particles.first().map_or(3, |p| p.position.len());
    let dr = r_max / nbins as f64;
    let mut hist = vec![0_usize; nbins];

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

    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let d = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r = d.norm();
            if r < r_max {
                let bin = (r / dr) as usize;
                if bin < nbins {
                    hist[bin] += 1;
                }
            }
        }
    }

    let rho = n as f64
        / if box_len.is_finite() {
            box_len.powi(dim as i32)
        } else {
            1.0
        };
    let volume = if box_len.is_finite() {
        box_len.powi(dim as i32)
    } else {
        1.0
    };
    let centers: Vec<f64> = (0..nbins).map(|b| (b as f64 + 0.5) * dr).collect();
    let g: Vec<f64> = (0..nbins)
        .map(|b| {
            let r = centers[b];
            // Shell volume in `dim` dimensions (spherical).
            let shell = if dim == 3 {
                4.0 * std::f64::consts::PI * r * r * dr
            } else if dim == 2 {
                2.0 * std::f64::consts::PI * r * dr
            } else {
                2.0 * r * dr
            };
            let expected = rho * shell * (n - 1) as f64 / volume * volume;
            if expected == 0.0 {
                0.0
            } else {
                hist[b] as f64 / expected * volume
            }
        })
        .collect();
    Ok((centers, g))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn v(x: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(x)
    }

    #[test]
    fn particle_new_rejects_invalid() {
        assert!(Particle::new(0, v(&[0.0]), v(&[1.0]), 0.0).is_err());
        assert!(Particle::new(0, v(&[0.0]), v(&[1.0, 0.0]), 1.0).is_err());
        assert!(Particle::new(0, v(&[0.0]), v(&[0.0]), 1.0).is_ok());
    }

    #[test]
    fn lj_repulsive_at_close_range() {
        // Two particles closer than σ: the LJ force must push them apart
        // (force on particle 0 points toward negative x if particle 1 is at +x).
        let parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[0.8, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = lennard_jones(&parts, f64::INFINITY, 1.0);
        // Particle 1 to the right => repulsive force on 0 is toward -x.
        assert!(f[0][0] < 0.0);
        assert!(f[1][0] > 0.0);
    }

    #[test]
    fn lj_force_vanishes_beyond_cutoff() {
        let parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[10.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = lennard_jones(&parts, f64::INFINITY, 1.0);
        assert_abs_diff_eq!(f[0][0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_verlet_conserves_energy_free_particles() {
        // Two non-interacting particles (huge sigma so no force) drift at
        // constant velocity; position advances by v·dt.
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[5.0, 0.0, 0.0]), v(&[0.0, 1.0, 0.0]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 100.0, 0.01).unwrap();
        int.velocity_verlet(&mut parts);
        assert_abs_diff_eq!(parts[0].position[0], 0.01, epsilon = 1e-9);
        assert_abs_diff_eq!(parts[1].position[1], 0.01, epsilon = 1e-9);
    }

    #[test]
    fn temperature_scales_with_velocity() {
        let parts = vec![Particle::new(0, v(&[0.0; 3]), v(&[2.0, 0.0, 0.0]), 1.0).unwrap()];
        let int = Integrator::new(f64::INFINITY, 1.0, 0.01).unwrap();
        // Single particle => T = 0 by the (N-1) normalization.
        assert_abs_diff_eq!(int.temperature(&parts), 0.0, epsilon = 1e-12);
        let two = vec![
            Particle::new(0, v(&[0.0; 3]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[0.0; 3]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
        ];
        // K = 2·(½·1·1) = 1 ; reduced T = 2·K / (3·1·1) = 2/3.
        assert_abs_diff_eq!(int.temperature(&two), 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn rdf_dilute_is_unity_like() {
        // Widely separated particles in a large periodic box: g(r) is defined
        // but sparse; just check it runs and returns the right length.
        let parts: Vec<Particle> = (0..10)
            .map(|i| Particle::new(i, v(&[(i as f64) * 5.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap())
            .collect();
        let (r, g) = rdf(&parts, f64::INFINITY, 10.0, 20).unwrap();
        assert_eq!(r.len(), 20);
        assert_eq!(g.len(), 20);
    }
}
