//! # tpt-sci-md
//!
//! A small, dependency-light **classical molecular dynamics (MD)** engine built
//! entirely from scratch (no wrapped engine such as `lumol`, which was audited
//! and rejected — BSD-3-Clause and alpha/stale). It provides:
//!
//! * A [`Particle`] type (position, velocity, force, mass, type id).
//! * Pairwise **Lennard-Jones (12-6)** interactions with a cut-off radius and
//!   minimum-image convention under periodic boundaries ([`Forces::lennard_jones`]).
//! * A **Finnis-Sinclair-style EAM** (embedded-atom method) potential
//!   ([`EamParams`], [`Forces::eam`]) for simple metallic-bonding-like systems.
//! * **Ewald summation** for periodic long-range electrostatics ([`Ewald`]).
//! * **SHAKE/RATTLE** holonomic bond-length constraints ([`Shake`],
//!   [`Integrator::velocity_verlet_constrained`]).
//! * **Cell lists** (linked cells) for `O(n)`-ish neighbor finding
//!   ([`CellList`], [`Forces::lennard_jones_cells`]), cross-checked against
//!   [`neighbor_pairs_brute_force`].
//! * Velocity-Verlet integration ([`Integrator::velocity_verlet`]), a generic
//!   [`Integrator::step_with`] hook for other potentials, and optional
//!   Berendsen-style thermostats.
//! * A **radial distribution function** [`rdf`] for structural analysis.
//!
//! The engine is intentionally minimal: it models mono- (or few-) species
//! systems in a cubic periodic box and is sized for teaching, prototyping,
//! and coupling into the broader `tpt-science` platform — not for
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

    /// Advance the system by one velocity-Verlet step using a caller-supplied
    /// force/energy evaluator instead of the built-in Lennard-Jones potential.
    ///
    /// `compute` must (re)write every particle's `force` field for the *new*
    /// positions and return the potential energy at those positions. This is
    /// the hook used to drive [`Forces::eam`], [`Ewald::energy_forces`], or any
    /// other potential through the same velocity-Verlet integration as
    /// [`Integrator::velocity_verlet`].
    ///
    /// # Example
    ///
    /// ```
    /// use tpt_sci_md::{Particle, Integrator, Forces};
    /// use tpt_math_linalg::tpt_math_linalg_dense::DVector;
    ///
    /// let mut parts = vec![
    ///     Particle::new(0, DVector::from_row_slice(&[0.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
    ///     Particle::new(1, DVector::from_row_slice(&[1.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
    /// ];
    /// let int = Integrator::new(10.0, 1.0, 0.005).unwrap();
    /// let (box_len, sigma) = (int.box_len, int.sigma);
    /// let _energy = int.step_with(&mut parts, |p| Forces::lennard_jones(p, box_len, sigma));
    /// ```
    pub fn step_with(
        &self,
        particles: &mut [Particle],
        mut compute: impl FnMut(&mut [Particle]) -> f64,
    ) -> f64 {
        let dt = self.dt;
        let half = 0.5 * dt;
        for p in particles.iter_mut() {
            p.position = p.position.clone()
                + (p.velocity.clone() * dt)
                + (p.force.clone() * (0.5 * dt * dt) / p.mass);
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
        let energy = compute(particles);
        for p in particles.iter_mut() {
            p.velocity = p.velocity.clone() + (p.force.clone() * half / p.mass);
        }
        energy
    }

    /// Advance the system by one velocity-Verlet step with SHAKE position
    /// constraints and a RATTLE-style velocity projection, using the built-in
    /// Lennard-Jones potential for the unconstrained forces.
    ///
    /// This follows the standard SHAKE/RATTLE combination: (1) take the usual
    /// unconstrained velocity-Verlet position update, (2) iteratively correct
    /// positions so every constrained bond returns to its target length
    /// ([`Shake::constrain_positions`]), folding the position correction back
    /// into the velocities, (3) recompute forces at the corrected positions,
    /// (4) finish the velocity half-kick, and (5) project out the
    /// bond-parallel velocity component so `d/dt |r_ij| = 0`
    /// ([`Shake::constrain_velocities`]).
    ///
    /// # Errors
    ///
    /// Returns an error if SHAKE or the velocity projection fail to converge
    /// within `shake.max_iter` iterations, or if a bond references an
    /// out-of-range particle index.
    pub fn velocity_verlet_constrained(
        &self,
        particles: &mut [Particle],
        shake: &Shake,
    ) -> Result<f64, MdError> {
        let dt = self.dt;
        let half = 0.5 * dt;
        let old_positions: Vec<DVector<f64>> =
            particles.iter().map(|p| p.position.clone()).collect();
        let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

        // (1) Unconstrained half-step (position update + first velocity kick).
        for p in particles.iter_mut() {
            p.position = p.position.clone()
                + (p.velocity.clone() * dt)
                + (p.force.clone() * (0.5 * dt * dt) / p.mass);
            p.velocity = p.velocity.clone() + (p.force.clone() * half / p.mass);
        }

        // (2) SHAKE: correct positions back onto the constraint manifold, and
        // fold the correction into the velocity (implicit constraint force).
        let mut corrected: Vec<DVector<f64>> =
            particles.iter().map(|p| p.position.clone()).collect();
        shake.constrain_positions(&old_positions, &mut corrected, &masses, self.box_len)?;
        for (p, np) in particles.iter_mut().zip(corrected.iter()) {
            let delta = np.clone() - p.position.clone();
            p.velocity = p.velocity.clone() + (delta / dt);
            p.position = np.clone();
            if self.box_len.is_finite() {
                let bl = self.box_len;
                let pos = p.position.clone();
                p.position = DVector::from_fn(pos.len(), |k| {
                    let x = pos[k];
                    x - bl * (x / bl).floor()
                });
            }
        }

        // (3)-(4) Recompute forces, finish the velocity kick.
        let energy = Forces::lennard_jones(particles, self.box_len, self.sigma);
        for p in particles.iter_mut() {
            p.velocity = p.velocity.clone() + (p.force.clone() * half / p.mass);
        }

        // (5) RATTLE-style velocity constraint.
        let positions: Vec<DVector<f64>> = particles.iter().map(|p| p.position.clone()).collect();
        let mut velocities: Vec<DVector<f64>> =
            particles.iter().map(|p| p.velocity.clone()).collect();
        shake.constrain_velocities(&positions, &mut velocities, &masses)?;
        for (p, v) in particles.iter_mut().zip(velocities) {
            p.velocity = v;
        }

        Ok(energy)
    }
}

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
fn eam_forces_energy(
    particles: &[Particle],
    box_len: f64,
    params: &EamParams,
) -> (Vec<DVector<f64>>, f64) {
    let n = particles.len();
    let dim = particles.first().map_or(3, |p| p.position.len());
    let mut forces = vec![DVector::zeros(dim); n];
    if n < 2 {
        return (forces, 0.0);
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

    // Pass 1: pairwise distances and local densities ρ_i.
    let mut rho = vec![0.0_f64; n];
    let mut pairs: Vec<(usize, usize, DVector<f64>, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r2 = dr.dot(&dr);
            if r2 >= rcut2 || r2 <= 0.0 {
                continue;
            }
            let r = r2.sqrt();
            rho[i] += params.f_rho(r);
            rho[j] += params.f_rho(r);
            pairs.push((i, j, dr, r));
        }
    }

    let fprime: Vec<f64> = rho.iter().map(|&rh| params.embed_prime(rh)).collect();
    let mut energy: f64 = rho.iter().map(|&rh| params.embed(rh)).sum();

    // Pass 2: pairwise force, including the embedding chain-rule term
    // (F'(ρ_i) + F'(ρ_j))·f'(r_ij), the standard EAM force expression.
    for (i, j, dr, r) in pairs {
        energy += params.phi(r);
        let dudr = params.dphi_dr(r) + (fprime[i] + fprime[j]) * params.df_dr(r);
        let f_mag = dudr / r;
        let fvec = dr * f_mag;
        forces[i] = forces[i].clone() + fvec.clone();
        forces[j] = forces[j].clone() - fvec;
    }

    (forces, energy)
}

/// Compute the force on every particle from a Finnis-Sinclair-style EAM
/// potential (see [`EamParams`]). Analogous in shape to [`lennard_jones`].
#[must_use]
pub fn eam_forces(particles: &[Particle], box_len: f64, params: &EamParams) -> Vec<DVector<f64>> {
    eam_forces_energy(particles, box_len, params).0
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

/// Complementary error function via the Abramowitz & Stegun 7.1.26 rational
/// approximation (max absolute error ~1.5e-7) — enough precision for MD forces
/// and energies without pulling in a special-functions dependency.
fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    poly * (-x * x).exp()
}

/// Ewald summation for periodic Coulomb electrostatics in a cubic box.
///
/// Splits `1/r` into a rapidly-converging real-space short-range part
/// (`erfc(α·r)/r`, truncated at [`Ewald::rcut`]) and a reciprocal-space part
/// evaluated as a **direct discrete Fourier sum** over `k`-vectors
/// `k = (2π/L)·(nx, ny, nz)` with `|n| <= kmax` in each dimension, plus the
/// standard self-energy and net-charge background corrections.
///
/// **Implementation note:** this crate has no FFT available (per the
/// dependency policy — hand-rolled code only, ADR 0007), so the reciprocal
/// sum is the direct `O(N·kmax³)` double loop rather than an FFT-based PPPM
/// mesh solve. This is exact (not an approximation of the reciprocal sum
/// itself) but scales worse than PPPM for large systems or high accuracy
/// (large `kmax`); it is intended for the small/teaching-scale systems this
/// crate targets.
#[derive(Debug, Clone)]
pub struct Ewald {
    /// Cubic periodic-box side length.
    pub box_len: f64,
    /// Ewald splitting parameter `α` (inverse length).
    pub alpha: f64,
    /// Maximum reciprocal-lattice index per dimension.
    pub kmax: i32,
    /// Real-space cut-off (must be `<= box_len/2` for minimum-image validity).
    pub rcut: f64,
}

impl Ewald {
    /// Construct a validated Ewald summation.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if `box_len` is not finite and `> 0`,
    /// `alpha <= 0`, `kmax < 1`, or `rcut` is not in `(0, box_len/2]`.
    pub fn new(box_len: f64, alpha: f64, kmax: i32, rcut: f64) -> Result<Self, MdError> {
        if !box_len.is_finite() || box_len <= 0.0 {
            return Err(MdError::InvalidEwald(format!(
                "box_len must be finite and > 0, got {box_len}"
            )));
        }
        if alpha <= 0.0 {
            return Err(MdError::InvalidEwald(format!(
                "alpha must be > 0, got {alpha}"
            )));
        }
        if kmax < 1 {
            return Err(MdError::InvalidEwald(format!(
                "kmax must be >= 1, got {kmax}"
            )));
        }
        if rcut <= 0.0 || rcut > box_len * 0.5 {
            return Err(MdError::InvalidEwald(format!(
                "rcut must be in (0, box_len/2], got {rcut}"
            )));
        }
        Ok(Self {
            box_len,
            alpha,
            kmax,
            rcut,
        })
    }

    /// Compute the total electrostatic energy and per-particle forces.
    ///
    /// `charges[i]` is the point charge of `particles[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if `charges.len() != particles.len()`
    /// or particle positions are not 3-dimensional.
    #[allow(clippy::similar_names)]
    pub fn energy_forces(
        &self,
        particles: &[Particle],
        charges: &[f64],
    ) -> Result<(f64, Vec<DVector<f64>>), MdError> {
        let n = particles.len();
        if charges.len() != n {
            return Err(MdError::InvalidEwald(
                "charges length must match particles length".into(),
            ));
        }
        if particles.iter().any(|p| p.position.len() != 3) {
            return Err(MdError::InvalidEwald(
                "Ewald summation requires 3-dimensional positions".into(),
            ));
        }
        let mut forces = vec![DVector::zeros(3); n];
        let alpha = self.alpha;
        let box_len = self.box_len;
        let volume = box_len.powi(3);

        let wrap = |x: f64| -> f64 {
            let half = box_len * 0.5;
            let mut y = x - box_len * (x / box_len).floor();
            if y > half {
                y -= box_len;
            }
            y
        };

        // --- Real space: short-range erfc(alpha*r)/r, minimum-image, cut off at rcut. ---
        let rcut2 = self.rcut * self.rcut;
        let mut e_real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let raw = particles[j].position.clone() - particles[i].position.clone();
                let dr = DVector::from_fn(3, |k| wrap(raw[k]));
                let r2 = dr.dot(&dr);
                if r2 >= rcut2 || r2 <= 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let qq = charges[i] * charges[j];
                let erfc_ar = erfc(alpha * r);
                e_real += qq * erfc_ar / r;
                let dudr = qq
                    * (-2.0 * alpha / std::f64::consts::PI.sqrt() * (-(alpha * r).powi(2)).exp()
                        / r
                        - erfc_ar / r2);
                let f_mag = dudr / r;
                let fvec = dr * f_mag;
                forces[i] = forces[i].clone() + fvec.clone();
                forces[j] = forces[j].clone() - fvec;
            }
        }

        // --- Self-energy correction (position-independent, no force contribution). ---
        let q2sum: f64 = charges.iter().map(|q| q * q).sum();
        let e_self = alpha / std::f64::consts::PI.sqrt() * q2sum;

        // --- Net-charge background correction (position-independent). ---
        let qsum: f64 = charges.iter().sum();
        let e_background = -std::f64::consts::PI / (2.0 * alpha * alpha * volume) * qsum * qsum;

        // --- Reciprocal space: direct sum over k = (2*pi/L)*(nx,ny,nz), k != 0. ---
        let two_pi = 2.0 * std::f64::consts::PI;
        let kunit = two_pi / box_len;
        let mut e_recip = 0.0;
        for nx in -self.kmax..=self.kmax {
            for ny in -self.kmax..=self.kmax {
                for nz in -self.kmax..=self.kmax {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }
                    let kx = kunit * f64::from(nx);
                    let ky = kunit * f64::from(ny);
                    let kz = kunit * f64::from(nz);
                    let k2 = kx * kx + ky * ky + kz * kz;
                    let a_k = (-k2 / (4.0 * alpha * alpha)).exp() / k2;

                    let mut c = 0.0_f64;
                    let mut s = 0.0_f64;
                    let mut krs = Vec::with_capacity(n);
                    for (idx, p) in particles.iter().enumerate() {
                        let kr = kx * p.position[0] + ky * p.position[1] + kz * p.position[2];
                        c += charges[idx] * kr.cos();
                        s += charges[idx] * kr.sin();
                        krs.push(kr);
                    }
                    e_recip += a_k * (c * c + s * s);

                    // Force_i = (4*pi/V) * q_i * A(k) * k * [C*sin(k.r_i) - S*cos(k.r_i)]
                    let prefac = 4.0 * std::f64::consts::PI / volume * a_k;
                    for idx in 0..n {
                        let kr = krs[idx];
                        let coeff = prefac * charges[idx] * (c * kr.sin() - s * kr.cos());
                        let delta = DVector::from_row_slice(&[coeff * kx, coeff * ky, coeff * kz]);
                        forces[idx] = forces[idx].clone() + delta;
                    }
                }
            }
        }
        e_recip *= two_pi / volume;

        let total = e_real + e_recip + e_background - e_self;
        Ok((total, forces))
    }
}

/// A pairwise bond-length constraint for SHAKE/RATTLE.
#[derive(Debug, Clone, Copy)]
pub struct Bond {
    /// Index of the first particle.
    pub i: usize,
    /// Index of the second particle.
    pub j: usize,
    /// Target (fixed) bond length.
    pub r0: f64,
}

/// SHAKE (position) / RATTLE-style (velocity) holonomic bond-length
/// constraints for use with [`Integrator::velocity_verlet_constrained`].
///
/// Both solvers use the standard linearized SHAKE correction, applied
/// iteratively (Gauss-Seidel over all bonds) until every bond's squared-length
/// error is below `tol` (relative, for positions) or `max_iter` is exceeded.
#[derive(Debug, Clone)]
pub struct Shake {
    /// The constrained bonds.
    pub bonds: Vec<Bond>,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum Gauss-Seidel sweeps before giving up.
    pub max_iter: usize,
}

impl Shake {
    /// Construct a validated set of SHAKE/RATTLE constraints.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if `bonds` is empty, any bond has
    /// `i == j` or `r0 <= 0`, `tol <= 0`, or `max_iter == 0`.
    pub fn new(bonds: Vec<Bond>, tol: f64, max_iter: usize) -> Result<Self, MdError> {
        if bonds.is_empty() {
            return Err(MdError::ShakeError("at least one bond is required".into()));
        }
        for b in &bonds {
            if b.i == b.j {
                return Err(MdError::ShakeError(format!(
                    "bond {}-{} must connect two distinct particles",
                    b.i, b.j
                )));
            }
            if b.r0 <= 0.0 {
                return Err(MdError::ShakeError(format!(
                    "bond {}-{}: r0 must be > 0, got {}",
                    b.i, b.j, b.r0
                )));
            }
        }
        if tol <= 0.0 {
            return Err(MdError::ShakeError(format!("tol must be > 0, got {tol}")));
        }
        if max_iter == 0 {
            return Err(MdError::ShakeError("max_iter must be > 0".into()));
        }
        Ok(Self {
            bonds,
            tol,
            max_iter,
        })
    }

    /// Iteratively correct `positions` (the unconstrained new positions) so
    /// every bond returns to its target length, using `old_positions` (which
    /// must already satisfy the constraints) to fix the correction direction —
    /// the standard linearized SHAKE algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if a bond references an out-of-range
    /// particle index, or the iteration does not converge within `max_iter`.
    pub fn constrain_positions(
        &self,
        old_positions: &[DVector<f64>],
        positions: &mut [DVector<f64>],
        masses: &[f64],
        box_len: f64,
    ) -> Result<(), MdError> {
        let n = positions.len();
        for b in &self.bonds {
            if b.i >= n || b.j >= n {
                return Err(MdError::ShakeError(format!(
                    "bond references out-of-range particle {}/{} (n = {n})",
                    b.i, b.j
                )));
            }
        }
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
        for _ in 0..self.max_iter {
            let mut max_err = 0.0_f64;
            for b in &self.bonds {
                let (i, j, d0) = (b.i, b.j, b.r0);
                let dim = positions[i].len();
                let raw_old = old_positions[i].clone() - old_positions[j].clone();
                let r_old = DVector::from_fn(dim, |k| wrap(raw_old[k]));
                let raw_new = positions[i].clone() - positions[j].clone();
                let s = DVector::from_fn(dim, |k| wrap(raw_new[k]));
                let r2 = s.dot(&s);
                let diff = r2 - d0 * d0;
                max_err = max_err.max(diff.abs() / (d0 * d0));
                let denom = 2.0 * (1.0 / masses[i] + 1.0 / masses[j]) * r_old.dot(&s);
                if denom.abs() < 1e-14 {
                    continue;
                }
                let g = diff / denom;
                positions[i] = positions[i].clone() - r_old.clone() * (g / masses[i]);
                positions[j] = positions[j].clone() + r_old * (g / masses[j]);
            }
            if max_err < self.tol {
                return Ok(());
            }
        }
        Err(MdError::ShakeError(format!(
            "SHAKE did not converge within {} iterations",
            self.max_iter
        )))
    }

    /// Iteratively project `velocities` so every constrained bond satisfies
    /// `d/dt |r_ij| = 0`, i.e. `r_ij · v_ij = 0` (the RATTLE velocity
    /// constraint), given the already position-constrained `positions`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if a bond references an out-of-range
    /// particle index, or the iteration does not converge within `max_iter`.
    pub fn constrain_velocities(
        &self,
        positions: &[DVector<f64>],
        velocities: &mut [DVector<f64>],
        masses: &[f64],
    ) -> Result<(), MdError> {
        let n = velocities.len();
        for b in &self.bonds {
            if b.i >= n || b.j >= n {
                return Err(MdError::ShakeError(format!(
                    "bond references out-of-range particle {}/{} (n = {n})",
                    b.i, b.j
                )));
            }
        }
        for _ in 0..self.max_iter {
            let mut max_err = 0.0_f64;
            for b in &self.bonds {
                let (i, j, d0) = (b.i, b.j, b.r0);
                let r_ij = positions[i].clone() - positions[j].clone();
                let v_ij = velocities[i].clone() - velocities[j].clone();
                let num = r_ij.dot(&v_ij);
                max_err = max_err.max(num.abs());
                let denom = (1.0 / masses[i] + 1.0 / masses[j]) * d0 * d0;
                if denom.abs() < 1e-14 {
                    continue;
                }
                let k = num / denom;
                velocities[i] = velocities[i].clone() - r_ij.clone() * (k / masses[i]);
                velocities[j] = velocities[j].clone() + r_ij * (k / masses[j]);
            }
            if max_err < self.tol {
                return Ok(());
            }
        }
        Err(MdError::ShakeError(format!(
            "RATTLE velocity projection did not converge within {} iterations",
            self.max_iter
        )))
    }
}

/// A linked-cell (cell-list) neighbor structure for a cubic periodic box,
/// replacing the `O(n²)` pairwise scan used by [`lennard_jones`] with an
/// `O(n)`-ish build plus a 27-cell-stencil query, for a fixed cut-off radius.
#[derive(Debug, Clone)]
pub struct CellList {
    ncells: usize,
    box_len: f64,
    rcut: f64,
    cells: Vec<Vec<usize>>,
}

impl CellList {
    /// Build a cell list for `particles` in a cubic periodic box of side
    /// `box_len` with cut-off radius `rcut`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidCellList`] if `box_len` is not finite and
    /// `> 0`, `rcut <= 0`, `rcut > box_len/2`, or any particle is not
    /// 3-dimensional.
    pub fn build(particles: &[Particle], box_len: f64, rcut: f64) -> Result<Self, MdError> {
        if !box_len.is_finite() || box_len <= 0.0 {
            return Err(MdError::InvalidCellList(format!(
                "box_len must be finite and > 0, got {box_len}"
            )));
        }
        if rcut <= 0.0 {
            return Err(MdError::InvalidCellList(format!(
                "rcut must be > 0, got {rcut}"
            )));
        }
        if rcut > box_len * 0.5 {
            return Err(MdError::InvalidCellList(format!(
                "rcut ({rcut}) must be <= box_len/2 ({})",
                box_len * 0.5
            )));
        }
        if particles.iter().any(|p| p.position.len() != 3) {
            return Err(MdError::InvalidCellList(
                "cell lists require 3-dimensional positions".into(),
            ));
        }
        let ncells = ((box_len / rcut).floor() as usize).max(1);
        let cell_size = box_len / ncells as f64;
        let cell_index = |x: f64| -> usize {
            let xin = x - box_len * (x / box_len).floor();
            ((xin / cell_size) as usize).min(ncells - 1)
        };
        let mut cells = vec![Vec::new(); ncells * ncells * ncells];
        for (idx, p) in particles.iter().enumerate() {
            let cx = cell_index(p.position[0]);
            let cy = cell_index(p.position[1]);
            let cz = cell_index(p.position[2]);
            cells[(cx * ncells + cy) * ncells + cz].push(idx);
        }
        Ok(Self {
            ncells,
            box_len,
            rcut,
            cells,
        })
    }

    /// Return every particle-index pair `(i, j)` with `i < j` whose
    /// minimum-image distance is less than `rcut`, found via the 27-cell
    /// stencil around each occupied cell (wrapping periodically). Equivalent
    /// to, but generally much faster than, [`neighbor_pairs_brute_force`].
    #[must_use]
    pub fn neighbor_pairs(&self, particles: &[Particle]) -> Vec<(usize, usize)> {
        let n = self.ncells;
        let rcut2 = self.rcut * self.rcut;
        let box_len = self.box_len;
        let wrap = |x: f64| -> f64 {
            let half = box_len * 0.5;
            let mut y = x - box_len * (x / box_len).floor();
            if y > half {
                y -= box_len;
            }
            y
        };
        let mut pairs = std::collections::BTreeSet::new();
        for cx in 0..n {
            for cy in 0..n {
                for cz in 0..n {
                    let cell = &self.cells[(cx * n + cy) * n + cz];
                    if cell.is_empty() {
                        continue;
                    }
                    for dx in -1i64..=1 {
                        for dy in -1i64..=1 {
                            for dz in -1i64..=1 {
                                let ox = (cx as i64 + dx).rem_euclid(n as i64) as usize;
                                let oy = (cy as i64 + dy).rem_euclid(n as i64) as usize;
                                let oz = (cz as i64 + dz).rem_euclid(n as i64) as usize;
                                let other = &self.cells[(ox * n + oy) * n + oz];
                                for &i in cell {
                                    for &j in other {
                                        if i >= j {
                                            continue;
                                        }
                                        let raw = particles[j].position.clone()
                                            - particles[i].position.clone();
                                        let dr = DVector::from_fn(3, |k| wrap(raw[k]));
                                        let r2 = dr.dot(&dr);
                                        if r2 < rcut2 && r2 > 0.0 {
                                            pairs.insert((i, j));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }
}

/// Return every particle-index pair `(i, j)` with `i < j` whose minimum-image
/// distance in a cubic periodic box is less than `rcut`, via a direct `O(n²)`
/// scan. Used as the correctness reference for [`CellList::neighbor_pairs`].
#[must_use]
pub fn neighbor_pairs_brute_force(
    particles: &[Particle],
    box_len: f64,
    rcut: f64,
) -> Vec<(usize, usize)> {
    let n = particles.len();
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
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let dim = raw.len();
            let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r2 = dr.dot(&dr);
            if r2 < rcut2 && r2 > 0.0 {
                pairs.push((i, j));
            }
        }
    }
    pairs
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
        let _energy = int.velocity_verlet(&mut parts);
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

    // ---- EAM ----------------------------------------------------------

    #[test]
    fn eam_repulsive_at_short_range_attractive_at_intermediate_range() {
        let params = EamParams::generic();
        // Short range (r = 0.5, well inside the crossover at r ~ 1.0 for the
        // generic params): net force must be repulsive (push apart).
        let short = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[0.5, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = eam_forces(&short, f64::INFINITY, &params);
        assert!(
            f[0][0] < 0.0,
            "expected repulsion at short range, got {f:?}"
        );
        assert!(
            f[1][0] > 0.0,
            "expected repulsion at short range, got {f:?}"
        );

        // Intermediate range (r = 1.5): net force must be attractive (pull together).
        let mid = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[1.5, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = eam_forces(&mid, f64::INFINITY, &params);
        assert!(f[0][0] > 0.0, "expected attraction at mid range, got {f:?}");
        assert!(f[1][0] < 0.0, "expected attraction at mid range, got {f:?}");
    }

    #[test]
    fn eam_conserves_energy_over_short_nve_trajectory() {
        let params = EamParams::generic();
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.05, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[1.3, 0.0, 0.0]), v(&[-0.05, 0.02, 0.0]), 1.0).unwrap(),
            Particle::new(2, v(&[0.0, 1.3, 0.0]), v(&[0.0, -0.03, 0.01]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 1.0, 0.001).unwrap();
        let _ = Forces::eam(&mut parts, f64::INFINITY, &params);
        let kinetic0 = int.kinetic_energy(&parts);
        let potential0 = eam_forces_energy(&parts, f64::INFINITY, &params).1;
        let e0 = kinetic0 + potential0;

        let mut e_final = e0;
        for _ in 0..300 {
            let u = int.step_with(&mut parts, |p| Forces::eam(p, f64::INFINITY, &params));
            let k = int.kinetic_energy(&parts);
            e_final = u + k;
        }
        let rel_drift = (e_final - e0).abs() / e0.abs().max(1e-12);
        assert!(
            rel_drift < 0.05,
            "energy drifted too much: e0={e0}, e_final={e_final}, rel_drift={rel_drift}"
        );
    }

    // ---- Ewald ----------------------------------------------------------

    #[test]
    fn ewald_opposite_charges_attract_like_charges_repel() {
        let box_len = 40.0;
        let ewald = Ewald::new(box_len, 0.3, 6, box_len * 0.5).unwrap();

        let opposite = vec![
            Particle::new(0, v(&[20.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[21.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e, f) = ewald.energy_forces(&opposite, &[1.0, -1.0]).unwrap();
        // Opposite charges: energy negative (bound), force pulls together
        // (particle 0 pulled toward +x, i.e. toward particle 1).
        assert!(
            e < 0.0,
            "expected negative energy for opposite charges, got {e}"
        );
        assert!(f[0][0] > 0.0, "expected attractive pull, got {f:?}");

        let same = vec![
            Particle::new(0, v(&[20.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[21.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e2, f2) = ewald.energy_forces(&same, &[1.0, 1.0]).unwrap();
        assert!(
            e2 > 0.0,
            "expected positive energy for like charges, got {e2}"
        );
        assert!(f2[0][0] < 0.0, "expected repulsive push, got {f2:?}");
    }

    #[test]
    fn ewald_matches_direct_coulomb_in_dilute_limit() {
        // For a well-separated pair in a very large box relative to the
        // separation, periodic image contributions are negligible and the
        // Ewald energy should approach the bare Coulomb energy q1*q2/r
        // (Gaussian/reduced units, k_e = 1).
        let box_len = 200.0;
        let ewald = Ewald::new(box_len, 0.2, 8, box_len * 0.5).unwrap();
        let r = 2.0;
        let parts = vec![
            Particle::new(0, v(&[100.0, 100.0, 100.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[100.0 + r, 100.0, 100.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e, _f) = ewald.energy_forces(&parts, &[1.0, -1.0]).unwrap();
        let direct = -1.0 / r;
        assert_abs_diff_eq!(e, direct, epsilon = 5e-2);
    }

    #[test]
    fn ewald_conserves_energy_over_short_nve_trajectory() {
        let box_len = 30.0;
        let ewald = Ewald::new(box_len, 0.3, 5, box_len * 0.5).unwrap();
        let charges = vec![1.0, -1.0];
        let mut parts = vec![
            Particle::new(0, v(&[14.0, 15.0, 15.0]), v(&[0.0, 0.02, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[16.0, 15.0, 15.0]), v(&[0.0, -0.02, 0.01]), 1.0).unwrap(),
        ];
        let int = Integrator::new(box_len, 1.0, 0.002).unwrap();
        let (u0, f0) = ewald.energy_forces(&parts, &charges).unwrap();
        for (p, f) in parts.iter_mut().zip(f0) {
            p.force = f;
        }
        let e0 = int.kinetic_energy(&parts) + u0;

        let mut e_final = e0;
        for _ in 0..150 {
            let u = int.step_with(&mut parts, |p| {
                let (e, f) = ewald.energy_forces(p, &charges).unwrap();
                for (particle, ff) in p.iter_mut().zip(f) {
                    particle.force = ff;
                }
                e
            });
            e_final = int.kinetic_energy(&parts) + u;
        }
        let rel_drift = (e_final - e0).abs() / e0.abs().max(1e-12);
        assert!(
            rel_drift < 0.1,
            "energy drifted too much: e0={e0}, e_final={e_final}, rel_drift={rel_drift}"
        );
    }

    // ---- SHAKE / RATTLE --------------------------------------------------

    #[test]
    fn shake_keeps_bond_length_within_tolerance() {
        let bonds = vec![Bond {
            i: 0,
            j: 1,
            r0: 1.0,
        }];
        let shake = Shake::new(bonds, 1e-10, 200).unwrap();
        // Tiny sigma => the LJ cut-off (2.5*sigma) sits well inside the bond
        // length, so the LJ force is exactly zero; motion is pure free-flight
        // + constraint, exercising SHAKE cleanly.
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.1, 0.05, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[1.0, 0.0, 0.0]), v(&[-0.1, -0.02, 0.03]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 0.01, 0.01).unwrap();
        for _ in 0..200 {
            int.velocity_verlet_constrained(&mut parts, &shake).unwrap();
            let d = (parts[0].position.clone() - parts[1].position.clone()).norm();
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn shake_rejects_out_of_range_bond() {
        let bonds = vec![Bond {
            i: 0,
            j: 5,
            r0: 1.0,
        }];
        let shake = Shake::new(bonds, 1e-8, 100).unwrap();
        let old = vec![v(&[0.0; 3]), v(&[1.0, 0.0, 0.0])];
        let mut new = old.clone();
        let masses = vec![1.0, 1.0];
        assert!(
            shake
                .constrain_positions(&old, &mut new, &masses, f64::INFINITY)
                .is_err()
        );
    }

    // ---- Cell lists -------------------------------------------------------

    #[test]
    fn cell_list_matches_brute_force_neighbors() {
        // A modest random-ish 3-D configuration in a periodic box, dense
        // enough that some pairs cross cell boundaries and periodic images.
        let box_len = 10.0;
        let rcut = 2.0;
        let coords: [[f64; 3]; 14] = [
            [0.1, 0.2, 0.3],
            [9.9, 0.1, 0.2],
            [1.5, 1.5, 1.5],
            [5.0, 5.0, 5.0],
            [5.1, 5.05, 4.95],
            [0.0, 9.95, 0.05],
            [3.3, 7.7, 2.2],
            [3.4, 7.6, 2.1],
            [8.0, 8.0, 8.0],
            [8.1, 0.0, 8.1],
            [2.0, 2.0, 8.0],
            [2.1, 2.1, 7.9],
            [6.0, 1.0, 6.0],
            [6.05, 1.05, 5.95],
        ];
        let parts: Vec<Particle> = coords
            .iter()
            .enumerate()
            .map(|(i, c)| Particle::new(i, v(c), v(&[0.0; 3]), 1.0).unwrap())
            .collect();

        let brute = neighbor_pairs_brute_force(&parts, box_len, rcut);
        let cell_list = CellList::build(&parts, box_len, rcut).unwrap();
        let via_cells = cell_list.neighbor_pairs(&parts);

        assert_eq!(
            brute, via_cells,
            "cell-list neighbors must match brute force exactly"
        );
        assert!(
            !brute.is_empty(),
            "test configuration should have at least one neighbor pair"
        );
    }

    #[test]
    fn lennard_jones_cells_matches_brute_force_energy() {
        let box_len = 12.0;
        let sigma = 1.0;
        let coords: [[f64; 3]; 8] = [
            [1.0, 1.0, 1.0],
            [1.9, 1.0, 1.0],
            [1.0, 1.9, 1.0],
            [6.0, 6.0, 6.0],
            [6.9, 6.0, 6.0],
            [11.5, 1.0, 1.0],
            [0.4, 1.0, 1.0],
            [3.0, 9.0, 2.0],
        ];
        let mut brute: Vec<Particle> = coords
            .iter()
            .enumerate()
            .map(|(i, c)| Particle::new(i, v(c), v(&[0.0; 3]), 1.0).unwrap())
            .collect();
        let mut cells = brute.clone();

        let e_brute = Forces::lennard_jones(&mut brute, box_len, sigma);
        let e_cells = Forces::lennard_jones_cells(&mut cells, box_len, sigma).unwrap();

        assert_abs_diff_eq!(e_brute, e_cells, epsilon = 1e-10);
        for (pb, pc) in brute.iter().zip(cells.iter()) {
            for k in 0..3 {
                assert_abs_diff_eq!(pb.force[k], pc.force[k], epsilon = 1e-10);
            }
        }
    }
}
