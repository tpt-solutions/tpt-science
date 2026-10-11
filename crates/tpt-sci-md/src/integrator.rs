//! Time integration: velocity-Verlet, its constrained variant, and a generic step hook.

use super::*;

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
