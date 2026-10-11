//! Particles: position, velocity, force, mass, and type id.

use super::*;

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
