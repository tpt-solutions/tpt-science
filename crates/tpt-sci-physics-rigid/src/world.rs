//! The rigid-body world: gravity, bounds, restitution and friction, and the time step.

use super::*;

/// A small rigid-body (sphere) simulation world.
///
/// All bodies share the same dimensionality, which is established by the first
/// body that is added. Gravity (if any), walls, and every body must agree on
/// that dimension.
#[derive(Debug, Clone)]
pub struct World {
    bodies: Vec<Body>,
    gravity: Option<DVector<f64>>,
    restitution: f64,
    /// Half-extents of an axis-aligned box centered at the origin. `None`
    /// means unbounded.
    bounds: Option<DVector<f64>>,
    /// Global Coulomb friction coefficient `mu >= 0`, applied to both
    /// body-body collisions and wall bounces.
    friction: f64,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Create an empty world with no gravity, perfectly elastic collisions
    /// (`restitution = 1.0`), and no bounding walls.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: None,
            restitution: 1.0,
            bounds: None,
            friction: 0.0,
        }
    }

    /// Create an empty world with the given constant gravitational acceleration
    /// vector. Its length fixes the world dimensionality.
    #[must_use]
    pub fn with_gravity(g: DVector<f64>) -> Self {
        Self {
            bodies: Vec::new(),
            gravity: Some(g),
            restitution: 1.0,
            bounds: None,
            friction: 0.0,
        }
    }

    /// Set the (dimensionless) coefficient of restitution used for all
    /// collisions and wall bounces. `1.0` is perfectly elastic.
    pub fn set_restitution(&mut self, restitution: f64) {
        self.restitution = restitution;
    }

    /// Set the global Coulomb friction coefficient `mu` (`mu >= 0`) used for
    /// both body-body collisions and wall bounces. `0.0` (the default)
    /// disables friction; the tangential impulse applied at a contact is
    /// clamped so `|friction_impulse| <= mu * |normal_impulse|`.
    pub fn set_friction(&mut self, mu: f64) {
        self.friction = mu.max(0.0);
    }

    /// Restrict the world to an axis-aligned box centered at the origin with the
    /// given half-extents; bodies bounce off the walls with the current
    /// restitution. The vector length fixes the world dimensionality.
    pub fn set_bounds(&mut self, half_extents: DVector<f64>) {
        self.bounds = Some(half_extents);
    }

    /// Add a body to the world.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::DuplicateId`] if a body with the same `id`
    /// already exists, or [`PhysicsError::DimensionMismatch`] if the body's
    /// dimension disagrees with the world's established dimension (or with an
    /// already-set gravity/bounds vector).
    pub fn add_body(&mut self, body: Body) -> Result<(), PhysicsError> {
        if self.bodies.iter().any(|b| b.id == body.id) {
            return Err(PhysicsError::DuplicateId(body.id));
        }
        let dim = body.position.len();
        if let Some(first) = self.bodies.first() {
            let expected = first.position.len();
            if expected != dim {
                return Err(PhysicsError::DimensionMismatch { expected, got: dim });
            }
        }
        if let Some(g) = &self.gravity {
            if g.len() != dim {
                return Err(PhysicsError::DimensionMismatch {
                    expected: g.len(),
                    got: dim,
                });
            }
        }
        if let Some(b) = &self.bounds {
            if b.len() != dim {
                return Err(PhysicsError::DimensionMismatch {
                    expected: b.len(),
                    got: dim,
                });
            }
        }
        self.bodies.push(body);
        Ok(())
    }

    /// Borrow the body with the given `id`, if present.
    #[must_use]
    pub fn body(&self, id: usize) -> Option<&Body> {
        self.bodies.iter().find(|b| b.id == id)
    }

    /// Mutably borrow the body with the given `id`, if present.
    pub fn body_mut(&mut self, id: usize) -> Option<&mut Body> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    /// Immutable access to all bodies.
    #[must_use]
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// Advance the simulation by `dt` seconds.
    ///
    /// The update order is: (1) semi-implicit Euler integration, (2) wall
    /// collisions, (3) pairwise body-body collision resolution.
    pub fn step(&mut self, dt: f64) {
        // (1) Semi-implicit (symplectic) Euler integration.
        for b in &mut self.bodies {
            if let Some(g) = &self.gravity {
                // Advance velocity first (semi-implicit), then position. The
                // position update uses the pre-update velocity plus a half-step
                // acceleration term, which is exact for constant acceleration
                // (e.g. free-fall: Δ = ½·g·dt²) and keeps the scheme symplectic.
                let new_velocity = b.velocity.clone() + (g.clone() * dt);
                b.position =
                    b.position.clone() + (b.velocity.clone() * dt) + (g.clone() * (0.5 * dt * dt));
                b.velocity = new_velocity;
            } else {
                b.position = b.position.clone() + (b.velocity.clone() * dt);
            }
        }

        // (2) Wall collisions. `DVector` has no `IndexMut`, so rebuild the
        // position/velocity vectors from plain `Vec<f64>` buffers (reflecting
        // the normal component and clamping the center inside the
        // half-extent). Friction, if enabled, damps the tangential (all
        // non-colliding axes) velocity component, bounded by
        // `mu * |normal_impulse|` per Coulomb's law.
        if let Some(bounds) = &self.bounds {
            let rest = self.restitution;
            let mu = self.friction;
            for b in &mut self.bodies {
                let dim = b.position.len();
                let r = b.radius;
                let mut pos: Vec<f64> = (0..dim).map(|axis| b.position[axis]).collect();
                let mut vel: Vec<f64> = (0..dim).map(|axis| b.velocity[axis]).collect();
                for axis in 0..dim {
                    if axis >= bounds.len() {
                        continue;
                    }
                    let half = bounds[axis];
                    let p = pos[axis];
                    let collided = p - r < -half || p + r > half;
                    if !collided {
                        continue;
                    }
                    pos[axis] = if p - r < -half { -half + r } else { half - r };

                    // Normal impulse (per unit mass) delivered by the wall.
                    let vn = vel[axis];
                    let new_vn = -rest * vn;
                    let j_n = (vn - new_vn).abs();
                    vel[axis] = new_vn;

                    if mu > 0.0 && j_n > 0.0 {
                        let tangent_speed_sq: f64 = vel
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| k != axis)
                            .map(|(_, v)| v * v)
                            .sum();
                        if tangent_speed_sq > 1e-18 {
                            let tangent_speed = tangent_speed_sq.sqrt();
                            // Coulomb bound: friction impulse (per unit mass)
                            // cannot exceed `mu * j_n`, and cannot reverse the
                            // tangential motion.
                            let j_t = (mu * j_n).min(tangent_speed);
                            let scale = j_t / tangent_speed;
                            for (k, v) in vel.iter_mut().enumerate() {
                                if k != axis {
                                    *v -= *v * scale;
                                }
                            }
                        }
                    }
                }
                b.position = DVector::from_row_slice(&pos);
                b.velocity = DVector::from_row_slice(&vel);
            }
        }

        // (3) Pairwise body-body collisions. A sweep-and-prune broad-phase
        // ([`sap_candidate_pairs`]) narrows the candidate set before the
        // exact narrow-phase resolution in [`resolve_pair`].
        let candidate_pairs = sap_candidate_pairs(&self.bodies);
        let restitution = self.restitution;
        let friction = self.friction;
        for (i, j) in candidate_pairs {
            let (a, b) = self.bodies.split_at_mut(j);
            resolve_pair(&mut a[i], &mut b[0], restitution, friction);
        }
    }
}
