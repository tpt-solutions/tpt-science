//! # tpt-sci-physics-rigid
//!
//! A small, dependency-light **rigid-body (sphere) physics world** with analytic
//! collision resolution, implemented entirely from scratch (no wrapped physics
//! engine such as `rapier`). Bodies are spheres of arbitrary dimension (2-D or
//! 3-D) described by [`Body`], simulated inside a [`World`] that supports
//! constant gravity, axis-aligned bounding walls, and pairwise elastic
//! collisions with a configurable restitution coefficient.
//!
//! The integrator is a semi-implicit (symplectic) Euler scheme; collisions are
//! resolved with the standard two-sphere impulse plus a positional
//! (Baumgarte-style) correction to keep overlapping bodies from sinking into
//! one another.
//!
//! # Examples
//!
//! ```
//! use tpt_sci_physics_rigid::{Body, World, PhysicsError};
//! use tpt_math_linalg::tpt_math_linalg_dense::DVector;
//!
//! let mut world = World::new();
//! let a = Body::new(
//!     0,
//!     DVector::from_row_slice(&[0.0_f64, 0.0]),
//!     DVector::from_row_slice(&[1.0_f64, 0.0]),
//!     1.0,
//!     0.5,
//! )?;
//! world.add_body(a).unwrap();
//! let b = Body::new(
//!     1,
//!     DVector::from_row_slice(&[1.0_f64, 0.0]),
//!     DVector::from_row_slice(&[0.0_f64, 0.0]),
//!     1.0,
//!     0.5,
//! )?;
//! world.add_body(b).unwrap();
//!
//! let p0 = world.body(0).unwrap().position.clone();
//! world.step(0.5);
//! let p1 = world.body(0).unwrap().position.clone();
//! assert!((p1[0] - p0[0]).abs() > 1e-9);
//! # Ok::<(), PhysicsError>(())
//! ```
#![forbid(unsafe_code)]

use tpt_math_linalg::tpt_math_linalg_dense::DVector;

mod error;

pub use error::PhysicsError;

/// A spherical rigid body living in an `n`-dimensional space (the dimension is
/// implied by the length of its [`Body::position`]/`[Body::velocity]` vectors).
#[derive(Debug, Clone)]
pub struct Body {
    /// Center of mass position.
    pub position: DVector<f64>,
    /// Linear velocity.
    pub velocity: DVector<f64>,
    /// Positive mass.
    pub mass: f64,
    /// Positive collision radius.
    pub radius: f64,
    /// Unique identifier within a [`World`].
    pub id: usize,
}

impl Body {
    /// Construct a validated body.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::InvalidBody`] if `mass <= 0`, `radius <= 0`,
    /// `position` and `velocity` have different lengths, or any component is
    /// non-finite.
    pub fn new(
        id: usize,
        position: DVector<f64>,
        velocity: DVector<f64>,
        mass: f64,
        radius: f64,
    ) -> Result<Self, PhysicsError> {
        if position.len() != velocity.len() {
            return Err(PhysicsError::InvalidBody(format!(
                "body {id}: position/velocity dimension mismatch ({} vs {})",
                position.len(),
                velocity.len()
            )));
        }
        if mass <= 0.0 {
            return Err(PhysicsError::InvalidBody(format!(
                "body {id}: mass must be > 0, got {mass}"
            )));
        }
        if radius <= 0.0 {
            return Err(PhysicsError::InvalidBody(format!(
                "body {id}: radius must be > 0, got {radius}"
            )));
        }
        for (name, v) in [("position", &position), ("velocity", &velocity)] {
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(PhysicsError::InvalidBody(format!(
                        "body {id}: non-finite {name} component at index {i}"
                    )));
                }
            }
        }
        Ok(Self {
            position,
            velocity,
            mass,
            radius,
            id,
        })
    }
}

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
        }
    }

    /// Set the (dimensionless) coefficient of restitution used for all
    /// collisions and wall bounces. `1.0` is perfectly elastic.
    pub fn set_restitution(&mut self, restitution: f64) {
        self.restitution = restitution;
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
        // position/velocity vectors via `from_fn` (reflecting the normal
        // component and clamping the center inside the half-extent).
        if let Some(bounds) = &self.bounds {
            let rest = self.restitution;
            for b in &mut self.bodies {
                let dim = b.position.len();
                let r = b.radius;
                let new_position = DVector::from_fn(dim, |axis| {
                    if axis >= bounds.len() {
                        return b.position[axis];
                    }
                    let half = bounds[axis];
                    let mut p = b.position[axis];
                    if p - r < -half {
                        p = -half + r;
                    } else if p + r > half {
                        p = half - r;
                    }
                    p
                });
                let new_velocity = DVector::from_fn(dim, |axis| {
                    if axis >= bounds.len() {
                        return b.velocity[axis];
                    }
                    let half = bounds[axis];
                    let p = b.position[axis];
                    let mut vel = b.velocity[axis];
                    if p - r < -half || p + r > half {
                        vel *= -rest;
                    }
                    vel
                });
                b.position = new_position;
                b.velocity = new_velocity;
            }
        }

        // (3) Pairwise body-body collisions.
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = self.bodies.split_at_mut(j);
                resolve_pair(&mut a[i], &mut b[0], self.restitution);
            }
        }
    }
}

/// Unit vector in the direction of `v` (assumes `v` is non-zero).
fn normalize(v: &DVector<f64>) -> DVector<f64> {
    let n = v.norm();
    v.clone() * (1.0 / n)
}

/// Resolve a single colliding pair of spheres: exchange the normal impulse when
/// the bodies are approaching, and push them apart to remove overlap.
fn resolve_pair(a: &mut Body, b: &mut Body, restitution: f64) {
    let delta = b.position.clone() - a.position.clone();
    let dist = delta.norm();
    let sum_r = a.radius + b.radius;
    if dist >= sum_r || !dist.is_finite() || dist <= 0.0 {
        return;
    }

    let normal = normalize(&delta); // points from `a` toward `b`
    let inv_mass_a = 1.0 / a.mass;
    let inv_mass_b = 1.0 / b.mass;

    // Relative velocity of `b` with respect to `a`, projected on the normal.
    let rel_vel = b.velocity.clone() - a.velocity.clone();
    let vrel = rel_vel.dot(&normal);

    // Negative `vrel` means the bodies are approaching along the normal.
    if vrel < 0.0 {
        let j_imp = -(1.0 + restitution) * vrel / (inv_mass_a + inv_mass_b);
        let impulse = normal.clone() * j_imp;
        a.velocity = a.velocity.clone() - (impulse.clone() * inv_mass_a);
        b.velocity = b.velocity.clone() + (impulse * inv_mass_b);
    }

    // Positional correction: split the overlap by inverse mass so that the
    // heavier body moves less, eliminating sink-through.
    let overlap = sum_r - dist;
    let correction = normal * (overlap / (inv_mass_a + inv_mass_b));
    a.position = a.position.clone() - (correction.clone() * inv_mass_a);
    b.position = b.position.clone() + (correction * inv_mass_b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_linalg::tpt_math_linalg_dense::DVector;

    fn v(x: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(x)
    }

    #[test]
    fn body_new_rejects_invalid() {
        assert!(Body::new(0, v(&[0.0]), v(&[1.0]), 0.0, 0.5).is_err());
        assert!(Body::new(0, v(&[0.0]), v(&[1.0]), 1.0, 0.0).is_err());
        assert!(Body::new(0, v(&[0.0]), v(&[1.0, 0.0]), 1.0, 0.5).is_err());
        assert!(Body::new(0, v(&[f64::NAN]), v(&[0.0]), 1.0, 0.5).is_err());
        assert!(Body::new(0, v(&[0.0]), v(&[1.0]), 1.0, 0.5).is_ok());
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut world = World::new();
        world
            .add_body(Body::new(0, v(&[0.0]), v(&[0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        let err = world
            .add_body(Body::new(0, v(&[1.0]), v(&[0.0]), 1.0, 0.5).unwrap())
            .unwrap_err();
        assert!(matches!(err, PhysicsError::DuplicateId(0)));
    }

    #[test]
    fn elastic_head_on_swaps_velocities() {
        let mut world = World::new();
        // Centers exactly r_i + r_j apart => touching; they overlap after dt.
        world
            .add_body(Body::new(0, v(&[0.0, 0.0]), v(&[1.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        world
            .add_body(Body::new(1, v(&[1.0, 0.0]), v(&[0.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();

        let p_before: f64 = world.bodies().iter().map(|b| b.mass * b.velocity[0]).sum();
        let ke_before: f64 = world
            .bodies()
            .iter()
            .map(|b| 0.5 * b.mass * b.velocity.dot(&b.velocity))
            .sum();

        world.step(0.5);

        let a = world.body(0).unwrap();
        let b = world.body(1).unwrap();
        assert_abs_diff_eq!(a.velocity[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(b.velocity[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(a.velocity[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(b.velocity[1], 0.0, epsilon = 1e-6);

        let p_after: f64 = world.bodies().iter().map(|b| b.mass * b.velocity[0]).sum();
        let ke_after: f64 = world
            .bodies()
            .iter()
            .map(|b| 0.5 * b.mass * b.velocity.dot(&b.velocity))
            .sum();
        assert_abs_diff_eq!(p_after, p_before, epsilon = 1e-6);
        assert_abs_diff_eq!(ke_after, ke_before, epsilon = 1e-6);
        assert_abs_diff_eq!(p_before, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ke_before, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn gravity_free_fall() {
        let mut world = World::with_gravity(v(&[0.0, -9.8]));
        world
            .add_body(Body::new(0, v(&[0.0, 10.0]), v(&[0.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        world.step(1.0);
        let b = world.body(0).unwrap();
        // Δy = ½·g·t² = -4.9
        assert_abs_diff_eq!(b.position[1], 10.0 - 4.9, epsilon = 1e-6);
        assert_abs_diff_eq!(b.velocity[1], -9.8, epsilon = 1e-6);
    }

    #[test]
    fn wall_bounce() {
        let mut world = World::new();
        world.set_bounds(v(&[5.0, 5.0]));
        world
            .add_body(Body::new(0, v(&[4.8, 0.0]), v(&[1.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        world.step(0.5);
        let b = world.body(0).unwrap();
        assert!(b.velocity[0] < 0.0, "velocity should be reflected");
        // Sphere must be fully inside the box: |p ± r| within half-extents.
        assert!(b.position[0] - b.radius >= -5.0 - 1e-9);
        assert!(b.position[0] + b.radius <= 5.0 + 1e-9);
        assert!(b.position[0].abs() <= 5.0 + 1e-9);
    }

    #[test]
    fn momentum_conserved_multi_body_elastic() {
        let mut world = World::new();
        world
            .add_body(Body::new(0, v(&[0.0, 0.0]), v(&[1.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        world
            .add_body(Body::new(1, v(&[1.0, 0.0]), v(&[0.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();

        let before: DVector<f64> = world.bodies().iter().fold(DVector::zeros(2), |acc, b| {
            acc + (b.velocity.clone() * b.mass)
        });

        world.step(0.5);

        let after: DVector<f64> = world.bodies().iter().fold(DVector::zeros(2), |acc, b| {
            acc + (b.velocity.clone() * b.mass)
        });

        assert_abs_diff_eq!(after[0], before[0], epsilon = 1e-6);
        assert_abs_diff_eq!(after[1], before[1], epsilon = 1e-6);
    }
}
