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
//! Each [`Body`] is a genuine rigid body, not just a point mass: it carries an
//! orientation quaternion and an angular velocity, so torques change its spin
//! and its orientation integrates forward under [`Body::spin`]. For a sphere the
//! moment of inertia is isotropic (default `(2/5)·m·r²`), captured by a single
//! scalar [`Body::inertia`].
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

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

mod error;

pub use error::PhysicsError;

/// A spherical rigid body living in an `n`-dimensional space (the dimension is
/// implied by the length of its [`Body::position`]/`[Body::velocity]` vectors).
///
/// Beyond the point-mass kinematics, a body also carries an orientation
/// (unit quaternion `[w, x, y, z]`) and an angular velocity, so it behaves as
/// a true rigid body: torques change its spin and its orientation integrates
/// forward in time. For a sphere the moment of inertia is isotropic, so a
/// single scalar `inertia` (defaulting to `(2/5)·m·r²`) is sufficient.
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
    /// Orientation as a unit quaternion `[w, x, y, z]`.
    pub orientation: [f64; 4],
    /// Angular velocity (rad/s) as a 3-vector.
    pub angular_velocity: [f64; 3],
    /// Scalar moment of inertia about any principal axis (sphere: isotropic).
    pub inertia: f64,
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
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            inertia: 0.4 * mass * radius * radius,
        })
    }
}

impl Body {
    /// Set the 3×3 rotation matrix `R` corresponding to this body's
    /// orientation quaternion. Useful for orienting contact normals or
    /// rendering.
    #[must_use]
    pub fn orientation_matrix(&self) -> DMatrix<f64> {
        quat_to_matrix(&self.orientation)
    }

    /// Overwrite the orientation quaternion (must be a unit quaternion).
    pub fn set_orientation(&mut self, q: [f64; 4]) {
        self.orientation = q;
    }

    /// Overwrite the angular velocity (rad/s).
    pub fn set_angular_velocity(&mut self, omega: [f64; 3]) {
        self.angular_velocity = omega;
    }

    /// Set the scalar moment of inertia (overrides the default `(2/5)·m·r²`).
    pub fn set_inertia(&mut self, inertia: f64) {
        self.inertia = inertia;
    }

    /// Apply a torque vector `tau` (N·m) for `dt` seconds, changing the angular
    /// velocity via `ω += τ·dt / I`. For an isotropic (spherical) body this
    /// fully captures the rigid-body rotational dynamics.
    pub fn apply_torque(&mut self, tau: [f64; 3], dt: f64) {
        if self.inertia == 0.0 {
            return;
        }
        for (k, av) in self.angular_velocity.iter_mut().enumerate() {
            *av += tau[k] * dt / self.inertia;
        }
    }

    /// Integrate the orientation forward by `dt` seconds under the current
    /// angular velocity (quaternion kinematics `q̇ = ½·(0, ω)⊗q`), then
    /// renormalise so the quaternion stays unit.
    pub fn spin(&mut self, dt: f64) {
        let [w, x, y, z] = self.orientation;
        let [ox, oy, oz] = self.angular_velocity;
        // (0, ω) ⊗ q  (Hamilton product).
        let dq = [
            -0.5 * (ox * x + oy * y + oz * z),
            0.5 * (ox * w + oy * z - oz * y),
            0.5 * (-ox * z + oy * w + oz * x),
            0.5 * (ox * y - oy * x + oz * w),
        ];
        let nx = w + dq[0] * dt;
        let ny = x + dq[1] * dt;
        let nz = y + dq[2] * dt;
        let nw = z + dq[3] * dt;
        self.orientation = quat_normalize(nx, ny, nz, nw);
    }
}

/// Multiply two quaternions `a ⊗ b` (Hamilton product), both `[w, x, y, z]`.
#[must_use]
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Normalise a quaternion to unit length, returning `[w, x, y, z]`.
#[must_use]
pub fn quat_normalize(w: f64, x: f64, y: f64, z: f64) -> [f64; 4] {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n == 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [w / n, x / n, y / n, z / n]
    }
}

/// Convert a unit quaternion `[w, x, y, z]` to a 3×3 rotation matrix.
#[must_use]
pub fn quat_to_matrix(q: &[f64; 4]) -> DMatrix<f64> {
    let [w, x, y, z] = *q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    DMatrix::from_fn(3, 3, |r, c| match (r, c) {
        (0, 0) => 1.0 - (yy + zz),
        (0, 1) => xy - wz,
        (0, 2) => xz + wy,
        (1, 0) => xy + wz,
        (1, 1) => 1.0 - (xx + zz),
        (1, 2) => yz - wx,
        (2, 0) => xz - wy,
        (2, 1) => yz + wx,
        (2, 2) => 1.0 - (xx + yy),
        _ => 0.0,
    })
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

    #[test]
    fn apply_torque_changes_spin() {
        let mut b = Body::new(0, v(&[0.0, 0.0]), v(&[0.0, 0.0]), 2.0, 0.5).unwrap();
        // Default inertia for a sphere = 0.4·m·r² = 0.4·2·0.25 = 0.2.
        assert_abs_diff_eq!(b.inertia, 0.2, epsilon = 1e-12);
        // τ = (0,0,1) for dt = 0.1  =>  ω_z = τ·dt / I = 0.1/0.2 = 0.5.
        b.apply_torque([0.0, 0.0, 1.0], 0.1);
        assert_abs_diff_eq!(b.angular_velocity[2], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(b.angular_velocity[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn spin_integrates_orientation() {
        let mut b = Body::new(0, v(&[0.0, 0.0]), v(&[0.0, 0.0]), 1.0, 0.5).unwrap();
        b.set_angular_velocity([0.0, 0.0, 1.0]); // spin about z
        let dt = 0.01_f64;
        let theta = 1.0 * dt; // small angle
        b.spin(dt);
        let r = b.orientation_matrix();
        // Rotating the x-axis by `theta` about z should give (cosθ, sinθ, 0).
        assert_abs_diff_eq!(r[(0, 0)], theta.cos(), epsilon = 1e-3);
        assert_abs_diff_eq!(r[(1, 0)], theta.sin(), epsilon = 1e-3);
        // The quaternion must stay (approximately) unit.
        let q = b.orientation;
        assert_abs_diff_eq!((q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn quat_helpers_roundtrip() {
        let q = [0.0, 0.0, 0.0, 1.0]; // 180° about z
        let r = quat_to_matrix(&q);
        // Maps (1,0,0) -> (-1,0,0) and (0,1,0) -> (0,-1,0).
        assert_abs_diff_eq!(r[(0, 0)], -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r[(1, 1)], -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r[(2, 2)], 1.0, epsilon = 1e-9);
    }
}
