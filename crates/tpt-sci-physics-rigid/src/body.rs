//! Rigid bodies: position, velocity, orientation, angular velocity, mass and inertia.

use super::*;

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
