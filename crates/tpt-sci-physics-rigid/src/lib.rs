//! # tpt-sci-physics-rigid
//!
//! A small, dependency-light **rigid-body (sphere) physics world** with analytic
//! collision resolution, implemented entirely from scratch (no wrapped physics
//! engine such as `rapier`). Bodies are spheres of arbitrary dimension (2-D or
//! 3-D) described by [`Body`], simulated inside a [`World`] that supports
//! constant gravity, axis-aligned bounding walls, and pairwise elastic
//! collisions with a configurable restitution coefficient and an optional
//! global Coulomb friction coefficient ([`World::set_friction`]).
//!
//! The integrator is a semi-implicit (symplectic) Euler scheme; collisions are
//! resolved with the standard two-sphere normal impulse, a tangential impulse
//! bounded by Coulomb's law (`|friction_impulse| <= mu * |normal_impulse|`),
//! plus a positional (Baumgarte-style) correction to keep overlapping bodies
//! from sinking into one another. Candidate colliding pairs are found with a
//! sweep-and-prune broad-phase rather than an all-pairs `O(n^2)` scan.
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

mod body;
mod collision;
mod quaternion;
mod world;

pub use body::Body;
#[cfg(test)]
use collision::{aabb_overlap, normalize};
pub(crate) use collision::{resolve_pair, sap_candidate_pairs};
pub use quaternion::{quat_mul, quat_normalize, quat_to_matrix};
pub use world::World;

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
        assert_abs_diff_eq!(
            (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt(),
            1.0,
            epsilon = 1e-9
        );
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

    /// Build a world with a sphere dropping under gravity onto the floor
    /// (a bounding wall) while sliding sideways (nonzero tangential
    /// velocity), and return the tangential (x-axis) speed after `steps`
    /// small time steps.
    fn sliding_tangential_speed(mu: f64, steps: usize) -> f64 {
        let mut world = World::with_gravity(v(&[0.0, -9.8]));
        world.set_bounds(v(&[100.0, 5.0]));
        world.set_restitution(0.0); // inelastic, so the body stays on the floor
        world.set_friction(mu);
        world
            .add_body(Body::new(0, v(&[0.0, 4.6]), v(&[2.0, 0.0]), 1.0, 0.5).unwrap())
            .unwrap();
        let dt = 0.01;
        for _ in 0..steps {
            world.step(dt);
        }
        world.body(0).unwrap().velocity[0].abs()
    }

    #[test]
    fn wall_friction_decelerates_tangential_slide() {
        let speed_no_friction = sliding_tangential_speed(0.0, 200);
        let speed_with_friction = sliding_tangential_speed(0.8, 200);
        // Without friction the tangential (x) velocity is unaffected by the
        // (purely normal) wall bounce; with friction it must be strictly
        // smaller after the same number of steps sliding on the floor.
        assert_abs_diff_eq!(speed_no_friction, 2.0, epsilon = 1e-6);
        assert!(
            speed_with_friction < speed_no_friction,
            "friction should reduce tangential speed: {speed_with_friction} vs {speed_no_friction}"
        );
    }

    #[test]
    fn body_body_friction_decelerates_tangential_relative_velocity() {
        // Two overlapping spheres centered along the x-axis (so the contact
        // normal is exactly x), with `a` approaching `b` along x while also
        // sliding relative to `b` along y (tangential to the contact).
        // Resolve directly via `resolve_pair` (bypassing `World::step`'s
        // position integration, which would otherwise perturb the contact
        // normal) so the normal/tangential decomposition is exact.
        fn run(mu: f64) -> f64 {
            let mut a = Body::new(0, v(&[0.0, 0.0]), v(&[1.0, 1.0]), 1.0, 0.5).unwrap();
            let mut b = Body::new(1, v(&[0.9, 0.0]), v(&[0.0, 0.0]), 1.0, 0.5).unwrap();
            resolve_pair(&mut a, &mut b, 0.0, mu); // inelastic (restitution = 0)
            let rel = b.velocity - a.velocity;
            rel[1].abs() // relative tangential (y) speed after collision
        }

        let rel_tangential_no_friction = run(0.0);
        let rel_tangential_with_friction = run(0.9);
        assert_abs_diff_eq!(rel_tangential_no_friction, 1.0, epsilon = 1e-6);
        assert!(
            rel_tangential_with_friction < rel_tangential_no_friction,
            "friction should reduce relative tangential speed: {rel_tangential_with_friction} vs {rel_tangential_no_friction}"
        );
    }

    #[test]
    fn friction_impulse_never_exceeds_coulomb_bound() {
        // Directly probe `resolve_pair` at a variety of `mu` and check that
        // the total impulse magnitude applied to `b` decomposes into a normal
        // part `j_n` and tangential part `j_t` with `j_t <= mu * j_n` (up to
        // float slop).
        for &mu in &[0.0_f64, 0.1, 0.5, 1.0, 2.5] {
            let mut a = Body::new(0, v(&[0.0, 0.0]), v(&[1.0, 0.5]), 1.0, 0.5).unwrap();
            let mut b = Body::new(1, v(&[0.9, 0.0]), v(&[-0.5, -0.2]), 2.0, 0.5).unwrap();
            let v_before = b.velocity.clone();
            resolve_pair(&mut a, &mut b, 0.3, mu);
            let delta_v = b.velocity.clone() - v_before;
            let normal = normalize(&(v(&[0.9, 0.0]) - v(&[0.0, 0.0])));
            let j_n = delta_v.dot(&normal); // impulse/mass along the normal
            let tangential = delta_v.clone() - (normal.clone() * j_n);
            let j_t = tangential.norm();
            assert!(
                j_t <= mu * j_n.abs() + 1e-9,
                "mu={mu}: j_t={j_t} exceeds mu*j_n={}",
                mu * j_n.abs()
            );
        }
    }

    #[test]
    fn sap_matches_brute_force_pair_set() {
        // Deterministic pseudo-random body layout (linear congruential
        // generator, no external `rand` dependency) with several dozen
        // bodies of varying radii, some clustered (overlapping) and some far
        // apart, in 3-D.
        fn lcg_next(state: &mut u64) -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (u32::MAX as f64) // in [0, 1)
        }

        let mut state = 42_u64;
        let mut bodies = Vec::new();
        for id in 0..60 {
            let x = lcg_next(&mut state) * 20.0 - 10.0;
            let y = lcg_next(&mut state) * 20.0 - 10.0;
            let z = lcg_next(&mut state) * 20.0 - 10.0;
            let r = 0.1 + lcg_next(&mut state) * 0.9;
            bodies.push(Body::new(id, v(&[x, y, z]), v(&[0.0, 0.0, 0.0]), 1.0, r).unwrap());
        }

        let sap: std::collections::HashSet<(usize, usize)> =
            sap_candidate_pairs(&bodies).into_iter().collect();

        let mut brute = std::collections::HashSet::new();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                if aabb_overlap(&bodies[i], &bodies[j]) {
                    brute.insert((i, j));
                }
            }
        }

        assert_eq!(sap, brute);
        assert!(!brute.is_empty(), "test setup should produce some overlaps");
    }

    #[test]
    fn sap_excludes_clearly_separated_pair() {
        let bodies = vec![
            Body::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0, 0.0, 0.0]), 1.0, 0.5).unwrap(),
            Body::new(1, v(&[100.0, 100.0, 100.0]), v(&[0.0, 0.0, 0.0]), 1.0, 0.5).unwrap(),
        ];
        let pairs = sap_candidate_pairs(&bodies);
        assert!(
            pairs.is_empty(),
            "far-apart, non-overlapping AABBs should not be a candidate pair: {pairs:?}"
        );
        assert!(!aabb_overlap(&bodies[0], &bodies[1]));
    }
}
