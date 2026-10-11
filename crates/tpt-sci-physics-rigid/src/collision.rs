//! Contact resolution and broad-phase (sweep-and-prune) collision detection.

use super::*;

/// Unit vector in the direction of `v` (assumes `v` is non-zero).
pub(crate) fn normalize(v: &DVector<f64>) -> DVector<f64> {
    let n = v.norm();
    v.clone() * (1.0 / n)
}

/// Resolve a single colliding pair of spheres: exchange the normal impulse
/// when the bodies are approaching, apply a Coulomb-bounded tangential
/// friction impulse, and push them apart to remove overlap.
pub(crate) fn resolve_pair(a: &mut Body, b: &mut Body, restitution: f64, friction: f64) {
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

        // Coulomb friction: clamp the tangential impulse needed to cancel
        // the relative tangential (contact-plane) velocity to
        // `mu * |normal impulse|`, applied along the tangential direction of
        // the post-normal-impulse relative velocity.
        if friction > 0.0 && j_imp > 0.0 {
            let rel_vel2 = b.velocity.clone() - a.velocity.clone();
            let vn2 = rel_vel2.dot(&normal);
            let tangent_vec = rel_vel2 - (normal.clone() * vn2);
            let tangent_speed = tangent_vec.norm();
            if tangent_speed > 1e-9 {
                // `tangent_dir` points along the existing relative tangential
                // velocity (b relative to a); the friction impulse must
                // oppose that motion, so apply it with the opposite sign
                // convention from the normal impulse (subtract from `b`, add
                // to `a`).
                let tangent_dir = tangent_vec * (1.0 / tangent_speed);
                let j_t_full = tangent_speed / (inv_mass_a + inv_mass_b);
                let j_t = j_t_full.min(friction * j_imp);
                let friction_impulse = tangent_dir * j_t;
                a.velocity = a.velocity.clone() + (friction_impulse.clone() * inv_mass_a);
                b.velocity = b.velocity.clone() - (friction_impulse * inv_mass_b);
            }
        }
    }

    // Positional correction: split the overlap by inverse mass so that the
    // heavier body moves less, eliminating sink-through.
    let overlap = sum_r - dist;
    let correction = normal * (overlap / (inv_mass_a + inv_mass_b));
    a.position = a.position.clone() - (correction.clone() * inv_mass_a);
    b.position = b.position.clone() + (correction * inv_mass_b);
}

/// `true` if the axis-aligned bounding boxes of `a` and `b` (each sphere's
/// center `±` radius on every shared axis) overlap.
pub(crate) fn aabb_overlap(a: &Body, b: &Body) -> bool {
    let dim = a.position.len().min(b.position.len());
    for axis in 0..dim {
        let a_min = a.position[axis] - a.radius;
        let a_max = a.position[axis] + a.radius;
        let b_min = b.position[axis] - b.radius;
        let b_max = b.position[axis] + b.radius;
        if a_max < b_min || b_max < a_min {
            return false;
        }
    }
    true
}

/// Sweep-and-prune broad-phase collision candidate search.
///
/// Sorts bodies by the lower extent of their axis-0 AABB interval and sweeps
/// an "active" set, testing each newly-swept body's full AABB against every
/// still-active body whose axis-0 interval could overlap it. This produces
/// exactly the set of index pairs `(i, j)` (`i < j`, indices into `bodies`)
/// that a brute-force `O(n^2)` all-pairs AABB check would produce, but visits
/// far fewer pairs when the bodies are sparsely distributed along axis 0.
pub(crate) fn sap_candidate_pairs(bodies: &[Body]) -> Vec<(usize, usize)> {
    let n = bodies.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let mi = bodies[i].position[0] - bodies[i].radius;
        let mj = bodies[j].position[0] - bodies[j].radius;
        mi.partial_cmp(&mj).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut pairs = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for idx in order {
        let min_x = bodies[idx].position[0] - bodies[idx].radius;
        // Drop actives whose interval has already ended before `idx` starts.
        active.retain(|&other| bodies[other].position[0] + bodies[other].radius >= min_x);
        for &other in &active {
            if aabb_overlap(&bodies[idx], &bodies[other]) {
                pairs.push(if idx < other {
                    (idx, other)
                } else {
                    (other, idx)
                });
            }
        }
        active.push(idx);
    }
    pairs
}
