//! # tpt-sci-astro
//!
//! Orbital-mechanics and coordinate-frame primitives for the `tpt-science`
//! pillar, built entirely from scratch on top of the in-house `tpt-math-linalg`
//! dense linear algebra (no external astrodynamics or geometry wrappers).
//!
//! The crate implements the **classical two-body problem** in an
//! Earth-Centered Inertial (ECI) reference frame:
//!
//! * Classical (Keplerian) [`OrbitalElements`], validated on construction.
//! * Conversion between Keplerian elements and ECI Cartesian state vectors
//!   ([`OrbitalElements::state_vector`] and [`OrbitalElements::from_state`]).
//! * Time propagation via Kepler's equation (`state -> mean anomaly -> advance
//!   -> solve -> true anomaly`) in [`OrbitalElements::propagate`].
//!
//! All angles are in **radians**. The model assumes an ideal point-mass central
//! body and neglects perturbations (J2, drag, third bodies, SRP, ...), so it is
//! exact only for the two-body problem.
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg::tpt_math_linalg_dense::{DVector, DMatrix};
//! use tpt_sci_astro::OrbitalElements;
//!
//! // A unit circular orbit about a unit-mass body.
//! let el = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
//! let (r, _v) = el.state_vector();
//! assert!((r.norm() - 1.0).abs() < 1e-9);
//! ```
//!
//! Licensed under either of MIT or Apache-2.0 at your option.

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

pub use error::AstroError;

mod error;

/// Gravitational parameter of the Earth, μ = GM, in km³·s⁻².
///
/// Useful as a default when working in Earth-Centered Inertial (ECI) frames
/// with kilometre and second units.
pub const EARTH_MU: f64 = 398_600.441_8;

/// A classical (Keplerian) set of orbital elements describing an elliptical
/// orbit in an Earth-Centered Inertial (ECI) frame.
///
/// All angles ([`OrbitalElements::i`], [`OrbitalElements::raan`],
/// [`OrbitalElements::argp`], [`OrbitalElements::nu`]) are stored in **radians**.
/// The orbit is restricted to the elliptical regime (`0 ≤ e < 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitalElements {
    /// Semi-major axis (distance units). Must be strictly positive.
    pub a: f64,
    /// Eccentricity. Restricted to `0 ≤ e < 1` (elliptical only).
    pub e: f64,
    /// Inclination (radians).
    pub i: f64,
    /// Right ascension of the ascending node (radians).
    pub raan: f64,
    /// Argument of periapsis (radians).
    pub argp: f64,
    /// True anomaly (radians).
    pub nu: f64,
    /// Gravitational parameter μ of the central body (distance³·time⁻²).
    pub mu: f64,
}

impl OrbitalElements {
    /// Construct and validate a set of orbital elements.
    ///
    /// # Errors
    ///
    /// Returns [`AstroError::InvalidElements`] when any invariant is violated:
    /// `a` must be positive, `e` must satisfy `0 ≤ e < 1` (elliptical only),
    /// `mu` must be positive, and all angles must be finite.
    pub fn new(
        a: f64,
        e: f64,
        i: f64,
        raan: f64,
        argp: f64,
        nu: f64,
        mu: f64,
    ) -> Result<Self, AstroError> {
        if !a.is_finite() || a <= 0.0 {
            return Err(AstroError::InvalidElements(format!(
                "semi-major axis must be > 0, got {a}"
            )));
        }
        if !e.is_finite() || !(0.0..1.0).contains(&e) {
            return Err(AstroError::InvalidElements(format!(
                "eccentricity must satisfy 0 <= e < 1, got {e}"
            )));
        }
        for (name, x) in [
            ("inclination", i),
            ("raan", raan),
            ("argument of periapsis", argp),
            ("true anomaly", nu),
        ] {
            if !x.is_finite() {
                return Err(AstroError::InvalidElements(format!(
                    "{name} must be finite"
                )));
            }
        }
        if !mu.is_finite() || mu <= 0.0 {
            return Err(AstroError::InvalidElements(format!(
                "gravitational parameter must be > 0, got {mu}"
            )));
        }
        Ok(Self {
            a,
            e,
            i,
            raan,
            argp,
            nu,
            mu,
        })
    }

    /// Orbital period `T = 2π √(a³ / μ)` (seconds for SI-like units).
    #[must_use]
    pub fn period(&self) -> f64 {
        const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
        TWO_PI * (self.a.powi(3) / self.mu).sqrt()
    }

    /// Compute the ECI position and velocity vectors from these elements.
    ///
    /// Returns `(r_eci, v_eci)`, each a length-3 [`DVector`]. The position is
    /// obtained in the perifocal frame and rotated into ECI; the velocity
    /// follows the same rotation.
    #[must_use]
    pub fn state_vector(&self) -> (DVector<f64>, DVector<f64>) {
        let p = self.a * (1.0 - self.e.powi(2));
        let r = p / (1.0 + self.e * self.nu.cos());
        let r_pf = DVector::from_vec(vec![r * self.nu.cos(), r * self.nu.sin(), 0.0]);

        let v_scale = (self.mu / p).sqrt();
        let v_pf = DVector::from_vec(vec![-self.nu.sin(), self.e + self.nu.cos(), 0.0]) * v_scale;

        let q = perifocal_to_eci(self.raan, self.i, self.argp);
        let r_eci = q.clone() * r_pf;
        let v_eci = q * v_pf;
        (r_eci, v_eci)
    }

    /// Recover classical orbital elements from an ECI position/velocity state.
    ///
    /// Implements the standard inverse two-body transform. Edge cases (nearly
    /// circular orbits and/or equatorial orbits) are handled by clamping the
    /// relevant angle to a conventional zero so the transformation stays
    /// well-defined.
    ///
    /// # Errors
    ///
    /// Returns [`AstroError::InvalidElements`] if the inputs are non-finite or
    /// `mu` is not positive, or [`AstroError::DegenerateGeometry`] if the state
    /// is degenerate (zero position magnitude, zero angular momentum, or a
    /// non-elliptical `a`).
    pub fn from_state(pos: &DVector<f64>, vel: &DVector<f64>, mu: f64) -> Result<Self, AstroError> {
        if !mu.is_finite() || mu <= 0.0 {
            return Err(AstroError::InvalidElements(format!(
                "gravitational parameter must be > 0, got {mu}"
            )));
        }
        for (name, v) in [("position", pos), ("velocity", vel)] {
            for (k, x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(AstroError::InvalidElements(format!(
                        "{name} component {k} must be finite"
                    )));
                }
            }
        }

        let r = pos.norm();
        let v = vel.norm();
        if r <= 0.0 {
            return Err(AstroError::DegenerateGeometry(
                "position magnitude is zero".to_string(),
            ));
        }

        let energy = v * v / 2.0 - mu / r;
        let a = -mu / (2.0 * energy);
        if !a.is_finite() || a <= 0.0 {
            return Err(AstroError::DegenerateGeometry(
                "semi-major axis is not positive (orbit is not elliptical)".to_string(),
            ));
        }

        let h = cross3(pos, vel);
        let h_mag = h.norm();
        if h_mag <= 0.0 {
            return Err(AstroError::DegenerateGeometry(
                "angular momentum is zero".to_string(),
            ));
        }

        let e_vec = (pos.clone() * (v * v - mu / r) - vel.clone() * pos.dot(vel)) / mu;
        let e = e_vec.norm();

        let i = (h[2] / h_mag).clamp(-1.0, 1.0).acos();

        // Node vector n = k × h, with k = [0,0,1].
        let n = DVector::from_vec(vec![-h[1], h[0], 0.0]);
        let n_mag = n.norm();

        let raan = if n_mag < 1e-12 {
            // Equatorial orbit: RAAN is undefined; choose 0.
            0.0
        } else {
            let mut raan = (n[0] / n_mag).clamp(-1.0, 1.0).acos();
            if n[1] < 0.0 {
                raan = 2.0 * std::f64::consts::PI - raan;
            }
            raan
        };

        let argp = if e < 1e-12 || n_mag < 1e-12 {
            // Circular orbit: argument of periapsis is undefined; choose 0.
            0.0
        } else {
            let mut argp = (n.dot(&e_vec) / (n_mag * e)).clamp(-1.0, 1.0).acos();
            if e_vec[2] < 0.0 {
                argp = 2.0 * std::f64::consts::PI - argp;
            }
            argp
        };

        let nu = if e < 1e-12 {
            // Circular orbit: true anomaly measured from the node line.
            if n_mag < 1e-12 {
                0.0
            } else {
                let mut nu = (n.dot(pos) / (n_mag * r)).clamp(-1.0, 1.0).acos();
                if pos.dot(vel) < 0.0 {
                    nu = 2.0 * std::f64::consts::PI - nu;
                }
                nu
            }
        } else {
            let mut nu = (e_vec.dot(pos) / (e * r)).clamp(-1.0, 1.0).acos();
            if pos.dot(vel) < 0.0 {
                nu = 2.0 * std::f64::consts::PI - nu;
            }
            nu
        };

        OrbitalElements::new(a, e, i, raan, argp, nu, mu)
    }

    /// Advance the orbit by `dt` seconds and return the new elements.
    ///
    /// Propagation is performed by converting the current true anomaly to a
    /// mean anomaly, applying the mean motion over `dt`, solving Kepler's
    /// equation for the new eccentric anomaly (Newton iteration), and converting
    /// back to a true anomaly. The other elements (`a`, `e`, `i`, `raan`,
    /// `argp`, `mu`) are unchanged.
    #[must_use]
    pub fn propagate(&self, dt: f64) -> OrbitalElements {
        let n_motion = (self.mu / self.a.powi(3)).sqrt();
        let e = self.e;

        let e0 = true_to_eccentric(self.nu, e);
        let m0 = e0 - e * e0.sin();
        let m1 = m0 + n_motion * dt;
        let e1 = solve_kepler(m1, e);
        let nu1 = eccentric_to_true(e1, e);

        let mut nu = nu1;
        // Keep nu in a canonical [0, 2π) range for stable comparisons.
        nu = nu.rem_euclid(2.0 * std::f64::consts::PI);

        OrbitalElements {
            a: self.a,
            e: self.e,
            i: self.i,
            raan: self.raan,
            argp: self.argp,
            nu,
            mu: self.mu,
        }
    }
}

/// The 3×3 rotation matrix `Q` that maps perifocal-frame coordinates to ECI.
///
/// `Q` is built from the right ascension of the ascending node `raan`, the
/// inclination `i`, and the argument of periapsis `argp`.
#[must_use]
pub fn perifocal_to_eci(raan: f64, i: f64, argp: f64) -> DMatrix<f64> {
    let c_o = raan.cos();
    let s_o = raan.sin();
    let ci = i.cos();
    let si = i.sin();
    let cw = argp.cos();
    let sw = argp.sin();

    DMatrix::from_fn(3, 3, |row, col| match (row, col) {
        (0, 0) => c_o * cw - s_o * sw * ci,
        (0, 1) => -c_o * sw - s_o * cw * ci,
        (0, 2) => s_o * si,
        (1, 0) => s_o * cw + c_o * sw * ci,
        (1, 1) => -s_o * sw + c_o * cw * ci,
        (1, 2) => -c_o * si,
        (2, 0) => sw * si,
        (2, 1) => cw * si,
        (2, 2) => ci,
        _ => 0.0,
    })
}

/// Cross product of two 3-vectors, `a × b`.
///
/// The inputs are treated as 3-vectors; only the first three components are
/// used. Returns a length-3 [`DVector`].
#[must_use]
pub fn cross3(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    DVector::from_vec(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Convert a true anomaly `nu` to an eccentric anomaly `E` for eccentricity `e`.
#[must_use]
pub fn true_to_eccentric(nu: f64, e: f64) -> f64 {
    let s = (1.0 - e).sqrt() * (nu / 2.0).sin();
    let c = (1.0 + e).sqrt() * (nu / 2.0).cos();
    2.0 * s.atan2(c)
}

/// Convert an eccentric anomaly `E` to a true anomaly `nu` for eccentricity `e`.
#[must_use]
pub fn eccentric_to_true(ecc: f64, e: f64) -> f64 {
    let s = (1.0 + e).sqrt() * (ecc / 2.0).sin();
    let c = (1.0 - e).sqrt() * (ecc / 2.0).cos();
    2.0 * s.atan2(c)
}

/// Solve Kepler's equation `M = E - e·sin(E)` for `E` via Newton iteration.
///
/// Converges quadratically for `0 ≤ e < 1`; `E` is initialised to `M`.
#[must_use]
pub fn solve_kepler(m: f64, e: f64) -> f64 {
    let mut ecc = m;
    for _ in 0..50 {
        let f = ecc - e * ecc.sin() - m;
        let fp = 1.0 - e * ecc.cos();
        let delta = f / fp;
        ecc -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    ecc
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn vec3(x: f64, y: f64, z: f64) -> DVector<f64> {
        DVector::from_vec(vec![x, y, z])
    }

    #[test]
    fn circular_orbit_state_norms() {
        // e = 0, a = 1, mu = 1 -> r = 1, v = 1 for every true anomaly.
        let el = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        for nu in [0.0, 0.3, 1.0, 2.5, 5.0] {
            let e = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, nu, 1.0).unwrap();
            let (r, v) = e.state_vector();
            assert_abs_diff_eq!(r.norm(), 1.0, epsilon = 1e-9);
            assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 1e-9);
        }
        let _ = el;
    }

    #[test]
    fn period_unit_circle() {
        let el = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(el.period(), 2.0 * std::f64::consts::PI, epsilon = 1e-9);
    }

    #[test]
    fn propagate_half_period_flips_position() {
        let el = OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        let (r0, _) = el.state_vector();
        let advanced = el.propagate(el.period() / 2.0);
        let (r1, _) = advanced.state_vector();
        assert_abs_diff_eq!(r1[0], -r0[0], epsilon = 1e-6);
        assert_abs_diff_eq!(r1[1], -r0[1], epsilon = 1e-6);
        assert_abs_diff_eq!(r1[2], -r0[2], epsilon = 1e-6);
    }

    #[test]
    fn round_trip_elements() {
        let el = OrbitalElements::new(2.0, 0.3, 0.4, 0.2, 0.1, 0.5, 1.0).unwrap();
        let (r, v) = el.state_vector();
        let recovered = OrbitalElements::from_state(&r, &v, 1.0).unwrap();
        assert_abs_diff_eq!(recovered.a, el.a, epsilon = 1e-6);
        assert_abs_diff_eq!(recovered.e, el.e, epsilon = 1e-6);
        assert_abs_diff_eq!(recovered.i, el.i, epsilon = 1e-6);
    }

    #[test]
    fn propagate_zero_is_identity() {
        let el = OrbitalElements::new(2.0, 0.3, 0.4, 0.2, 0.1, 0.5, 1.0).unwrap();
        let advanced = el.propagate(0.0);
        // Compare canonical [0, 2π) representations.
        let n0 = el.nu.rem_euclid(2.0 * std::f64::consts::PI);
        let n1 = advanced.nu.rem_euclid(2.0 * std::f64::consts::PI);
        assert_abs_diff_eq!(n0, n1, epsilon = 1e-9);
    }

    #[test]
    fn invalid_elements_rejected() {
        assert!(OrbitalElements::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).is_err());
        assert!(OrbitalElements::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0).is_err());
        assert!(OrbitalElements::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(OrbitalElements::new(1.0, 0.0, f64::NAN, 0.0, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn degenerate_state_rejected() {
        let pos = vec3(0.0, 0.0, 0.0);
        let vel = vec3(1.0, 0.0, 0.0);
        assert!(OrbitalElements::from_state(&pos, &vel, 1.0).is_err());
    }

    #[test]
    fn cross3_orthogonal() {
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        let c = cross3(&a, &b);
        assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[2], 1.0, epsilon = 1e-12);
    }
}
