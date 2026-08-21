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
//! * First-order secular `J₂` perturbation: [`OrbitalElements::propagate_j2`]
//!   and [`OrbitalElements::j2_secular_rates`] give the dominant long-term
//!   nodal regression and apsidal precession for oblate-body missions (e.g.
//!   sun-synchronous orbit design). [`OrbitalElements::propagate_j4`] /
//!   [`OrbitalElements::j4_secular_rates`] extend this with the `J₄` zonal
//!   term.
//! * Atmospheric drag: [`atmospheric_density`] (single-band exponential
//!   model) and [`OrbitalElements::propagate_drag`] /
//!   [`OrbitalElements::drag_da_dt`] for secular along-track decay.
//! * Simplified (Kozai-Lidov, quadrupole-order) third-body secular
//!   perturbation: [`OrbitalElements::third_body_secular_rates`] /
//!   [`OrbitalElements::propagate_third_body`].
//! * Cannonball solar radiation pressure with a cylindrical Earth-shadow
//!   eclipse test: [`srp_acceleration`], [`in_earth_shadow`],
//!   [`OrbitalElements::srp_acceleration_vector`].
//!
//! All angles are in **radians**. The model assumes an ideal point-mass central
//! body for the pure two-body propagation; each perturbation above is an
//! independent first-order secular add-on (not a combined integrated force
//! model), so short-periodic oscillations are not captured and the
//! perturbations are not accumulated together automatically.
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

/// Earth's second zonal harmonic `J₂` (dimensionless), the leading oblateness
/// term that drives nodal regression and apsidal precession.
pub const EARTH_J2: f64 = 1.082_626_68e-3;

/// Earth's equatorial radius `Rₑ` in km, the reference length for the `J₂`
/// perturbation (the perturbation scales as `(Rₑ/p)²`).
pub const EARTH_RADIUS_EQ: f64 = 6378.137;

/// Earth's fourth zonal harmonic `J₄` (dimensionless), the next zonal
/// oblateness term after `J₂`.
pub const EARTH_J4: f64 = -1.620_836_15e-6;

/// Reference atmospheric density (kg/m³) for the single-band exponential
/// Earth atmosphere model, evaluated at [`EARTH_ATM_H0_KM`].
///
/// Coefficients are consistent with the ~400 km altitude band of the
/// exponential atmospheric density model tabulated in Vallado,
/// *Fundamentals of Astrodynamics and Applications*, 4th ed., Table 8-4
/// ("Exponential Atmospheric Density Model"), which piecewise-fits Earth's
/// atmosphere with `ρ(h) = ρ0 · exp(-(h - h0) / H)` in each altitude band.
/// Adequate for order-of-magnitude drag work in the ~300-500 km LEO band;
/// not a substitute for a full reference atmosphere (e.g. NRLMSISE-00) for
/// precision work.
pub const EARTH_ATM_RHO0_KG_M3: f64 = 5.428e-13;

/// Reference altitude (km) for [`EARTH_ATM_RHO0_KG_M3`] /
/// [`EARTH_ATM_SCALE_HEIGHT_KM`].
pub const EARTH_ATM_H0_KM: f64 = 400.0;

/// Atmospheric scale height (km) for the ~400 km exponential density band
/// (see [`EARTH_ATM_RHO0_KG_M3`]).
pub const EARTH_ATM_SCALE_HEIGHT_KM: f64 = 58.515;

/// Solar radiation pressure at 1 AU, in N/m² (solar constant / speed of
/// light, ≈ 1361 W/m² / 2.998e8 m/s).
pub const SOLAR_PRESSURE_1AU: f64 = 4.56e-6;

/// One astronomical unit, in km.
pub const ASTRONOMICAL_UNIT_KM: f64 = 1.495_978_707e8;

/// Gravitational parameter of the Sun, `μ☉ = GM☉`, in km³·s⁻².
pub const SUN_MU: f64 = 1.327_124_400_18e11;

/// Gravitational parameter of the Moon, `μ_Moon = GM_Moon`, in km³·s⁻².
pub const MOON_MU: f64 = 4_902.800_66;

/// Mean Earth-Moon distance, in km.
pub const MOON_DISTANCE_KM: f64 = 384_400.0;

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

    /// First-order secular J2 perturbation rates (Brouwer/Lyddane secular
    /// terms) for this orbit: the time derivatives of the right ascension of
    /// the ascending node (`Ω̇`) and the argument of periapsis (`ω̇`).
    ///
    /// These are the dominant long-term perturbations from Earth's oblateness:
    /// `Ω̇` (nodal regression) and `ω̇` (apsidal precession). They depend on the
    /// `J₂` coefficient and the reference equatorial radius `r_eq` (both
    /// supplied so the same routine works for any oblate body). The semi-major
    /// axis, eccentricity, and inclination are constant to first order in `J₂`.
    ///
    /// Returns `(raan_dot, argp_dot)` in radians per unit time.
    #[must_use]
    pub fn j2_secular_rates(&self, j2: f64, r_eq: f64) -> (f64, f64) {
        let n = (self.mu / self.a.powi(3)).sqrt();
        let p = self.a * (1.0 - self.e.powi(2));
        let factor = 1.5 * n * j2 * (r_eq / p).powi(2);
        let ci = self.i.cos();
        let raan_dot = -factor * ci;
        let argp_dot = 0.5 * factor * (5.0 * ci * ci - 1.0);
        (raan_dot, argp_dot)
    }

    /// Propagate the orbit under the first-order secular `J₂` perturbation for
    /// `dt` units of time.
    ///
    /// The in-plane motion (mean anomaly / true anomaly) is advanced with the
    /// two-body mean motion, while the secular `J₂` drifts are added to the
    /// right ascension of the ascending node and the argument of periapsis. The
    /// semi-major axis, eccentricity, and inclination are held fixed (they are
    /// constant to first order in `J₂`). This is the standard model used for
    /// long-term RAAN drift and sun-synchronous orbit design; it does not
    /// include short-periodic `J₂` oscillations.
    ///
    /// `j2` is the body's zonal harmonic (e.g. [`EARTH_J2`]) and `r_eq` its
    /// equatorial radius (e.g. [`EARTH_RADIUS_EQ`]).
    #[must_use]
    pub fn propagate_j2(&self, dt: f64, j2: f64, r_eq: f64) -> OrbitalElements {
        let in_plane = self.propagate(dt);
        let (raan_dot, argp_dot) = self.j2_secular_rates(j2, r_eq);
        let two_pi = 2.0 * std::f64::consts::PI;
        OrbitalElements {
            a: in_plane.a,
            e: in_plane.e,
            i: in_plane.i,
            raan: (in_plane.raan + raan_dot * dt).rem_euclid(two_pi),
            argp: (in_plane.argp + argp_dot * dt).rem_euclid(two_pi),
            nu: in_plane.nu,
            mu: in_plane.mu,
        }
    }

    /// Secular along-track semi-major-axis decay rate due to atmospheric
    /// drag (distance units of `a` per unit time, e.g. km/s).
    ///
    /// Uses the standard first-order averaged drag decay rate (Vallado,
    /// *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.6.3
    /// "Atmospheric Drag"; also King-Hele, *Theory of Satellite Orbits in an
    /// Atmosphere*, 1964):
    ///
    /// `da/dt = -(ρ · Cd·A/m) · n · a² · √((1+e)/(1-e))`
    ///
    /// `ρ` is the atmospheric density ([`atmospheric_density`]) evaluated at
    /// the current perigee altitude (drag is dominated by the perigee pass
    /// for eccentric orbits; for a circular orbit perigee altitude reduces
    /// to `a - Rₑ`), `n` is the mean motion, and `cd_a_over_m` is the
    /// ballistic drag term `Cd·A/m` in m²/kg. The `a²` factor is evaluated
    /// in metres so it is dimensionally consistent with `ρ` (kg/m³) and
    /// `cd_a_over_m` (m²/kg); the result is converted back to the same
    /// distance unit as `a` (assumed km, matching [`EARTH_RADIUS_EQ`]).
    #[must_use]
    pub fn drag_da_dt(&self, cd_a_over_m: f64) -> f64 {
        let perigee_altitude_km = self.a * (1.0 - self.e) - EARTH_RADIUS_EQ;
        let rho = atmospheric_density(perigee_altitude_km);
        let n = (self.mu / self.a.powi(3)).sqrt();
        let a_m = self.a * 1000.0;
        let ecc_factor = ((1.0 + self.e) / (1.0 - self.e)).sqrt();
        let da_dt_m_per_s = -rho * cd_a_over_m * n * a_m * a_m * ecc_factor;
        da_dt_m_per_s / 1000.0
    }

    /// Propagate the orbit including secular along-track decay of the
    /// semi-major axis due to atmospheric drag, over `dt` seconds.
    ///
    /// Combines ordinary two-body propagation ([`OrbitalElements::propagate`])
    /// with a linear-in-`dt` decay of `a` from
    /// [`OrbitalElements::drag_da_dt`]. Eccentricity, inclination, and the
    /// node/apsis angles are held fixed; a full drag model also
    /// circularizes the orbit by damping `e`, but this first-order model
    /// captures only the dominant `a` decay (matching the level of the
    /// existing `J₂` model). `cd_a_over_m` is the ballistic term `Cd·A/m` in
    /// m²/kg.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_astro::{OrbitalElements, EARTH_MU};
    /// let el = OrbitalElements::new(6778.0, 0.001, 0.9, 0.0, 0.0, 0.0, EARTH_MU).unwrap();
    /// let decayed = el.propagate_drag(3600.0, 0.02);
    /// assert!(decayed.a < el.a); // drag always shrinks the semi-major axis
    /// ```
    #[must_use]
    pub fn propagate_drag(&self, dt: f64, cd_a_over_m: f64) -> OrbitalElements {
        let in_plane = self.propagate(dt);
        let da_dt = self.drag_da_dt(cd_a_over_m);
        let new_a = (in_plane.a + da_dt * dt).max(EARTH_RADIUS_EQ * 1e-3);
        OrbitalElements {
            a: new_a,
            ..in_plane
        }
    }

    /// First-order secular perturbation rates due to a third body (e.g. the
    /// Sun or Moon) restricted to a circular orbit lying in the reference
    /// plane, to quadrupole order in `a / d` (the classic Kozai (1959) /
    /// Lidov (1961) restricted third-body secular theory; see also Murray &
    /// Dermott, *Solar System Dynamics*, §7, and Naoz, "The Eccentric
    /// Kozai-Lidov Effect", *ARA&A* 54 (2016), the quadrupole term with the
    /// perturber's own eccentricity set to zero).
    ///
    /// `mu_third` is the perturbing body's gravitational parameter (e.g.
    /// [`SUN_MU`] or [`MOON_MU`]) and `dist_third` its distance from the
    /// central body (e.g. [`ASTRONOMICAL_UNIT_KM`] or [`MOON_DISTANCE_KM`]).
    ///
    /// Returns `(raan_dot, argp_dot, i_dot, e_dot)`. Unlike the pure `J₂`
    /// model, a third body secularly couples eccentricity and inclination
    /// (the Kozai-Lidov mechanism): to first order the combination
    /// `(1-e²)cos²i` is conserved by `(i_dot, e_dot)` alone.
    ///
    /// This is a simplification: the real Sun and Moon orbits are neither
    /// circular nor coplanar with the Earth's equator, so this captures the
    /// dominant secular behaviour rather than exact lunisolar perturbations.
    #[must_use]
    pub fn third_body_secular_rates(&self, mu_third: f64, dist_third: f64) -> (f64, f64, f64, f64) {
        let n = (self.mu / self.a.powi(3)).sqrt();
        let n3_sq = mu_third / dist_third.powi(3);
        let beta = n3_sq / n;

        let e = self.e;
        let e2 = e * e;
        let ome2 = (1.0 - e2).max(1e-12);
        let sqrt_ome2 = ome2.sqrt();

        let ci = self.i.cos();
        let si = self.i.sin();
        let ci2 = ci * ci;
        let si2 = si * si;

        let s2w = (2.0 * self.argp).sin();
        let c2w = (2.0 * self.argp).cos();

        let e_dot = (15.0 / 8.0) * beta * sqrt_ome2 * e * si2 * s2w;
        let i_dot = -(15.0 / 16.0) * beta * e2 * s2w * (2.0 * si * ci) / sqrt_ome2;
        let raan_dot = (3.0 / 8.0) * beta * (ci / sqrt_ome2) * (2.0 + 3.0 * e2 - 5.0 * e2 * c2w);
        let argp_dot_e_term =
            (3.0 / 8.0) * beta * sqrt_ome2 * ((3.0 * ci2 - 1.0) + 5.0 * si2 * c2w);
        let argp_dot_i_term =
            (3.0 / 8.0) * beta * (ci2 / sqrt_ome2) * (2.0 + 3.0 * e2 - 5.0 * e2 * c2w);
        let argp_dot = argp_dot_e_term + argp_dot_i_term;

        (raan_dot, argp_dot, i_dot, e_dot)
    }

    /// Propagate the orbit under the simplified secular third-body
    /// perturbation ([`OrbitalElements::third_body_secular_rates`]) for `dt`
    /// seconds.
    ///
    /// # Errors
    ///
    /// Returns [`AstroError::InvalidElements`] if the perturbed elements (in
    /// particular `e`, which a third body secularly pumps via the
    /// Kozai-Lidov mechanism) leave the valid range; callers integrating
    /// over long spans should use small steps.
    pub fn propagate_third_body(
        &self,
        dt: f64,
        mu_third: f64,
        dist_third: f64,
    ) -> Result<OrbitalElements, AstroError> {
        let in_plane = self.propagate(dt);
        let (raan_dot, argp_dot, i_dot, e_dot) =
            self.third_body_secular_rates(mu_third, dist_third);
        let two_pi = 2.0 * std::f64::consts::PI;
        OrbitalElements::new(
            in_plane.a,
            (in_plane.e + e_dot * dt).clamp(0.0, 1.0 - 1e-9),
            (in_plane.i + i_dot * dt).clamp(0.0, std::f64::consts::PI),
            (in_plane.raan + raan_dot * dt).rem_euclid(two_pi),
            (in_plane.argp + argp_dot * dt).rem_euclid(two_pi),
            in_plane.nu,
            in_plane.mu,
        )
    }

    /// Cannonball-model solar radiation pressure acceleration vector in ECI,
    /// in km/s², including the cylindrical Earth-shadow eclipse test
    /// ([`in_earth_shadow`]).
    ///
    /// `cr` is the dimensionless radiation-pressure coefficient (`1` for a
    /// perfectly absorbing surface, up to `2` for perfectly reflecting),
    /// `area_to_mass_m2_per_kg` is `A/m` in m²/kg, and `sun_pos_km` is the
    /// Sun's ECI position relative to the central body (km). Returns the
    /// zero vector when the satellite is in Earth's shadow.
    #[must_use]
    pub fn srp_acceleration_vector(
        &self,
        cr: f64,
        area_to_mass_m2_per_kg: f64,
        sun_pos_km: &DVector<f64>,
    ) -> DVector<f64> {
        let (r_eci, _v) = self.state_vector();
        if in_earth_shadow(&r_eci, sun_pos_km, EARTH_RADIUS_EQ) {
            return DVector::from_vec(vec![0.0, 0.0, 0.0]);
        }
        let sun_dist = sun_pos_km.norm();
        if sun_dist <= 0.0 {
            return DVector::from_vec(vec![0.0, 0.0, 0.0]);
        }
        // Force direction is Sun -> satellite; since |r_eci| << sun_dist this
        // is well-approximated by the anti-sunward direction from Earth.
        let dir = sun_pos_km.clone() * (-1.0 / sun_dist);
        let mag = srp_acceleration(cr, area_to_mass_m2_per_kg, sun_dist);
        dir * mag
    }

    /// Combined first-order secular `J₂` + `J₄` zonal-harmonic perturbation
    /// rates (nodal regression `Ω̇` and apsidal precession `ω̇`).
    ///
    /// Extends [`OrbitalElements::j2_secular_rates`] with the next zonal
    /// term. The `J₄` contribution follows the standard combined-zonal
    /// secular theory (e.g. Vallado, *Fundamentals of Astrodynamics and
    /// Applications*, §9.2 "Combined Effects of Zonal Harmonics"; Schaub &
    /// Junkins, *Analytical Mechanics of Space Systems*, ch. 9): it is a
    /// small correction, of relative order `(J₄/J₂)·(Rₑ/p)²` versus the `J₂`
    /// term, that partially offsets the `J₂` nodal regression for typical
    /// low/mid inclinations.
    ///
    /// Returns `(raan_dot, argp_dot)` in radians per unit time, `J₂` and
    /// `J₄` combined.
    #[must_use]
    pub fn j4_secular_rates(&self, j2: f64, j4: f64, r_eq: f64) -> (f64, f64) {
        let (raan_dot_j2, argp_dot_j2) = self.j2_secular_rates(j2, r_eq);

        let n = (self.mu / self.a.powi(3)).sqrt();
        let p = self.a * (1.0 - self.e.powi(2));
        let factor4 = n * j4 * (r_eq / p).powi(4);
        let ci = self.i.cos();
        let si2 = self.i.sin().powi(2);

        let raan_dot_j4 = (15.0 / 32.0) * factor4 * ci * (12.0 - 21.0 * si2);
        let argp_dot_j4 = -(45.0 / 128.0) * factor4 * (8.0 - 40.0 * si2 + 35.0 * si2 * si2);

        (raan_dot_j2 + raan_dot_j4, argp_dot_j2 + argp_dot_j4)
    }

    /// Propagate the orbit under the combined secular `J₂` + `J₄`
    /// perturbation for `dt` units of time.
    ///
    /// See [`OrbitalElements::j4_secular_rates`] and
    /// [`OrbitalElements::propagate_j2`] (the `J₂`-only analogue, whose
    /// in-plane / secular-drift structure this mirrors exactly).
    #[must_use]
    pub fn propagate_j4(&self, dt: f64, j2: f64, j4: f64, r_eq: f64) -> OrbitalElements {
        let in_plane = self.propagate(dt);
        let (raan_dot, argp_dot) = self.j4_secular_rates(j2, j4, r_eq);
        let two_pi = 2.0 * std::f64::consts::PI;
        OrbitalElements {
            a: in_plane.a,
            e: in_plane.e,
            i: in_plane.i,
            raan: (in_plane.raan + raan_dot * dt).rem_euclid(two_pi),
            argp: (in_plane.argp + argp_dot * dt).rem_euclid(two_pi),
            nu: in_plane.nu,
            mu: in_plane.mu,
        }
    }
}

/// Exponential atmospheric density model, `ρ(h) = ρ0 · exp(-(h - h0) / H)`.
///
/// A single-band exponential fit to Earth's atmosphere (see
/// [`EARTH_ATM_RHO0_KG_M3`] for the reference and its limitations), adequate
/// for order-of-magnitude drag estimates in the ~300-500 km LEO band.
/// `altitude_km` is the geodetic altitude above the Earth's surface in km.
/// Returns density in kg/m³.
///
/// # Examples
///
/// ```
/// use tpt_sci_astro::{atmospheric_density, EARTH_ATM_H0_KM, EARTH_ATM_RHO0_KG_M3};
/// let rho = atmospheric_density(EARTH_ATM_H0_KM);
/// assert!((rho - EARTH_ATM_RHO0_KG_M3).abs() < 1e-20);
/// assert!(atmospheric_density(500.0) < rho); // density falls off with altitude
/// ```
#[must_use]
pub fn atmospheric_density(altitude_km: f64) -> f64 {
    EARTH_ATM_RHO0_KG_M3 * (-(altitude_km - EARTH_ATM_H0_KM) / EARTH_ATM_SCALE_HEIGHT_KM).exp()
}

/// Cylindrical Earth-shadow (eclipse) test.
///
/// Returns `true` when `pos` (satellite position relative to the central
/// body, km) lies within the infinite cylinder of radius `earth_radius`
/// extending anti-sunward from the Earth, i.e. the satellite is
/// geometrically eclipsed under the standard cylindrical-shadow
/// approximation (ignoring the Sun's finite angular size / penumbra, which
/// is adequate for cannonball SRP on/off modeling).
///
/// `sun_pos` is the Sun's position relative to the central body (same frame
/// and units as `pos`, e.g. ECI km).
#[must_use]
pub fn in_earth_shadow(pos: &DVector<f64>, sun_pos: &DVector<f64>, earth_radius: f64) -> bool {
    let sun_dist = sun_pos.norm();
    if sun_dist <= 0.0 {
        return false;
    }
    let sun_dir = sun_pos.clone() * (1.0 / sun_dist);
    let along = pos.dot(&sun_dir);
    if along >= 0.0 {
        // Satellite is on the sunward side (or in the terminator plane): lit.
        return false;
    }
    let perp = pos.clone() - sun_dir * along;
    perp.norm() < earth_radius
}

/// Cannonball-model solar radiation pressure acceleration magnitude, in
/// km/s².
///
/// `cr` is the dimensionless radiation-pressure coefficient (`1` for a
/// perfectly absorbing surface, up to `2` for perfectly reflecting),
/// `area_to_mass_m2_per_kg` is `A/m` in m²/kg, and `sun_distance_km` is the
/// satellite-to-Sun distance. `P_srp` at 1 AU is [`SOLAR_PRESSURE_1AU`],
/// scaled by the inverse-square law to the actual Sun distance:
/// `F = P_srp · Cr · A/m`.
#[must_use]
pub fn srp_acceleration(cr: f64, area_to_mass_m2_per_kg: f64, sun_distance_km: f64) -> f64 {
    let p_srp = SOLAR_PRESSURE_1AU * (ASTRONOMICAL_UNIT_KM / sun_distance_km).powi(2);
    let accel_m_s2 = p_srp * cr * area_to_mass_m2_per_kg;
    accel_m_s2 / 1000.0
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
/// Converges quadratically for `0 ≤ e < 1`. The iteration is seeded with
/// `E₀ = M + e·sin(M)`, a first-order series approximation of the solution
/// (Danby 1992, ch. 6). Unlike a bare `E₀ = M` seed this stays close to the
/// root for near-parabolic orbits (`e → 1`), where `E ≈ M + e` and a `M` seed
/// degrades badly. A finite-difference step guards the derivative singularity
/// as `e → 1` so the loop still terminates when `1 - e·cos(E)` approaches zero.
#[must_use]
pub fn solve_kepler(m: f64, e: f64) -> f64 {
    debug_assert!((0.0..1.0).contains(&e), "solve_kepler requires 0 ≤ e < 1");
    let mut ecc = m + e * m.sin();
    for _ in 0..60 {
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
    fn solve_kepler_seed_is_accurate_at_high_eccentricity() {
        // With the `E0 = M + e·sin(M)` seed the residual stays tiny even as
        // e -> 1, where a bare `M` seed degrades. Check several M, e = 0.9.
        for m in [0.1, 1.0, 2.5, 5.0] {
            let e = 0.9;
            let ecc = solve_kepler(m, e);
            let residual = (ecc - e * ecc.sin() - m).abs();
            assert!(residual < 1e-10, "M={m} residual={residual}");
        }
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

    #[test]
    fn j2_rates_match_formula() {
        // LEO orbit: a = 7000 km, e = 0.01, i = 50 deg.
        let el = OrbitalElements::new(7000.0, 0.01, 50.0_f64.to_radians(), 0.0, 0.0, 0.0, EARTH_MU)
            .unwrap();
        let (raan_dot, argp_dot) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
        let n = (EARTH_MU / el.a.powi(3)).sqrt();
        let p = el.a * (1.0 - el.e.powi(2));
        let factor = 1.5 * n * EARTH_J2 * (EARTH_RADIUS_EQ / p).powi(2);
        let ci = el.i.cos();
        assert_abs_diff_eq!(raan_dot, -factor * ci, epsilon = 1e-12);
        assert_abs_diff_eq!(
            argp_dot,
            0.5 * factor * (5.0 * ci * ci - 1.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn j2_regresses_raan_for_prograde() {
        // A 50° inclined LEO orbit should see its RAAN regress (decrease) over
        // a day, by roughly the analytic secular rate.
        let el = OrbitalElements::new(7000.0, 0.01, 50.0_f64.to_radians(), 1.0, 0.0, 0.0, EARTH_MU)
            .unwrap();
        let dt = 86_400.0; // one day, seconds
        let advanced = el.propagate_j2(dt, EARTH_J2, EARTH_RADIUS_EQ);
        let (raan_dot, _) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
        let expected = (el.raan + raan_dot * dt).rem_euclid(2.0 * std::f64::consts::PI);
        assert_abs_diff_eq!(advanced.raan, expected, epsilon = 1e-9);
        // a, e, i are preserved by the secular model.
        assert_abs_diff_eq!(advanced.a, el.a, epsilon = 1e-9);
        assert_abs_diff_eq!(advanced.e, el.e, epsilon = 1e-12);
        assert_abs_diff_eq!(advanced.i, el.i, epsilon = 1e-12);
        // Prograde (i < 90°): RAAN regresses.
        assert!(raan_dot < 0.0);
    }

    #[test]
    fn atmospheric_density_reference_and_decay() {
        assert_abs_diff_eq!(
            atmospheric_density(EARTH_ATM_H0_KM),
            EARTH_ATM_RHO0_KG_M3,
            epsilon = 1e-20
        );
        // Monotonically decreasing with altitude.
        let rho_low = atmospheric_density(300.0);
        let rho_mid = atmospheric_density(400.0);
        let rho_high = atmospheric_density(600.0);
        assert!(rho_low > rho_mid);
        assert!(rho_mid > rho_high);
        assert!(rho_high > 0.0);
    }

    #[test]
    fn drag_decay_shrinks_semi_major_axis() {
        // ~400 km altitude near-circular LEO orbit, modest ballistic term.
        let el = OrbitalElements::new(
            6778.0,
            0.001,
            51.6_f64.to_radians(),
            0.0,
            0.0,
            0.0,
            EARTH_MU,
        )
        .unwrap();
        let cd_a_over_m = 0.02; // m^2/kg
        let da_dt = el.drag_da_dt(cd_a_over_m);
        // Drag always removes energy: da/dt must be strictly negative.
        assert!(da_dt < 0.0, "da_dt = {da_dt}");
        // Order-of-magnitude sanity: decay should be small but nonzero over
        // one day (tens of metres, not kilometres, for this modest BC).
        let decay_per_day_km = da_dt.abs() * 86_400.0;
        assert!(
            decay_per_day_km > 1e-6,
            "decay_per_day_km = {decay_per_day_km}"
        );
        assert!(
            decay_per_day_km < 5.0,
            "decay_per_day_km = {decay_per_day_km}"
        );

        let advanced = el.propagate_drag(3600.0, cd_a_over_m);
        assert!(advanced.a < el.a);
    }

    #[test]
    fn drag_rejects_denser_lower_orbit_faster() {
        // A lower orbit sees denser atmosphere and should decay faster.
        let hi = OrbitalElements::new(6978.0, 0.0, 0.5, 0.0, 0.0, 0.0, EARTH_MU).unwrap();
        let lo = OrbitalElements::new(6678.0, 0.0, 0.5, 0.0, 0.0, 0.0, EARTH_MU).unwrap();
        let bc = 0.02;
        assert!(lo.drag_da_dt(bc).abs() > hi.drag_da_dt(bc).abs());
    }

    #[test]
    fn shadow_function_eclipses_only_behind_earth() {
        let sun_pos = vec3(ASTRONOMICAL_UNIT_KM, 0.0, 0.0);
        // Satellite on the sunward side: never eclipsed.
        let lit = vec3(7000.0, 0.0, 0.0);
        assert!(!in_earth_shadow(&lit, &sun_pos, EARTH_RADIUS_EQ));
        // Satellite directly behind Earth, within the shadow cylinder.
        let eclipsed = vec3(-7000.0, 0.0, 0.0);
        assert!(in_earth_shadow(&eclipsed, &sun_pos, EARTH_RADIUS_EQ));
        // Satellite behind Earth but offset far enough to miss the cylinder.
        let grazing = vec3(-7000.0, 20_000.0, 0.0);
        assert!(!in_earth_shadow(&grazing, &sun_pos, EARTH_RADIUS_EQ));
    }

    #[test]
    fn srp_acceleration_vector_zero_in_shadow() {
        let sun_pos = vec3(ASTRONOMICAL_UNIT_KM, 0.0, 0.0);
        // Circular orbit; nu = pi puts the satellite on the -x side (opposite
        // the Sun), i.e. in eclipse for this equatorial, zero-inclination case.
        let el = OrbitalElements::new(7000.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::PI, EARTH_MU)
            .unwrap();
        let a_shadow = el.srp_acceleration_vector(1.5, 0.02, &sun_pos);
        assert_abs_diff_eq!(a_shadow.norm(), 0.0, epsilon = 1e-30);

        // nu = 0 puts the satellite on the +x (sunward) side: lit, nonzero SRP.
        let el_lit = OrbitalElements::new(7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, EARTH_MU).unwrap();
        let a_lit = el_lit.srp_acceleration_vector(1.5, 0.02, &sun_pos);
        assert!(a_lit.norm() > 0.0);
        // Force points away from the Sun: the satellite sits on the +x
        // (sunward) side here, so the anti-sunward push is in -x.
        assert!(a_lit[0] < 0.0);
    }

    #[test]
    fn srp_magnitude_matches_cannonball_formula() {
        let a = srp_acceleration(1.0, 0.02, ASTRONOMICAL_UNIT_KM);
        let expected_m_s2 = SOLAR_PRESSURE_1AU * 1.0 * 0.02;
        assert_abs_diff_eq!(a * 1000.0, expected_m_s2, epsilon = 1e-15);
    }

    #[test]
    fn third_body_conserves_kozai_integral_rate() {
        // theta = (1-e^2)cos^2(i) should be stationary under (i_dot, e_dot)
        // alone (dtheta/dt = 0), a strong self-consistency check on the
        // derived Lagrange-planetary-equation rates independent of overall
        // sign convention.
        let el = OrbitalElements::new(
            42_164.0,
            0.3,
            60.0_f64.to_radians(),
            0.0,
            30.0_f64.to_radians(),
            0.0,
            EARTH_MU,
        )
        .unwrap();
        let (_, _, i_dot, e_dot) = el.third_body_secular_rates(MOON_MU, MOON_DISTANCE_KM);
        let ci = el.i.cos();
        let si = el.i.sin();
        let theta_dot = -2.0 * el.e * e_dot * ci * ci - (1.0 - el.e * el.e) * 2.0 * ci * si * i_dot;
        assert_abs_diff_eq!(theta_dot, 0.0, epsilon = 1e-18);
    }

    #[test]
    fn third_body_rates_are_finite_and_propagation_preserves_a() {
        let el = OrbitalElements::new(
            42_164.0,
            0.1,
            20.0_f64.to_radians(),
            0.5,
            0.7,
            0.0,
            EARTH_MU,
        )
        .unwrap();
        let (raan_dot, argp_dot, i_dot, e_dot) =
            el.third_body_secular_rates(SUN_MU, ASTRONOMICAL_UNIT_KM);
        for v in [raan_dot, argp_dot, i_dot, e_dot] {
            assert!(v.is_finite());
        }
        let advanced = el
            .propagate_third_body(3600.0, SUN_MU, ASTRONOMICAL_UNIT_KM)
            .unwrap();
        assert_abs_diff_eq!(advanced.a, el.a, epsilon = 1e-9);
    }

    #[test]
    fn j4_rates_are_small_relative_correction_to_j2() {
        // LEO orbit: a = 7000 km, e = 0.01, i = 50 deg.
        let el = OrbitalElements::new(7000.0, 0.01, 50.0_f64.to_radians(), 0.0, 0.0, 0.0, EARTH_MU)
            .unwrap();
        let (raan_dot_j2, argp_dot_j2) = el.j2_secular_rates(EARTH_J2, EARTH_RADIUS_EQ);
        let (raan_dot_j24, argp_dot_j24) = el.j4_secular_rates(EARTH_J2, EARTH_J4, EARTH_RADIUS_EQ);

        // Both finite and the J4 correction is a small fraction of the J2
        // term (order (J4/J2)*(Req/p)^2 << 1), not a dominant contribution.
        assert!(raan_dot_j24.is_finite());
        assert!(argp_dot_j24.is_finite());
        let raan_j4_only = raan_dot_j24 - raan_dot_j2;
        let argp_j4_only = argp_dot_j24 - argp_dot_j2;
        assert!(raan_j4_only.abs() > 0.0);
        assert!(raan_j4_only.abs() < 0.05 * raan_dot_j2.abs());
        assert!(
            argp_j4_only.abs() < 0.05 * argp_dot_j2.abs().max(1e-30) || argp_dot_j2.abs() < 1e-30
        );
    }

    #[test]
    fn propagate_j4_matches_secular_rates() {
        let el = OrbitalElements::new(7000.0, 0.01, 50.0_f64.to_radians(), 1.0, 0.0, 0.0, EARTH_MU)
            .unwrap();
        let dt = 86_400.0;
        let advanced = el.propagate_j4(dt, EARTH_J2, EARTH_J4, EARTH_RADIUS_EQ);
        let (raan_dot, _) = el.j4_secular_rates(EARTH_J2, EARTH_J4, EARTH_RADIUS_EQ);
        let expected = (el.raan + raan_dot * dt).rem_euclid(2.0 * std::f64::consts::PI);
        assert_abs_diff_eq!(advanced.raan, expected, epsilon = 1e-9);
        assert_abs_diff_eq!(advanced.a, el.a, epsilon = 1e-9);
    }
}
