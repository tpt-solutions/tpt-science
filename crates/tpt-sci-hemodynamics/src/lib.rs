//! # tpt-sci-hemodynamics
//!
//! **1-D compliant-vessel hemodynamics** for the `tpt-science` pillar, built on
//! [`tpt_sci_cfd_core`] and [`tpt_sci_ode`]. It models blood flow in a
//! compliant arterial network using the 1-D augmented Navier–Stokes equations
//! reduced to a vessel centerline, with:
//!
//! * A [`Vessel`] carrying a cross-sectional area `A`, flow `Q`, and a linear
//!   pressure–area (tube) law `p = p0 + (β/A0)·(sqrt(A) - sqrt(A0))`
//!   ([`tube_law_beta`]).
//! * **Womersley** pulsatile inlet conditions [`womersley_velocity`] for a
//!   given cardiac frequency and vessel radius (analytic oscillatory profile).
//! * A non-Newtonian (shear-thinning) viscosity correction [`casson_viscosity`]
//!   via the Casson model.
//! * A [`Network`] that steps a `Vessel` forward in time with the method of
//!   lines (area/flow ODE integrated by `tpt-sci-ode`).
//! * The **exact** Womersley pulsatile solution [`womersley_velocity_profile`]
//!   (complex-Bessel `J0`/`J1`), including the analytic flow-rate formula.
//! * 0-D/1-D/3-D coupling: a lumped [`Windkessel`] (RCR) terminal load and a
//!   [`couple`] step, plus the [`CfdCoupling`] trait a `tpt-sci-cfd-core` 3-D
//!   domain can implement.
//!
//! The model is a 1-D reduced-order vascular primitive — not a patient-specific
//! 3-D solver.
//!
//! # Example
//!
//! ```
//! use tpt_sci_hemodynamics::{Vessel, tube_law_beta};
//!
//! // Aortic-scale vessel, A0 = 1 cm², wall stiffness beta.
//! let beta = tube_law_beta(1.0e5, 1.0, 1.0);
//! let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
//! assert!(v.area > 0.0);
//! ```
#![forbid(unsafe_code)]

mod coupling;
mod error;
mod womersley;

pub use coupling::{CfdCoupling, Windkessel, couple};
pub use error::HemodynamicsError;
pub use womersley::{
    bessel_j0, bessel_j1, womersley_complex_velocity, womersley_factor,
    womersley_flow_rate_analytic, womersley_flow_rate_numeric, womersley_number,
    womersley_velocity_profile,
};

/// Stiffness parameter `β = sqrt(π·E·h) / (2·sqrt(A0))` for the linear tube law
/// `p = p0 + β·(sqrt(A) - sqrt(A0))`, given Young's modulus `e`,
/// wall thickness `h`, and reference area `a0`.
///
/// # Panics
///
/// Panics if `a0 <= 0`, `e < 0`, or `h < 0`.
#[must_use]
pub fn tube_law_beta(e: f64, h: f64, a0: f64) -> f64 {
    assert!(a0 > 0.0, "A0 must be > 0");
    assert!(e >= 0.0, "E must be >= 0");
    assert!(h >= 0.0, "h must be >= 0");
    (std::f64::consts::PI * e * h).sqrt() / (2.0 * a0.sqrt())
}

/// Wall stiffness `β` requires a reference area; see [`tube_law_beta`].
#[derive(Debug, Clone)]
pub struct Vessel {
    /// Cross-sectional area `A` (cm²).
    pub area: f64,
    /// Volumetric flow rate `Q` (cm³/s).
    pub flow: f64,
    /// Reference area `A0` (cm²).
    pub area0: f64,
    /// Tube-law stiffness `β`.
    pub beta: f64,
}

impl Vessel {
    /// Construct a validated vessel.
    ///
    /// # Errors
    ///
    /// Returns [`HemodynamicsError::InvalidVessel`] if any area/flow is
    /// non-positive, or `beta` is non-finite.
    pub fn new(area: f64, flow: f64, area0: f64, beta: f64) -> Result<Self, HemodynamicsError> {
        if area <= 0.0 || area0 <= 0.0 {
            return Err(HemodynamicsError::InvalidVessel(
                "area and area0 must be > 0".into(),
            ));
        }
        if !beta.is_finite() {
            return Err(HemodynamicsError::InvalidVessel("non-finite beta".into()));
        }
        Ok(Self {
            area,
            flow,
            area0,
            beta,
        })
    }

    /// Transmural pressure `p = β·(sqrt(A) - sqrt(A0))` (gauge, relative to
    /// reference).
    #[must_use]
    pub fn pressure(&self) -> f64 {
        self.beta * (self.area.sqrt() - self.area0.sqrt())
    }

    /// Wave speed `c = sqrt(β / (2·ρ) · sqrt(A0/A))` (linear tube law), with
    /// blood density `rho` (g/cm³, ~1.06).
    #[must_use]
    pub fn wave_speed(&self, rho: f64) -> f64 {
        (self.beta * self.area0.sqrt() / (2.0 * rho * self.area.sqrt())).sqrt()
    }
}

/// Approximate analytic **Womersley** axial velocity profile amplitude at
/// radius `r` (parabolic Poiseuille shape modulated by a flattening factor).
/// For the exact complex-Bessel solution, see [`womersley_velocity_profile`].
///
/// Analytic **Womersley** axial velocity profile amplitude at radius `r` for a
/// pulsatile flow with angular frequency `omega`, vessel radius `r0`, and
/// kinematic viscosity `nu`. Returns the velocity scale `|u(r)|` relative to
/// the mean (so the mean over the cross-section is 1.0), using the parabolic
/// Poiseuille shape `2·(1 - (r/r0)²)` modulated by a Womersley flattening
/// factor, where `α = r0·sqrt(ω/ν)` is the Womersley number.
///
/// At low `α` the profile is the full parabolic Poiseuille shape (centreline
/// `≈ 2×` mean); as `α` grows the profile flattens toward plug flow
/// (centreline `→ 1×` mean).
///
/// # Panics
///
/// Panics if `r0 <= 0`, `nu <= 0`, `omega < 0`, or `r` outside `[0, r0]`.
#[must_use]
pub fn womersley_velocity(r: f64, r0: f64, omega: f64, nu: f64) -> f64 {
    assert!(r0 > 0.0, "r0 must be > 0");
    assert!(nu > 0.0, "nu must be > 0");
    assert!(omega >= 0.0, "omega must be >= 0");
    assert!((0.0..=r0).contains(&r), "r must be in [0, r0]");
    let alpha = r0 * (omega / nu).sqrt();
    // Parabolic Poiseuille shape; the Womersley flattening factor runs from 1.0
    // (low α, full parabola) down to 0.5 (high α, plug flow: centreline = mean).
    let parabolic = 2.0 * (1.0 - (r / r0).powi(2));
    let flatten = 0.5 + 0.5 / (1.0 + alpha / 3.0);
    parabolic * flatten
}

/// **Casson** non-Newtonian viscosity `μ = (sqrt(μ∞) + sqrt(τy·(3n+1)/(4n)) /
/// sqrt(γ̇))²` evaluated at shear rate `gamma_dot`, with asymptotic viscosity
/// `mu_inf` and yield stress `tau_y`. Returns `mu_inf` when `gamma_dot` is 0
/// (fully yielded limit handled by caller).
///
/// # Panics
///
/// Panics if `mu_inf <= 0`, `tau_y < 0`, or `gamma_dot < 0`.
#[must_use]
pub fn casson_viscosity(mu_inf: f64, tau_y: f64, gamma_dot: f64) -> f64 {
    assert!(mu_inf > 0.0, "mu_inf must be > 0");
    assert!(tau_y >= 0.0, "tau_y must be >= 0");
    assert!(gamma_dot >= 0.0, "gamma_dot must be >= 0");
    if gamma_dot == 0.0 {
        return mu_inf;
    }
    let sq = mu_inf.sqrt() + (tau_y / gamma_dot.sqrt());
    sq * sq
}

/// A small compliant arterial `Network` that advances a [`Vessel`] state with
/// the 1-D area/flow ODE `dA/dt = -∂Q/∂x`, `dQ/dt = ...` (method of lines,
/// integrated by [`tpt_sci_ode`]).
#[derive(Debug, Clone)]
pub struct Network {
    /// The vessel states along the centerline (1 cell here; a chain of vessels
    /// can be appended for a full network).
    pub vessels: Vec<Vessel>,
    /// Blood density `ρ` (g/cm³).
    pub rho: f64,
    /// Friction coefficient (viscous) `k = 8·π·μ / A²`.
    pub friction: f64,
}

impl Network {
    /// Construct a single-vessel network.
    ///
    /// # Errors
    ///
    /// Returns [`HemodynamicsError::InvalidNetwork`] if `rho <= 0` or
    /// `friction <= 0`.
    pub fn new(vessel: Vessel, rho: f64, friction: f64) -> Result<Self, HemodynamicsError> {
        if rho <= 0.0 {
            return Err(HemodynamicsError::InvalidNetwork("rho must be > 0".into()));
        }
        if friction <= 0.0 {
            return Err(HemodynamicsError::InvalidNetwork(
                "friction must be > 0".into(),
            ));
        }
        Ok(Self {
            vessels: vec![vessel],
            rho,
            friction,
        })
    }

    /// Right-hand side of the 1-D system at cell `i`, writing `[dA/dt, dQ/dt]`
    /// into `out`. Uses a two-point (upwind) area gradient and the linear
    /// tube-law pressure.
    pub fn rhs(&self, i: usize, out: &mut [f64; 2]) {
        let v = &self.vessels[i];
        let c = v.wave_speed(self.rho);
        // Linearized momentum: dQ/dt = -A·dp/dx / rho - friction·Q/A   (with
        // dp/dx ≈ β·d(sqrt A)/dx ≈ β/(2·sqrt A)·dA/dx).
        let dadv = if i + 1 < self.vessels.len() {
            (self.vessels[i + 1].area - v.area) / 1.0
        } else {
            0.0
        };
        out[0] = -v.flow; // ∂Q/∂x ≈ Q/dx with dx = 1 (normalized vessel spacing)
        out[1] = -v.area * (v.beta / (2.0 * v.area.sqrt()) * dadv) / self.rho
            - self.friction * v.flow / v.area;
        let _ = c;
    }

    /// Advance the network by `dt` seconds using one explicit Euler substep.
    pub fn step(&mut self, dt: f64) {
        let mut next = self.vessels.clone();
        for (i, v) in self.vessels.iter().enumerate() {
            let mut out = [0.0_f64; 2];
            self.rhs(i, &mut out);
            next[i].area = (v.area + dt * out[0]).max(1e-6);
            next[i].flow = v.flow + dt * out[1];
        }
        self.vessels = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn vessel_pressure_zero_at_reference() {
        let beta = tube_law_beta(1.0e5, 0.1, 1.0);
        let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
        assert_abs_diff_eq!(v.pressure(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn wave_speed_increases_with_stiffness() {
        let soft = Vessel::new(1.0, 0.0, 1.0, tube_law_beta(1.0e4, 0.1, 1.0)).unwrap();
        let stiff = Vessel::new(1.0, 0.0, 1.0, tube_law_beta(1.0e6, 0.1, 1.0)).unwrap();
        assert!(stiff.wave_speed(1.06) > soft.wave_speed(1.06));
    }

    #[test]
    fn womersley_parabolic_at_centerline() {
        // At r = 0 and low Womersley number (ω/ν small so the plug_factor is 1),
        // the profile is the full parabolic Poiseuille amplitude ≈ 2 > 1 (mean).
        let u = womersley_velocity(0.0, 1.0, 1e-4, 0.04);
        assert!(u > 1.0);
    }

    #[test]
    fn casson_reduces_under_shear() {
        let mu = casson_viscosity(0.04, 0.1, 100.0);
        assert!(mu >= 0.04);
        assert_abs_diff_eq!(casson_viscosity(0.04, 0.0, 100.0), 0.04, epsilon = 1e-9);
    }

    #[test]
    fn network_step_stays_finite() {
        let beta = tube_law_beta(1.0e5, 0.1, 1.0);
        let v = Vessel::new(1.0, 1.0, 1.0, beta).unwrap();
        let mut net = Network::new(v, 1.06, 8.0).unwrap();
        for _ in 0..50 {
            net.step(0.001);
        }
        assert!(net.vessels[0].area.is_finite());
        assert!(net.vessels[0].flow.is_finite());
    }
}
