//! 0-D / 1-D / 3-D coupling interface for the hemodynamic network.
//!
//! This provides the *mechanism* only (no patient-specific 3-D meshing, which
//! stays out of scope per the repo-wide unstructured-FEM exclusion):
//!
//! * A lumped 3-element [`Windkessel`] (RCR) model that acts as a terminal load
//!   for a 1-D [`crate::Network`] outlet.
//! * A free [`couple`] step that drives a 1-D network outlet from (and back into)
//!   a [`Windkessel`].
//! * A documented [`CfdCoupling`] trait that a `tpt-sci-cfd-core` 3-D domain
//!   would implement to exchange boundary flow/pressure with the 1-D network.

use crate::{HemodynamicsError, Network, Vessel};

/// A 3-element RCR (Windkessel) lumped model: a characteristic resistance `r_c`
/// in series with a parallel compliance `c` and peripheral resistance `r_d`.
///
/// The capacitor pressure `p_c` obeys
/// `c·dp_c/dt = Q_in − (p_c − p_ven)/r_d`,
/// and the proximal pressure seen by the 1-D outlet is
/// `p_prox = p_c + r_c·Q_in`. After inflow stops (`Q_in = 0`) the pressure
/// decays as `p_c(t) = p_ven + (p_c0 − p_ven)·e^{−t/(r_d·c)}` — time constant
/// `τ = r_d·c`.
#[derive(Debug, Clone)]
pub struct Windkessel {
    /// Characteristic (proximal) resistance `R_c`.
    pub r_c: f64,
    /// Peripheral (distal) resistance `R_d`.
    pub r_d: f64,
    /// Vessel compliance `C`.
    pub c: f64,
    /// Venous (downstream) reference pressure `P_ven`.
    pub p_ven: f64,
    /// Capacitor (compliance) pressure `p_c`, the state variable.
    pub p_c: f64,
}

impl Windkessel {
    /// Construct a validated RCR Windkessel with initial capacitor pressure
    /// `p0`.
    ///
    /// # Errors
    ///
    /// Returns [`HemodynamicsError::InvalidWindkessel`] if any resistance or the
    /// compliance is non-positive.
    pub fn new(r_c: f64, r_d: f64, c: f64, p_ven: f64, p0: f64) -> Result<Self, HemodynamicsError> {
        if r_c <= 0.0 || r_d <= 0.0 || c <= 0.0 {
            return Err(HemodynamicsError::InvalidWindkessel(
                "resistances and compliance must be > 0".into(),
            ));
        }
        Ok(Self {
            r_c,
            r_d,
            c,
            p_ven,
            p_c: p0,
        })
    }

    /// Advance the capacitor pressure by `dt` seconds under a constant inflow
    /// `q_in`, using the exact solution of the linear ODE over `[t, t+dt]`.
    pub fn step(&mut self, q_in: f64, dt: f64) {
        let tau = self.r_d * self.c;
        let e = (-dt / tau).exp();
        self.p_c = self.p_ven + q_in * self.r_d * (1.0 - e) + (self.p_c - self.p_ven) * e;
    }

    /// Proximal pressure `p_prox = p_c + r_c·Q_in` for the given current inflow.
    #[must_use]
    pub fn proximal_pressure(&self, q_in: f64) -> f64 {
        self.p_c + self.r_c * q_in
    }

    /// Capacitor-pressure time constant `τ = r_d·c` (exponential decay rate once
    /// inflow stops).
    #[must_use]
    pub fn time_constant(&self) -> f64 {
        self.r_d * self.c
    }
}

/// Thin interface a `tpt-sci-cfd-core` 3-D domain implements so the 1-D
/// hemodynamic network can exchange boundary data with it. The 3-D domain
/// receives the 1-D outlet pressure and returns the boundary flow it imposes
/// back (the coupling mechanism; actual meshing/assembly is out of scope).
pub trait CfdCoupling {
    /// Advance the 3-D domain by `dt` driven by the 1-D outlet pressure
    /// `p_outlet`, returning the boundary flow it imposes back on the 1-D
    /// network.
    fn couple_step(&mut self, dt: f64, p_outlet: f64) -> f64;
    /// Current pressure the 3-D domain applies at the coupling boundary.
    fn boundary_pressure(&self) -> f64;
}

impl CfdCoupling for Windkessel {
    fn couple_step(&mut self, dt: f64, p_outlet: f64) -> f64 {
        let q = (p_outlet - self.p_ven) / (self.r_c + self.r_d);
        self.step(q, dt);
        q
    }
    fn boundary_pressure(&self) -> f64 {
        self.p_c
    }
}

/// Drive a 1-D [`Network`] outlet through a [`Windkessel`] terminal load for one
/// step `dt`. The last vessel's outflow enters the Windkessel; the resulting
/// proximal pressure is imposed back on the outlet vessel (through the linear
/// tube law) as a weak boundary condition. Returns the outlet pressure applied.
///
/// This is the 0-D ↔ 1-D coupling mechanism. The 3-D coupling reuses the same
/// outlet-pressure hand-off via [`CfdCoupling`].
pub fn couple(network: &mut Network, wk: &mut Windkessel, dt: f64) -> f64 {
    if network.vessels.is_empty() {
        return wk.p_c;
    }
    network.step(dt);
    let i = network.vessels.len().saturating_sub(1);
    let q_out = network.vessels[i].flow;
    wk.step(q_out, dt);
    let p_prox = wk.proximal_pressure(q_out);
    let v: &mut Vessel = &mut network.vessels[i];
    let target_area = (v.area0.sqrt() + p_prox / v.beta).powi(2);
    v.area = target_area.max(1e-6);
    p_prox
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Network, Vessel, tube_law_beta};
    use approx::assert_abs_diff_eq;

    #[test]
    fn windkessel_rejects_nonpositive_params() {
        assert!(Windkessel::new(0.0, 10.0, 0.1, 0.0, 1.0).is_err());
        assert!(Windkessel::new(1.0, 0.0, 0.1, 0.0, 1.0).is_err());
        assert!(Windkessel::new(1.0, 10.0, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn windkessel_exponential_decay() {
        // After inflow stops, p_c decays with τ = r_d·c as p_ven + (p0−p_ven)·e^{−t/τ}.
        let mut wk = Windkessel::new(1.0, 10.0, 0.1, 0.0, 100.0).unwrap();
        let tau = wk.time_constant();
        for _ in 0..10 {
            wk.step(5.0, 0.01);
        }
        let p0 = wk.p_c;
        let dt = 0.001;
        for _ in 0..1000 {
            wk.step(0.0, dt);
        }
        let t = 1000.0 * dt;
        let expected = wk.p_ven + (p0 - wk.p_ven) * (-t / tau).exp();
        assert_abs_diff_eq!(wk.p_c, expected, epsilon = 1e-9);
    }

    #[test]
    fn coupled_outlet_pressure_bounded() {
        // A 1-D network outlet driven by a pulsatile inlet stays bounded when
        // coupled to the Windkessel.
        let beta = tube_law_beta(1.0e5, 0.1, 1.0);
        let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
        let mut net = Network::new(v, 1.06, 8.0).unwrap();
        let mut wk = Windkessel::new(1.0, 10.0, 0.1, 0.0, 80.0).unwrap();
        let dt = 1e-3;
        let omega = 2.0 * std::f64::consts::PI;
        let mut max_p = 0.0_f64;
        for k in 0..2000 {
            let t = k as f64 * dt;
            net.vessels[0].flow = 1.0 + 0.8 * (omega * t).sin();
            let p = couple(&mut net, &mut wk, dt);
            assert!(p.is_finite(), "coupled outlet pressure must stay finite");
            max_p = max_p.max(p.abs());
        }
        assert!(
            max_p < 1e4,
            "coupled outlet pressure must stay bounded (max={max_p})"
        );
    }

    #[test]
    fn windkessel_implements_cfd_coupling() {
        let mut wk = Windkessel::new(1.0, 10.0, 0.1, 0.0, 80.0).unwrap();
        let q = wk.couple_step(0.001, 90.0);
        assert!(q.is_finite());
        assert!(wk.boundary_pressure().is_finite());
    }
}
