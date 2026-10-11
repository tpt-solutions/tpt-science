//! Hodgkin–Huxley (1952) squid-axon ionic model.

use super::*;

/// Classic Hodgkin–Huxley giant-axon membrane model.
///
/// State is `[V, m, h, n]` (membrane potential in mV, three activation/
/// inactivation gating variables in `[0, 1]`).
#[derive(Debug, Clone)]
pub struct HodgkinHuxley {
    /// Membrane potential `V` (mV).
    pub v: f64,
    /// Na activation `m`.
    pub m: f64,
    /// Na inactivation `h`.
    pub h: f64,
    /// K activation `n`.
    pub n: f64,
    /// Membrane capacitance `Cm` (µF/cm²).
    pub cm: f64,
    /// Max Na conductance `ḡNa` (mS/cm²).
    pub g_na: f64,
    /// Max K conductance `ḡK` (mS/cm²).
    pub g_k: f64,
    /// Leak conductance `ḡL` (mS/cm²).
    pub g_l: f64,
    /// Na reversal `E_Na` (mV).
    pub e_na: f64,
    /// K reversal `E_K` (mV).
    pub e_k: f64,
    /// Leak reversal `E_L` (mV).
    pub e_l: f64,
}

impl HodgkinHuxley {
    /// Construct a model with the canonical HH (squid axon) parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.0529,
            h: 0.5961,
            n: 0.3177,
            cm: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
        }
    }

    /// Resting-state instance (canonical initial gating values).
    #[must_use]
    pub fn resting() -> Self {
        Self::new()
    }

    /// Borrow the full state vector `[V, m, h, n]`.
    #[must_use]
    pub fn state(&self) -> Vec<f64> {
        vec![self.v, self.m, self.h, self.n]
    }

    /// Current membrane potential.
    #[must_use]
    pub fn voltage(&self) -> f64 {
        self.v
    }

    /// Total ionic current `I_ion` (µA/cm²), outward positive.
    #[must_use]
    pub fn ionic_current(&self) -> f64 {
        self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na)
            + self.g_k * self.n.powi(4) * (self.v - self.e_k)
            + self.g_l * (self.v - self.e_l)
    }

    /// Voltage-dependent rate `αm(V)` etc. (standard HH forms).
    #[must_use]
    fn alpha_m(v: f64) -> f64 {
        let x = v + 40.0;
        if x.abs() < 1e-6 {
            1.0
        } else {
            0.1 * x / (1.0 - (-x / 10.0).exp())
        }
    }

    #[must_use]
    fn beta_m(v: f64) -> f64 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    #[must_use]
    fn alpha_h(v: f64) -> f64 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    #[must_use]
    fn beta_h(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    #[must_use]
    fn alpha_n(v: f64) -> f64 {
        let x = v + 55.0;
        if x.abs() < 1e-6 {
            0.1
        } else {
            0.01 * x / (1.0 - (-x / 10.0).exp())
        }
    }

    #[must_use]
    fn beta_n(v: f64) -> f64 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }

    /// Advance the membrane ODE by `dt` seconds with one explicit Euler step
    /// (voltage and gating together — for standalone single-cell use).
    pub fn step(&mut self, dt: f64) {
        let am = Self::alpha_m(self.v);
        let bm = Self::beta_m(self.v);
        let ah = Self::alpha_h(self.v);
        let bh = Self::beta_h(self.v);
        let an = Self::alpha_n(self.v);
        let bn = Self::beta_n(self.v);

        let i_ion = self.ionic_current();
        let dv = -i_ion / self.cm;
        let dm = am * (1.0 - self.m) - bm * self.m;
        let dh = ah * (1.0 - self.h) - bh * self.h;
        let dn = an * (1.0 - self.n) - bn * self.n;

        self.v += dt * dv;
        self.m += dt * dm;
        self.h += dt * dh;
        self.n += dt * dn;
    }
}

impl Default for HodgkinHuxley {
    fn default() -> Self {
        Self::new()
    }
}

impl IonicModel for HodgkinHuxley {
    fn state(&self) -> Vec<f64> {
        self.state()
    }
    fn voltage(&self) -> f64 {
        self.voltage()
    }
    fn ionic_current(&self) -> f64 {
        self.ionic_current()
    }
    fn capacitance(&self) -> f64 {
        self.cm
    }
    fn advance_reaction(&mut self, v: f64, dt: f64) {
        // Hold the transmembrane voltage at `v`; evolve only the gating
        // variables at this potential (the voltage is integrated by the
        // tissue solver together with the diffusion term).
        self.v = v;
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);
        self.m += dt * (am * (1.0 - self.m) - bm * self.m);
        self.h += dt * (ah * (1.0 - self.h) - bh * self.h);
        self.n += dt * (an * (1.0 - self.n) - bn * self.n);
    }
    fn clone_box(&self) -> Box<dyn IonicModel> {
        Box::new(self.clone())
    }
}
