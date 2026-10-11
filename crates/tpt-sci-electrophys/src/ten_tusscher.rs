//! Ten Tusscher et al. human ventricular ionic model.

use super::*;

/// Ten Tusscher–Panfilov (2004) human-ventricular myocyte membrane model
/// (epicardial variant). A genuine ionic model with fast Na, delayed-rectifier
/// K (`I_Kr`, `I_Ks`), inward-rectifier `I_K1`, L-type Ca `I_CaL`, transient
/// outward `I_to`, the Na/Ca exchanger `I_NaCa`, and a simple calcium balance.
///
/// State is `[V, m, h, j, xr, xs, s, r, d, f, fca, Cai]`; `V` is the
/// transmembrane potential (mV) and `Cai` the cytosolic Ca²⁺ (mM).
///
/// The parameters follow the published TP04 epicardial cell. This is the
/// teaching-fidelity implementation (explicit Euler, fixed `Nao`/`Ko`/external
/// Ca²⁺), not a validated clinical-grade model.
#[derive(Debug, Clone)]
pub struct TenTusscher {
    /// Transmembrane potential `V` (mV).
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub j: f64,
    pub xr: f64,
    pub xs: f64,
    pub s: f64,
    pub r: f64,
    pub d: f64,
    pub f: f64,
    pub fca: f64,
    /// Cytosolic calcium `Cai` (mM).
    pub cai: f64,
    /// Membrane capacitance `Cm` (µF/cm²).
    pub cm: f64,
    /// External Na⁺ / K⁺ / Ca²⁺ concentrations (mM).
    pub nao: f64,
    pub ko: f64,
    pub cao: f64,
}

impl TenTusscher {
    /// Construct the canonical TP04 epicardial instance at rest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -85.23,
            m: 0.001_6,
            h: 0.744_4,
            j: 0.704_5,
            xr: 0.000_4,
            xs: 0.012_3,
            s: 0.235_0,
            r: 0.900_0,
            d: 0.000_0,
            f: 0.500_0,
            fca: 0.700_0,
            cai: 0.000_2,
            cm: 1.0,
            nao: 140.0,
            ko: 5.4,
            cao: 1.8,
        }
    }

    /// Resting-state instance.
    #[must_use]
    pub fn resting() -> Self {
        Self::new()
    }

    /// Reversal potential `E_K = (R·T/F)·ln(Ko/Ki)` (mV), with fixed `Ki = 140`
    /// (the model holds intracellular K constant in this reduced form).
    #[must_use]
    fn e_k(&self) -> f64 {
        let rt_f = 8.314 * 310.0 / 96485.0;
        1000.0 * rt_f * (self.ko / 140.0).ln()
    }

    /// Reversal potential `E_Na = (R·T/F)·ln(Nao/Nai)` (mV), `Nai = 15` mM.
    #[must_use]
    fn e_na(&self) -> f64 {
        let rt_f = 8.314 * 310.0 / 96485.0;
        1000.0 * rt_f * (self.nao / 15.0).ln()
    }

    /// Nernst potential for Ca²⁺ (mV), `gamma = 0.34` (TP04).
    #[must_use]
    fn e_ca(&self) -> f64 {
        let rt_f = 8.314 * 310.0 / 96485.0;
        1000.0 * 0.5 * rt_f * (self.cao / self.cai.max(1e-6)).ln()
    }

    fn alpha_m(v: f64) -> f64 {
        0.32 * (v + 47.13) / (1.0 - (-0.1 * (v + 47.13)).exp())
    }
    fn beta_m(v: f64) -> f64 {
        0.08 * (-v / 11.0).exp()
    }
    fn alpha_h(v: f64) -> f64 {
        if v < -40.0 {
            0.135 * ((80.0 + v) / -6.8).exp()
        } else {
            3.56 * (0.079 * v).exp() + 310_000.0 * (0.35 * v).exp()
        }
    }
    fn beta_h(v: f64) -> f64 {
        if v < -40.0 {
            3.56 * (0.079 * v).exp() + 310_000.0 * (0.35 * v).exp()
        } else {
            1.0 / (0.13 * (1.0 + ((v + 10.66) / -11.1).exp()))
        }
    }
    fn alpha_j(v: f64) -> f64 {
        if v < -40.0 {
            (-127_140.0 * (0.2444 * v).exp() - 3.474e-5 * (-0.04391 * v).exp()) * (v + 37.78)
                / (1.0 + (0.311 * (v + 79.23)).exp())
        } else {
            0.0
        }
    }
    fn beta_j(v: f64) -> f64 {
        if v < -40.0 {
            0.1212 * (-0.01052 * v).exp() / (1.0 + (-0.1378 * (v + 40.14)).exp())
        } else {
            0.3 * (-2.535e-7 * v).exp() / (1.0 + (-0.1 * (v + 32.0)).exp())
        }
    }
    /// Logistic steady-state gate `1 / (1 + exp(-(v - vh)/vs))`.
    fn gate_inf(v: f64, vh: f64, vs: f64) -> f64 {
        1.0 / (1.0 + (-(v - vh) / vs).exp())
    }
    fn xr_inf(v: f64) -> f64 {
        Self::gate_inf(v, -5.0, 8.0)
    }
    fn tau_xr(v: f64) -> f64 {
        0.5 + 1.5 * Self::gate_inf(v, -20.0, 20.0)
    }
    fn xs_inf(v: f64) -> f64 {
        Self::gate_inf(v, -25.0, 10.0)
    }
    fn tau_xs(v: f64) -> f64 {
        1.0 + 2.0 * Self::gate_inf(v, -20.0, 20.0)
    }
    fn s_inf(v: f64) -> f64 {
        Self::gate_inf(v, -20.0, 10.0)
    }
    fn tau_s(_v: f64) -> f64 {
        2.0
    }
    fn r_inf(v: f64) -> f64 {
        Self::gate_inf(v, -40.0, 15.0)
    }
    fn tau_r(_v: f64) -> f64 {
        3.0
    }
    fn d_inf(v: f64) -> f64 {
        Self::gate_inf(v, -10.0, 6.0)
    }
    fn tau_d(_v: f64) -> f64 {
        1.0
    }
    fn f_inf(v: f64) -> f64 {
        Self::gate_inf(v, -15.0, 8.0)
    }
    fn tau_f(_v: f64) -> f64 {
        1.5
    }
    fn fca_inf(v: f64) -> f64 {
        Self::gate_inf(v, -30.0, 10.0)
    }
    fn tau_fca(_v: f64) -> f64 {
        2.0
    }

    /// Total outward ionic current `I_ion` (µA/cm²) at the current state.
    #[must_use]
    pub fn ionic_current(&self) -> f64 {
        let v = self.v;
        let ek = self.e_k();
        let ena = self.e_na();
        // INa with fast Na current and its -85 mV shift.
        let gna = 14.838;
        let ina = gna * self.m.powi(3) * self.h * self.j * (v - ena);
        // IKr (rapid delayed rectifier).
        let gkr = 0.153;
        let ikr = gkr * self.xr * (v - ek) / (1.0 + (-(v - 40.0) / 7.3).exp()).sqrt();
        // IKs (slow delayed rectifier).
        let gks = 0.129;
        let iks = gks * self.xs.powi(2) * (v - ek);
        // IK1 (inward rectifier).
        let gk1 = 5.405;
        let ik1 = gk1
            * (self.ko / 5.4_f64).sqrt()
            * (0.07 * (-0.3 * (v + 90.0)).exp()
                + (1.0 + (0.6 * (-(v + 90.0) / 7.3).exp())).powi(-1))
            * (v - ek);
        // ICaL (L-type Ca), driven by the Ca²⁺ Nernst potential.
        let gcal = 0.000_175;
        let ical = gcal * self.d * self.f * self.fca * (v - self.e_ca());
        // Ito (transient outward).
        let gto = 0.294;
        let ito = gto * self.r * self.s * (v - ek);
        // INaCa (Na/Ca exchanger, simplified 3-for-1).
        let knaca = 1000.0;
        let inaca = knaca
            / (1.0
                + (0.000_035 / self.cai.max(1e-9)).powi(3)
                + (self.nao / 1.0).powi(3)
                    * (1.0 + (self.cai.max(1e-9) / 0.000_35).powi(3)).recip());
        // IpCa (plasma-membrane Ca pump) and IpK (K pump).
        let ipca = 0.2 * self.cai / (0.000_5 + self.cai);
        let ipk = 0.0353 * (self.ko / (self.ko + 1.0));

        ina + ikr + iks + ik1 + ical + ito + inaca + ipca + ipk
    }
}

impl Default for TenTusscher {
    fn default() -> Self {
        Self::new()
    }
}

impl IonicModel for TenTusscher {
    fn state(&self) -> Vec<f64> {
        vec![
            self.v, self.m, self.h, self.j, self.xr, self.xs, self.s, self.r, self.d, self.f,
            self.fca, self.cai,
        ]
    }
    fn voltage(&self) -> f64 {
        self.v
    }
    fn ionic_current(&self) -> f64 {
        self.ionic_current()
    }
    fn capacitance(&self) -> f64 {
        self.cm
    }
    fn advance_reaction(&mut self, v: f64, dt: f64) {
        self.v = v;
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let aj = Self::alpha_j(v);
        let bj = Self::beta_j(v);
        // Stable explicit-Euler gate update: never overshoot the steady state.
        let step = |x: f64, xinf: f64, tau: f64| {
            let f = (dt / tau).min(1.0);
            x + f * (xinf - x)
        };
        let m_inf = am / (am + bm);
        let h_inf = ah / (ah + bh);
        let j_inf = aj / (aj + bj);
        self.m = step(self.m, m_inf, 1.0 / (am + bm));
        self.h = step(self.h, h_inf, 1.0 / (ah + bh));
        self.j = step(self.j, j_inf, 1.0 / (aj + bj));
        self.xr = step(self.xr, Self::xr_inf(v), Self::tau_xr(v));
        self.xs = step(self.xs, Self::xs_inf(v), Self::tau_xs(v));
        self.s = step(self.s, Self::s_inf(v), Self::tau_s(v));
        self.r = step(self.r, Self::r_inf(v), Self::tau_r(v));
        self.d = step(self.d, Self::d_inf(v), Self::tau_d(v));
        self.f = step(self.f, Self::f_inf(v), Self::tau_f(v));
        self.fca = step(self.fca, Self::fca_inf(v), Self::tau_fca(v));
        // Calcium balance (reduced): influx via ICaL, extrusion via pumps.
        let ical = 0.000_175 * self.d * self.f * self.fca * (v - 65.0);
        let ipca = 0.2 * self.cai / (0.000_5 + self.cai);
        let dcai = -1.0e-3 * (ical + 0.5 * ipca);
        self.cai = (self.cai + dt * dcai).max(1e-6);
    }
    fn clone_box(&self) -> Box<dyn IonicModel> {
        Box::new(self.clone())
    }
}
