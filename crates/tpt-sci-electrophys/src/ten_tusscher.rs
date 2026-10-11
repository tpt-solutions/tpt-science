//! Ten Tusscher et al. human ventricular ionic model.

use super::*;

/// Gas constant (J/(mol·K)).
const R: f64 = 8.314;
/// Temperature (K).
const T: f64 = 310.0;
/// Faraday constant (C/mmol), as used by the CellML model.
const F: f64 = 96.485;
/// Faraday constant (C/mol), for the SI form of the GHK current.
const F_SI: f64 = 96_485.0;
/// `RT/F` (mV), the Nernst scale.
const RT_F: f64 = R * T / F;

/// Cell membrane capacitance (pF) and cell volume (µm³), epicardial values.
const CELL_CM_PF: f64 = 185.0;
const V_C_UM3: f64 = 16404.0;
/// Sarcoplasmic-reticulum volume (µm³).
const V_SR_UM3: f64 = 1094.0;

/// `1 µm³ = 1e-15 L`.
const V_C_L: f64 = V_C_UM3 * 1e-15;
/// Converts `pA` into the concentration rate `mM/ms` for a cell of volume `V_c`:
/// `dC/dt = -I·Cm/(z·V_c·F)`, with `I` in pA/pF, `Cm` in pF, and the 1e-3 for ms.
const NA_K_FLUX: f64 = 1e-3 * 1e-12 / (V_C_L * F);

/// Ten Tusscher–Panfilov (2004) human-ventricular myocyte membrane model,
/// epicardial variant.
///
/// The gating kinetics, conductances and calcium handling follow the published
/// CellML model (`tentusscher_noble_noble_panfilov_2004_epi`). It has the full
/// intracellular ion balance (`Na_i`, `K_i`) and calcium handling with the
/// sarcoplasmic reticulum (`Ca_i`, `Ca_SR`), with the external concentrations
/// held fixed. Currents are in pA/pF, which is numerically µA/cm² at unit
/// specific capacitance, so the tissue solver uses [`IonicModel::capacitance`] = 1.
///
/// State is `[V, m, h, j, Xr1, Xr2, Xs, r, s, d, f, fCa, g, Na_i, K_i, Ca_i, Ca_SR]`;
/// `V` is in mV, concentrations in mM, and time in ms.
#[derive(Debug, Clone)]
pub struct TenTusscher {
    /// Transmembrane potential `V` (mV).
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub j: f64,
    pub xr1: f64,
    pub xr2: f64,
    pub xs: f64,
    pub r: f64,
    pub s: f64,
    pub d: f64,
    pub f: f64,
    pub fca: f64,
    /// SR calcium-release inactivation gate.
    pub g: f64,
    /// Intracellular Na⁺ and K⁺ (mM).
    pub nai: f64,
    pub ki: f64,
    /// Cytosolic and sarcoplasmic-reticulum calcium (mM).
    pub cai: f64,
    pub ca_sr: f64,
    /// External Na⁺ / K⁺ / Ca²⁺ concentrations (mM).
    pub nao: f64,
    pub ko: f64,
    pub cao: f64,
}

/// Steady-state logistic gate `1 / (1 + exp((v - vh)/k))` with slope `k`.
fn logistic(v: f64, vh: f64, k: f64) -> f64 {
    1.0 / (1.0 + ((v - vh) / k).exp())
}

impl TenTusscher {
    /// Construct the canonical epicardial instance at rest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -86.2,
            m: 0.0,
            h: 0.75,
            j: 0.75,
            xr1: 0.0,
            xr2: 1.0,
            xs: 0.0,
            r: 0.0,
            s: 1.0,
            d: 0.0,
            f: 1.0,
            fca: 1.0,
            g: 1.0,
            nai: 11.6,
            ki: 138.3,
            cai: 0.0002,
            ca_sr: 0.2,
            nao: 140.0,
            ko: 5.4,
            cao: 2.0,
        }
    }

    /// Resting-state instance.
    #[must_use]
    pub fn resting() -> Self {
        Self::new()
    }

    /// Nernst potentials (mV).
    fn e_na(&self) -> f64 {
        RT_F * (self.nao / self.nai).ln()
    }
    fn e_k(&self) -> f64 {
        RT_F * (self.ko / self.ki).ln()
    }
    fn e_ks(&self) -> f64 {
        const P_KNA: f64 = 0.03;
        RT_F * ((self.ko + P_KNA * self.nao) / (self.ki + P_KNA * self.nai)).ln()
    }
    fn e_ca(&self) -> f64 {
        0.5 * RT_F * (self.cao / self.cai).ln()
    }

    /// Time-constant and steady-state helpers, straight from the CellML gates.
    fn xr1_inf(v: f64) -> f64 {
        logistic(v, -26.0, -7.0)
    }
    fn tau_xr1(v: f64) -> f64 {
        let alpha = 450.0 / (1.0 + ((-45.0 - v) / 10.0).exp());
        let beta = 6.0 / (1.0 + ((v + 30.0) / 11.5).exp());
        alpha * beta
    }
    fn xr2_inf(v: f64) -> f64 {
        logistic(v, -88.0, 24.0)
    }
    fn tau_xr2(v: f64) -> f64 {
        let alpha = 3.0 / (1.0 + ((-60.0 - v) / 20.0).exp());
        let beta = 1.12 / (1.0 + ((v - 60.0) / 20.0).exp());
        alpha * beta
    }
    fn xs_inf(v: f64) -> f64 {
        logistic(v, -5.0, -14.0)
    }
    fn tau_xs(v: f64) -> f64 {
        let alpha = 1100.0 / (1.0 + ((-10.0 - v) / 6.0).exp()).sqrt();
        let beta = 1.0 / (1.0 + ((v - 60.0) / 20.0).exp());
        alpha * beta
    }

    fn m_inf(v: f64) -> f64 {
        1.0 / (1.0 + ((-56.86 - v) / 9.03).exp()).powi(2)
    }
    fn tau_m(v: f64) -> f64 {
        let alpha = 1.0 / (1.0 + ((-60.0 - v) / 5.0).exp());
        let beta =
            0.1 / (1.0 + ((v + 35.0) / 5.0).exp()) + 0.1 / (1.0 + ((v - 50.0) / 200.0).exp());
        alpha * beta
    }
    fn h_inf(v: f64) -> f64 {
        1.0 / (1.0 + ((v + 71.55) / 7.43).exp()).powi(2)
    }
    /// `(alpha_h, beta_h)` for the fast Na inactivation gate.
    fn h_rates(v: f64) -> (f64, f64) {
        if v < -40.0 {
            (
                0.057 * (-(v + 80.0) / 6.8).exp(),
                2.7 * (0.079 * v).exp() + 310_000.0 * (0.3485 * v).exp(),
            )
        } else {
            (0.0, 0.77 / (0.13 * (1.0 + ((v + 10.66) / -11.1).exp())))
        }
    }
    /// `(alpha_j, beta_j)` for the slow Na inactivation gate.
    fn j_rates(v: f64) -> (f64, f64) {
        if v < -40.0 {
            (
                (-25_428.0 * (0.2444 * v).exp() - 6.948e-6 * (-0.04391 * v).exp()) * (v + 37.78)
                    / (1.0 + (0.311 * (v + 79.23)).exp()),
                0.02424 * (-0.01052 * v).exp() / (1.0 + (-0.1378 * (v + 40.14)).exp()),
            )
        } else {
            (
                0.0,
                0.6 * (0.057 * v).exp() / (1.0 + (-0.1 * (v + 32.0)).exp()),
            )
        }
    }

    fn d_inf(v: f64) -> f64 {
        logistic(v, -5.0, -7.5)
    }
    fn tau_d(v: f64) -> f64 {
        let alpha = 1.4 / (1.0 + ((-35.0 - v) / 13.0).exp()) + 0.25;
        let beta = 1.4 / (1.0 + ((v + 5.0) / 5.0).exp());
        let gamma = 1.0 / (1.0 + ((50.0 - v) / 20.0).exp());
        alpha * beta + gamma
    }
    fn f_inf(v: f64) -> f64 {
        logistic(v, -20.0, 7.0)
    }
    fn tau_f(v: f64) -> f64 {
        1125.0 * (-(v + 27.0).powi(2) / 240.0).exp()
            + 80.0
            + 165.0 / (1.0 + ((25.0 - v) / 10.0).exp())
    }
    fn fca_inf(cai: f64) -> f64 {
        let alpha = 1.0 / (1.0 + (cai / 0.000_325).powi(8));
        let beta = 0.1 / (1.0 + ((cai - 0.0005) / 0.0001).exp());
        let gamma = 0.2 / (1.0 + ((cai - 0.000_75) / 0.0008).exp());
        (alpha + beta + gamma + 0.23) / 1.46
    }

    fn s_inf(v: f64) -> f64 {
        logistic(v, -20.0, 5.0)
    }
    fn tau_s(v: f64) -> f64 {
        85.0 * (-(v + 45.0).powi(2) / 320.0).exp() + 5.0 / (1.0 + ((v - 20.0) / 5.0).exp()) + 3.0
    }
    fn r_inf(v: f64) -> f64 {
        logistic(v, 20.0, -6.0)
    }
    fn tau_r(v: f64) -> f64 {
        9.5 * (-(v + 40.0).powi(2) / 1800.0).exp() + 0.8
    }

    /// Current conductances and the `Ca_i` dependent SR-release factor
    /// `g_inf` (CellML `g_inf`).
    fn g_inf(cai: f64) -> f64 {
        if cai < 0.000_35 {
            1.0 / (1.0 + (cai / 0.000_35).powi(6))
        } else {
            1.0 / (1.0 + (cai / 0.000_35).powi(16))
        }
    }

    /// Individual ionic currents (pA/pF, outward positive).
    fn currents(&self) -> Currents {
        let v = self.v;
        let e_na = self.e_na();
        let e_k = self.e_k();
        let e_ca = self.e_ca();

        let alpha_k1 = 0.1 / (1.0 + (0.06 * (v - e_k - 200.0)).exp());
        let beta_k1 = (3.0 * (0.0002 * (v - e_k + 100.0)).exp() + (0.1 * (v - e_k - 10.0)).exp())
            / (1.0 + (-0.5 * (v - e_k)).exp());
        let k1_inf = alpha_k1 / (alpha_k1 + beta_k1);
        let ik1 = 5.405 * k1_inf * (self.ko / 5.4).sqrt() * (v - e_k);

        let ikr = 0.096 * (self.ko / 5.4).sqrt() * self.xr1 * self.xr2 * (v - e_k);
        let iks = 0.245 * self.xs.powi(2) * (v - self.e_ks());
        let ina = 14.838 * self.m.powi(3) * self.h * self.j * (v - e_na);
        let ib_na = 0.000_29 * (v - e_na);

        // L-type Ca current, GHK form. `V/expm1(x)` is taken as its limit at x = 0.
        let x = 2.0 * v * F / (R * T);
        let v_over_expm1 = if x.abs() < 1e-9 {
            RT_F / 2.0
        } else {
            v / x.exp_m1()
        };
        let ical = 0.175e-3
            * self.d
            * self.f
            * self.fca
            * 4.0
            * (F_SI.powi(2) / (R * T))
            * 1e-3
            * v_over_expm1
            * (self.cai * x.exp() - 0.341 * self.cao);
        let ib_ca = 0.000_592 * (v - e_ca);

        let ito = 0.294 * self.r * self.s * (v - e_k);

        let ina_k = (1.362 * self.ko / (self.ko + 1.0) * self.nai / (self.nai + 40.0))
            / (1.0 + 0.1245 * (-0.1 * v * F / (R * T)).exp() + 0.0353 * (-v * F / (R * T)).exp());

        let gamma = 0.35;
        let alpha = 2.5;
        let inaca = 1000.0
            * ((gamma * v * F / (R * T)).exp() * self.nai.powi(3) * self.cao
                - ((gamma - 1.0) * v * F / (R * T)).exp() * self.nao.powi(3) * self.cai * alpha)
            / ((87.5_f64.powi(3) + self.nao.powi(3))
                * (1.38 + self.cao)
                * (1.0 + 0.1 * ((gamma - 1.0) * v * F / (R * T)).exp()));

        let ipca = 0.825 * self.cai / (self.cai + 0.0005);
        let ipk = 0.0146 * (v - e_k) / (1.0 + ((25.0 - v) / 5.98).exp());

        Currents {
            ik1,
            ikr,
            iks,
            ina,
            ib_na,
            ical,
            ib_ca,
            ito,
            ina_k,
            inaca,
            ipca,
            ipk,
        }
    }

    /// Total outward ionic current `I_ion` (pA/pF, numerically µA/cm²) at the
    /// current state.
    #[must_use]
    pub fn ionic_current(&self) -> f64 {
        self.currents().total()
    }
}

/// The individual ionic currents at a given state (pA/pF, outward positive).
struct Currents {
    ik1: f64,
    ikr: f64,
    iks: f64,
    ina: f64,
    ib_na: f64,
    ical: f64,
    ib_ca: f64,
    ito: f64,
    ina_k: f64,
    inaca: f64,
    ipca: f64,
    ipk: f64,
}

impl Currents {
    /// Sum of all currents: `I_ion` (pA/pF).
    fn total(&self) -> f64 {
        self.ik1
            + self.ikr
            + self.iks
            + self.ina
            + self.ib_na
            + self.ical
            + self.ib_ca
            + self.ito
            + self.ina_k
            + self.inaca
            + self.ipca
            + self.ipk
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
            self.v, self.m, self.h, self.j, self.xr1, self.xr2, self.xs, self.r, self.s, self.d,
            self.f, self.fca, self.g, self.nai, self.ki, self.cai, self.ca_sr,
        ]
    }
    fn voltage(&self) -> f64 {
        self.v
    }
    fn ionic_current(&self) -> f64 {
        self.ionic_current()
    }
    /// Specific capacitance: currents are per unit capacitance, so this is 1.
    fn capacitance(&self) -> f64 {
        1.0
    }
    fn advance_reaction(&mut self, v: f64, dt: f64) {
        self.v = v;
        let c = self.currents();

        // Gate update, clamped so the explicit step never overshoots steady state.
        let step = |x: f64, xinf: f64, tau: f64| x + (dt / tau).min(1.0) * (xinf - x);

        let (ah, bh) = Self::h_rates(v);
        let (aj, bj) = Self::j_rates(v);
        let new_m = step(self.m, Self::m_inf(v), Self::tau_m(v));
        let new_h = step(self.h, Self::h_inf(v), 1.0 / (ah + bh));
        let new_j = step(self.j, Self::h_inf(v), 1.0 / (aj + bj));
        let new_xr1 = step(self.xr1, Self::xr1_inf(v), Self::tau_xr1(v));
        let new_xr2 = step(self.xr2, Self::xr2_inf(v), Self::tau_xr2(v));
        let new_xs = step(self.xs, Self::xs_inf(v), Self::tau_xs(v));
        let new_r = step(self.r, Self::r_inf(v), Self::tau_r(v));
        let new_s = step(self.s, Self::s_inf(v), Self::tau_s(v));
        let new_d = step(self.d, Self::d_inf(v), Self::tau_d(v));
        let new_f = step(self.f, Self::f_inf(v), Self::tau_f(v));
        // fCa and g do not recover while V is above -60 mV (CellML guard).
        let hold = |x: f64, xinf: f64| v > -60.0 && xinf > x;
        let new_fca = if hold(self.fca, Self::fca_inf(self.cai)) {
            self.fca
        } else {
            step(self.fca, Self::fca_inf(self.cai), 2.0)
        };
        let new_g = if hold(self.g, Self::g_inf(self.cai)) {
            self.g
        } else {
            step(self.g, Self::g_inf(self.cai), 2.0)
        };

        // Ion balance. Currents are pA/pF; multiplying by `CELL_CM_PF` gives pA.
        let ca_total_flux =
            -((c.ical + c.ib_ca + c.ipca) - 2.0 * c.inaca) * CELL_CM_PF * NA_K_FLUX / 2.0;
        let i_leak = 8e-5 * (self.ca_sr - self.cai);
        let i_up = 0.000_425 / (1.0 + (0.000_25 / self.cai).powi(2));
        let i_rel = ((0.016_464 * self.ca_sr.powi(2) / (0.25_f64.powi(2) + self.ca_sr.powi(2)))
            + 0.008_232)
            * self.d
            * self.g;
        let buf_c = 1.0 / (1.0 + (0.15 * 0.001) / (self.cai + 0.001).powi(2));
        let buf_sr = 1.0 / (1.0 + (10.0 * 0.3) / (self.ca_sr + 0.3).powi(2));
        let ca_flux = (ca_total_flux + i_leak - i_up + i_rel) * buf_c;
        let ca_sr_flux = (V_C_UM3 / V_SR_UM3) * (i_up - (i_rel + i_leak)) * buf_sr;

        let na_flux = -(c.ina + c.ib_na + 3.0 * c.ina_k + 3.0 * c.inaca) * CELL_CM_PF * NA_K_FLUX;
        let k_flux =
            -(c.ik1 + c.ito + c.ikr + c.iks + c.ipk - 2.0 * c.ina_k) * CELL_CM_PF * NA_K_FLUX;

        self.m = new_m;
        self.h = new_h;
        self.j = new_j;
        self.xr1 = new_xr1;
        self.xr2 = new_xr2;
        self.xs = new_xs;
        self.r = new_r;
        self.s = new_s;
        self.d = new_d;
        self.f = new_f;
        self.fca = new_fca;
        self.g = new_g;
        self.nai = (self.nai + dt * na_flux).max(1e-6);
        self.ki = (self.ki + dt * k_flux).max(1e-6);
        self.cai = (self.cai + dt * ca_flux).max(1e-7);
        self.ca_sr = (self.ca_sr + dt * ca_sr_flux).max(1e-6);
    }
    fn clone_box(&self) -> Box<dyn IonicModel> {
        Box::new(self.clone())
    }
}
