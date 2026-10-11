//! Land et al. active-tension model of human ventricular contraction, intact
//! myocardium at 37 °C, with length and velocity dependence.
//!
//! Ported from the authors' release (`Land2016.m`, in the CellML exposure
//! `BG_crossbridge_TRPN`, workspace 7fb, `intact` branch). Land et al. (2017),
//! "A model of cardiac contraction based on novel measurements of tension
//! development in human cardiomyocytes", PubMed 28392437.
//!
//! The model takes the sarcomere length ratio `λ` and its rate `dλ/dt` as inputs.
//! The release does not include a mechanics model that produces them: they are
//! prescribed by the caller, as in the release's own protocols.
//!
//! Units: calcium in µM, length as a ratio of resting length, time in ms, tension
//! in kPa.

/// Parameters from `Land2016.m`, `intact` branch.
const PERM50: f64 = 0.35;
const TRPN_N: f64 = 2.0;
const KOFF: f64 = 0.1;
const DR: f64 = 0.25;
const WFRAC: f64 = 0.5;
const CA50: f64 = 0.805; // µM
const TREF: f64 = 120.0; // kPa
const NPERM: f64 = 5.0;
const NU: f64 = 7.0;
const MU: f64 = 3.0;
const TOT_A: f64 = 25.0;
const KTM_UNBLOCK: f64 = 1.0;
const PHI: f64 = 2.23;
const GAMMA: f64 = 0.0085;
const GAMMA_WU: f64 = 0.615;
/// Length-dependence constants.
const BETA_0: f64 = 2.3;
const BETA_1: f64 = -2.4;
/// Passive element.
const PAR_K: f64 = 7.0;
const PASSIVE_B: f64 = 9.1;
const ETA_LONG: f64 = 200.0;
const ETA_SHORT: f64 = 20.0;
const PASSIVE_A: f64 = 2.1;

/// State: `[XS, XW, TRPN, TmBlocked, ζS, ζW, Cd]`, the strongly and weakly bound
/// cross-bridge fractions, troponin Ca²⁺ occupancy, the blocked tropomyosin
/// fraction, the two cross-bridge distortion states, and the passive-element state.
#[derive(Debug, Clone, PartialEq)]
pub struct LandActiveTension {
    /// Post-power-stroke (strongly bound) cross-bridge fraction.
    pub xs: f64,
    /// Pre-power-stroke (weakly bound) cross-bridge fraction.
    pub xw: f64,
    /// Troponin Ca²⁺ occupancy (fraction).
    pub trpn: f64,
    /// Tropomyosin-blocked fraction.
    pub tm_blocked: f64,
    /// Distortion of the strongly bound cross-bridges.
    pub zeta_s: f64,
    /// Distortion of the weakly bound cross-bridges.
    pub zeta_w: f64,
    /// Passive-element state.
    pub cd: f64,
}

/// Rate constants that depend only on the fixed parameters.
struct Rates {
    k_ws: f64,
    k_uw: f64,
    k_wu: f64,
    k_su: f64,
    cds: f64,
    cdw: f64,
    a: f64,
    ktm_block: f64,
}

impl Rates {
    fn new() -> Self {
        let k_ws = 0.004 * MU;
        let k_uw = 0.026 * NU;
        let k_wu = k_uw * (1.0 / WFRAC - 1.0) - k_ws;
        let k_su = k_ws * (1.0 / DR - 1.0) * WFRAC;
        let cdw = PHI * k_uw * (1.0 - DR) * (1.0 - WFRAC) / ((1.0 - DR) * WFRAC);
        let cds = PHI * k_ws * (1.0 - DR) * WFRAC / DR;
        let a = (0.25 * TOT_A) / ((1.0 - DR) * WFRAC + DR) * (DR / 0.25);
        // Tropomyosin-blocking rate, from the steady-state occupancy of the blocked
        // states at zero strain (`XSSS`, `XWSS`).
        let xsss = DR * 0.5;
        let xwss = (1.0 - DR) * WFRAC * 0.5;
        let ktm_block = KTM_UNBLOCK * PERM50.powf(NPERM) * 0.5 / (0.5 - xsss - xwss);
        Self {
            k_ws,
            k_uw,
            k_wu,
            k_su,
            cds,
            cdw,
            a,
            ktm_block,
        }
    }
}

/// Inputs held constant over an integration step.
#[derive(Debug, Clone, Copy)]
struct Inputs {
    cai: f64,
    lambda: f64,
    dlambda_dt: f64,
}

/// The length ratio as the model uses it: capped at 1.2.
fn clamp_lambda(lambda: f64) -> f64 {
    lambda.min(1.2)
}

impl LandActiveTension {
    /// Resting state at resting length (no calcium, all cross-bridges unattached,
    /// tropomyosin blocked, no distortion, no passive strain).
    #[must_use]
    pub fn new() -> Self {
        Self {
            xs: 0.0,
            xw: 0.0,
            trpn: 0.0,
            tm_blocked: 1.0,
            zeta_s: 0.0,
            zeta_w: 0.0,
            cd: 0.0,
        }
    }

    /// Active tension (kPa) at the current state and length ratio `lambda`.
    #[must_use]
    pub fn active_tension(&self, lambda: f64) -> f64 {
        let lam = clamp_lambda(lambda);
        let lfac = (1.0 + BETA_0 * (lam + lam.min(0.87) - 1.87)).max(0.0);
        lfac * (TREF / DR)
            * ((self.zeta_s + 1.0) * self.xs.max(0.0) + self.zeta_w * self.xw.max(0.0))
    }

    /// Passive tension (kPa) at the current state and length ratio `lambda`.
    #[must_use]
    pub fn passive_tension(&self, lambda: f64) -> f64 {
        let c = clamp_lambda(lambda) - 1.0;
        let eta = if c - self.cd < 0.0 {
            ETA_SHORT
        } else {
            ETA_LONG
        };
        let dcd = PAR_K * (c - self.cd) / eta;
        let fd = eta * dcd;
        let f1 = (PASSIVE_B * c).exp() - 1.0;
        PASSIVE_A * (f1 + fd)
    }

    /// Total tension (kPa): active plus passive.
    #[must_use]
    pub fn total_tension(&self, lambda: f64) -> f64 {
        self.active_tension(lambda) + self.passive_tension(lambda)
    }

    fn to_array(&self) -> [f64; 7] {
        [
            self.xs,
            self.xw,
            self.trpn,
            self.tm_blocked,
            self.zeta_s,
            self.zeta_w,
            self.cd,
        ]
    }

    fn from_array(y: [f64; 7]) -> Self {
        Self {
            xs: y[0],
            xw: y[1],
            trpn: y[2],
            tm_blocked: y[3],
            zeta_s: y[4],
            zeta_w: y[5],
            cd: y[6],
        }
    }

    /// Time derivatives of the state, as in `Land2016.m`.
    fn derivatives(&self, inp: Inputs, r: &Rates) -> [f64; 7] {
        let xs = self.xs.max(0.0);
        let xw = self.xw.max(0.0);
        let trpn = self.trpn.max(0.0);
        let lam = clamp_lambda(inp.lambda);
        let xu = (1.0 - self.tm_blocked) - xw - xs;

        let xb_ws = r.k_ws * xw;
        let xb_uw = r.k_uw * xu;
        let xb_wu = r.k_wu * xw;
        let xb_su = r.k_su * xs;

        // Strain-dependent detachment (strongly bound) and weak-state damping.
        let z = self.zeta_s;
        let gamma_rate = GAMMA
            * if z > 0.0 {
                z
            } else if z < -1.0 {
                -z - 1.0
            } else {
                0.0
            };
        let xb_su_gamma = gamma_rate * xs;
        let xb_wu_gamma = GAMMA_WU * self.zeta_w.abs() * xw;

        let d_xs = xb_ws - xb_su - xb_su_gamma;
        let d_xw = xb_uw - xb_wu - xb_ws - xb_wu_gamma;

        let ca50 = CA50 + BETA_1 * (lam - 1.0).min(0.2);
        let d_trpn = KOFF * ((inp.cai / ca50).powf(TRPN_N) * (1.0 - trpn) - trpn);

        // `min(100, TRPN^-(nperm/2))`: with no troponin occupancy the rate is capped.
        let block_pow = if trpn > 0.0 {
            trpn.powf(-NPERM / 2.0).min(100.0)
        } else {
            100.0
        };
        let unblock_pow = trpn.powf(NPERM / 2.0);
        let d_tm = r.ktm_block * block_pow * xu - KTM_UNBLOCK * unblock_pow * self.tm_blocked;

        let d_zeta_s = r.a * inp.dlambda_dt - r.cds * self.zeta_s;
        let d_zeta_w = r.a * inp.dlambda_dt - r.cdw * self.zeta_w;

        // Passive element, as in `passive_tension`.
        let c = lam - 1.0;
        let eta = if c - self.cd < 0.0 {
            ETA_SHORT
        } else {
            ETA_LONG
        };
        let d_cd = PAR_K * (c - self.cd) / eta;

        [d_xs, d_xw, d_trpn, d_tm, d_zeta_s, d_zeta_w, d_cd]
    }

    /// Advance by `dt` (ms) with fourth-order Runge–Kutta. The calcium `cai` (µM),
    /// length ratio `lambda`, and its rate `dlambda_dt` (1/ms) are held over the step.
    pub fn advance(&mut self, cai: f64, lambda: f64, dlambda_dt: f64, dt: f64) {
        let r = Rates::new();
        let inp = Inputs {
            cai,
            lambda,
            dlambda_dt,
        };
        let y = self.to_array();
        let shift = |base: [f64; 7], k: [f64; 7], s: f64| {
            let mut out = base;
            for i in 0..7 {
                out[i] += s * k[i];
            }
            out
        };
        let k1 = Self::from_array(y).derivatives(inp, &r);
        let k2 = Self::from_array(shift(y, k1, 0.5 * dt)).derivatives(inp, &r);
        let k3 = Self::from_array(shift(y, k2, 0.5 * dt)).derivatives(inp, &r);
        let k4 = Self::from_array(shift(y, k3, dt)).derivatives(inp, &r);
        let mut next = y;
        for i in 0..7 {
            next[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        *self = Self::from_array(next);
    }
}

impl Default for LandActiveTension {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_calcium_gives_no_tension() {
        let mut m = LandActiveTension::new();
        for _ in 0..10_000 {
            m.advance(0.0, 1.0, 0.0, 0.01);
        }
        assert!(
            m.total_tension(1.0).abs() < 1e-6,
            "tension {}",
            m.total_tension(1.0)
        );
    }

    #[test]
    fn saturating_calcium_develops_bounded_tension() {
        // Sustained high calcium saturates the troponin and the tension approaches
        // the reference scale `Tref` (120 kPa) from below.
        let mut m = LandActiveTension::new();
        for _ in 0..200_000 {
            m.advance(100.0, 1.0, 0.0, 0.01);
        }
        let t = m.total_tension(1.0);
        assert!(t > 20.0 && t < TREF, "steady tension {t} kPa");
    }

    #[test]
    fn tension_follows_a_calcium_pulse_with_a_lag() {
        // A calcium pulse peaking at 50 ms: active tension must peak later.
        let mut m = LandActiveTension::new();
        let (mut peak, mut peak_t) = (f64::MIN, 0.0);
        let dt = 0.01;
        for k in 0..30_000 {
            let t = k as f64 * dt;
            let cai = 0.1 + 0.8 * (-((t - 50.0) / 25.0).powi(2)).exp();
            m.advance(cai, 1.0, 0.0, dt);
            if m.active_tension(1.0) > peak {
                peak = m.active_tension(1.0);
                peak_t = t;
            }
        }
        assert!(peak > 1.0, "twitch too small: {peak} kPa");
        assert!(
            peak_t > 50.0,
            "tension should lag calcium: peak at {peak_t} ms"
        );
        assert!(m.xs.is_finite() && m.xw.is_finite() && m.trpn.is_finite());
    }

    #[test]
    fn shortening_reduces_tension_relative_to_isometric() {
        // Force–velocity: at the same calcium, shortening lowers the tension.
        let cai = |t: f64| 0.1 + 0.8 * (-((t - 50.0) / 25.0).powi(2)).exp();
        let run = |velocity: f64| {
            let mut m = LandActiveTension::new();
            let dt = 0.01;
            let mut lambda = 1.0;
            for k in 0..20_000 {
                let t = k as f64 * dt;
                let shortening = (100.0..300.0).contains(&t);
                let dl = if shortening { velocity } else { 0.0 };
                m.advance(cai(t), lambda, dl, dt);
                lambda += dl * dt;
            }
            m.total_tension(lambda)
        };
        let isometric = run(0.0);
        let shortening = run(-0.1 / 200.0);
        assert!(shortening.is_finite());
        assert!(
            shortening < isometric,
            "shortening {shortening} vs isometric {isometric}"
        );
    }
}
