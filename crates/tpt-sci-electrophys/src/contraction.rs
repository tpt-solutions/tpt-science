//! Land et al. active-tension model of human ventricular contraction, intact
//! myocardium at 37 °C, at fixed resting length (isometric).
//!
//! Ported from the authors' release (`Land2016.m`, in the CellML exposure
//! `BG_crossbridge_TRPN`, workspace 7fb, `intact` branch). Land et al. (2017),
//! "A model of cardiac contraction based on novel measurements of tension
//! development in human cardiomyocytes", PubMed 28392437.
//!
//! The release also models length and velocity dependence (distortion states
//! `ζ` and a passive element). At a fixed length (`λ = 1`, `dλ/dt = 0`) the
//! distortion states stay at zero, so the gamma terms vanish, `Lfac = 1`, the
//! passive tension is zero, and the active tension reduces to `Ta = (Tref/dr)·XS`.
//! The four remaining states are the ones kept here.
//!
//! Units: calcium in µM, time in ms, tension in kPa.

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

/// State: `[XS, XW, TRPN, TmBlocked]`, the post- and pre-power-stroke cross-bridge
/// fractions, troponin Ca²⁺ occupancy, and the blocked tropomyosin fraction.
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
}

/// Rate constants that depend only on the fixed parameters.
struct Rates {
    k_ws: f64,
    k_uw: f64,
    k_wu: f64,
    k_su: f64,
    ktm_block: f64,
}

impl Rates {
    fn new() -> Self {
        let k_ws = 0.004 * MU;
        let k_uw = 0.026 * NU;
        let k_wu = k_uw * (1.0 / WFRAC - 1.0) - k_ws;
        let k_su = k_ws * (1.0 / DR - 1.0) * WFRAC;
        // Tropomyosin-blocking rate, from the steady-state occupancy of the blocked
        // states at zero strain (`XSSS`, `XWSS`).
        let xsss = DR * 0.5;
        let xwss = (1.0 - DR) * WFRAC * 0.5;
        let ktm_block = PERM50.powf(NPERM) * 0.5 / (0.5 - xsss - xwss);
        Self {
            k_ws,
            k_uw,
            k_wu,
            k_su,
            ktm_block,
        }
    }
}

impl LandActiveTension {
    /// Resting state (no calcium, all cross-bridges unattached, tropomyosin blocked).
    #[must_use]
    pub fn new() -> Self {
        Self {
            xs: 0.0,
            xw: 0.0,
            trpn: 0.0,
            tm_blocked: 1.0,
        }
    }

    /// Active tension (kPa) at the current state.
    #[must_use]
    pub fn tension(&self) -> f64 {
        (TREF / DR) * self.xs.max(0.0)
    }

    /// Time derivatives of `[XS, XW, TRPN, TmBlocked]` at calcium `cai` (µM).
    fn derivatives(&self, cai: f64, r: &Rates) -> [f64; 4] {
        let xs = self.xs.max(0.0);
        let xw = self.xw.max(0.0);
        let trpn = self.trpn.max(0.0);
        let xu = (1.0 - self.tm_blocked) - xw - xs;

        let xb_ws = r.k_ws * xw;
        let xb_uw = r.k_uw * xu;
        let xb_wu = r.k_wu * xw;
        let xb_su = r.k_su * xs;

        let d_xs = xb_ws - xb_su;
        let d_xw = xb_uw - xb_wu - xb_ws;
        let d_trpn = KOFF * ((cai / CA50).powf(TRPN_N) * (1.0 - trpn) - trpn);
        // `min(100, ·)` caps the blocking rate when troponin occupancy is near zero.
        let block = r.ktm_block * (trpn.powf(-NPERM / 2.0)).min(100.0) * xu;
        let unblock = trpn.powf(NPERM / 2.0) * self.tm_blocked;
        let d_tm = block - unblock;
        [d_xs, d_xw, d_trpn, d_tm]
    }

    fn to_array(&self) -> [f64; 4] {
        [self.xs, self.xw, self.trpn, self.tm_blocked]
    }

    fn from_array(y: [f64; 4]) -> Self {
        Self {
            xs: y[0],
            xw: y[1],
            trpn: y[2],
            tm_blocked: y[3],
        }
    }

    /// Advance by `dt` (ms) with fourth-order Runge–Kutta, holding the calcium
    /// `cai` (µM) fixed over the step.
    pub fn advance(&mut self, cai: f64, dt: f64) {
        let r = Rates::new();
        let y = self.to_array();
        let add = |a: [f64; 4], b: [f64; 4], s: f64| {
            [
                a[0] + s * b[0],
                a[1] + s * b[1],
                a[2] + s * b[2],
                a[3] + s * b[3],
            ]
        };
        let k1 = Self::from_array(y).derivatives(cai, &r);
        let k2 = Self::from_array(add(y, k1, 0.5 * dt)).derivatives(cai, &r);
        let k3 = Self::from_array(add(y, k2, 0.5 * dt)).derivatives(cai, &r);
        let k4 = Self::from_array(add(y, k3, dt)).derivatives(cai, &r);
        let mut next = y;
        for i in 0..4 {
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
            m.advance(0.0, 0.01);
        }
        assert!(m.tension().abs() < 1e-6, "tension {}", m.tension());
    }

    #[test]
    fn saturating_calcium_develops_bounded_tension() {
        // Sustained high calcium saturates the troponin and the tension approaches
        // the reference scale `Tref` (120 kPa) from below.
        let mut m = LandActiveTension::new();
        for _ in 0..200_000 {
            m.advance(100.0, 0.01);
        }
        let t = m.tension();
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
            m.advance(cai, dt);
            if m.tension() > peak {
                peak = m.tension();
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
}
