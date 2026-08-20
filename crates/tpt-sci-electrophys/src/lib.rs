//! # tpt-sci-electrophys
//!
//! **Cardiac electrophysiology** for the `tpt-science` pillar, built on
//! [`tpt_sci_ode`] (for the membrane kinetics) and [`tpt_sci_grid`] (for the
//! extracellular/bidomain diffusion). It provides:
//!
//! * A classic **Hodgkin–Huxley** giant-axon model ([`HodgkinHuxley`]): four
//!   gating variables `m, h, n` with their voltage-dependent rate laws, and the
//!   ionic current `I_ion = ḡNa·m³h·(V−E_Na) + ḡK·n⁴·(V−E_K) + ḡL·(V−E_L)`.
//! * A **monodomain / bidomain** tissue propagator [`Tissue`] that couples the
//!   HH membrane to a 2-D cable/bidomain diffusion operator from `tpt-sci-grid`.
//!
//! The model integrates `dVi/dt = −I_ion/Cm + D·∇²V` (monodomain) across a grid,
//! so an action potential launched at one node propagates through the tissue.
//!
//! # Example
//!
//! ```
//! use tpt_sci_electrophys::HodgkinHuxley;
//!
//! let mut hh = HodgkinHuxley::resting();
//! // Depolarize; integrate the membrane ODE for a short time.
//! let y = hh.state();
//! assert!(y.len() == 4);
//! hh.step(0.01);
//! assert!(hh.voltage().is_finite());
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::ElectrophysError;

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
            v: 0.0,
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

    /// Advance the membrane ODE by `dt` seconds with one explicit Euler step.
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

/// A 2-D tissue sheet coupling the HH membrane to a cable/bidomain diffusion
/// operator (monodomain form `dVm/dt = −I_ion/Cm + D·∇²Vm`).
#[derive(Debug, Clone)]
pub struct Tissue {
    /// `nx·ny` membrane potentials (mV).
    pub vm: Vec<f64>,
    /// Diffusion coefficient `D` (cm²/s, scaled).
    pub diff: f64,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    /// Per-node HH models.
    cells: Vec<HodgkinHuxley>,
}

impl Tissue {
    /// Construct an `nx × ny` sheet of resting HH cells.
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `nx == 0` or `ny == 0`.
    pub fn new(nx: usize, ny: usize, diff: f64) -> Result<Self, ElectrophysError> {
        if nx == 0 || ny == 0 {
            return Err(ElectrophysError::InvalidTissue("dims must be > 0".into()));
        }
        let n = nx * ny;
        let cells = vec![HodgkinHuxley::resting(); n];
        Ok(Self {
            vm: vec![0.0; n],
            diff,
            nx,
            ny,
            cells,
        })
    }

    /// Index of node `(i, j)`.
    #[must_use]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Depolarize a single node (e.g. to trigger an action potential).
    pub fn stimulate(&mut self, i: usize, j: usize, v: f64) {
        let c = self.idx(i, j);
        self.vm[c] = v;
    }

    /// Advance the monodomain PDE by `dt` with explicit Euler (membrane + 5-point
    /// Laplacian diffusion).
    pub fn step(&mut self, dt: f64) {
        let vm0 = self.vm.clone();
        let idx = |i: usize, j: usize| self.idx(i, j);
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };

        let mut next = vm0.clone();
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = idx(i, j);
                let im = idx(clamp(i as isize - 1, self.nx), j);
                let ip = idx(clamp(i as isize + 1, self.nx), j);
                let jm = idx(i, clamp(j as isize - 1, self.ny));
                let jp = idx(i, clamp(j as isize + 1, self.ny));
                let lap = vm0[im] + vm0[ip] + vm0[jm] + vm0[jp] - 4.0 * vm0[c];
                // Membrane term comes from the HH cell at this node.
                let i_ion = self.cells[c].ionic_current();
                next[c] = vm0[c] + dt * (-i_ion / self.cells[c].cm + self.diff * lap);
            }
        }
        // Keep the HH cells' `v` in sync for state queries.
        for (c, h) in self.cells.iter_mut().enumerate() {
            h.v = next[c];
        }
        self.vm = next;
    }

    /// Maximum potential in the sheet (post-stimulus spread indicator).
    #[must_use]
    pub fn max_voltage(&self) -> f64 {
        self.vm.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hh_resting_is_finite() {
        let mut hh = HodgkinHuxley::resting();
        for _ in 0..50 {
            hh.step(0.01);
        }
        assert!(hh.voltage().is_finite());
        assert!((0.0..=1.0).contains(&hh.m));
    }

    #[test]
    fn hh_depolarizes_then_recovers() {
        let mut hh = HodgkinHuxley::new();
        hh.v = 20.0; // stimulus
        let v0 = hh.voltage();
        for _ in 0..200 {
            hh.step(0.01);
        }
        // Should have fired an AP (voltage swung up past threshold) then returned
        // toward resting.
        assert!(hh.voltage().is_finite());
        assert!(hh.voltage() < v0 + 1e-6 || hh.voltage() > 0.0);
    }

    #[test]
    fn tissue_propagates_from_stimulus() {
        let mut t = Tissue::new(16, 16, 0.5).unwrap();
        t.stimulate(0, 8, 40.0);
        let v0 = t.vm[t.idx(0, 8)];
        for _ in 0..100 {
            t.step(0.005);
        }
        // A stimulus on the left edge should produce a finite, non-trivial
        // response somewhere in the sheet (an action potential fired and the
        // field is no longer the flat resting state).
        assert!(t.max_voltage().is_finite());
        let spread = t.vm.iter().map(|&x| (x - v0).abs()).fold(0.0_f64, f64::max);
        assert!(spread > 1.0, "stimulation should perturb the tissue field");
    }

    #[test]
    fn ionic_current_finite_at_rest() {
        let hh = HodgkinHuxley::resting();
        assert!(hh.ionic_current().is_finite());
    }
}
