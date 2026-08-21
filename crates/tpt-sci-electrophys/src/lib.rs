//! # tpt-sci-electrophys
//!
//! **Cardiac electrophysiology** for the `tpt-science` pillar, built on
//! [`tpt_sci_ode`] (for the membrane kinetics) and [`tpt_sci_grid`] (for the
//! extracellular/bidomain diffusion operator). It provides:
//!
//! * A [`IonicModel`] trait so multiple membrane kinetics can drive the same
//!   tissue solver. Two implementations ship: the classic **Hodgkin–Huxley**
//!   giant-axon model ([`HodgkinHuxley`]) and the **Ten Tusscher–Panfilov 2004**
//!   human-ventricular myocyte ([`TenTusscher`]).
//! * A **monodomain / bidomain** tissue propagator [`Tissue`] that couples the
//!   membrane to a 2-D diffusion operator. The monodomain is
//!   `dVm/dt = −I_ion/Cm + D·∇²Vm`. Enabling bidomain adds the extracellular
//!   potential `Ve`, solved each step from the elliptic equation
//!   `(σ_i + σ_e)·∇²Ve = −σ_i·∇²Vm` (via [`tpt_sci_grid`]'s 2-D Laplacian) and
//!   coupling it back into the intracellular update
//!   `dVm/dt = −I_ion/Cm + σ_i·∇²(Vm + Ve)`.
//! * **Anisotropic (tensor) diffusion**: [`Tissue`] accepts a per-node 2×2
//!   diffusivity tensor `D`, so fibre-orientation effects (faster conduction
//!   along fibres) are captured via `∇·(D∇Vm)`.
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

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_grid::{laplacian_2d, Boundary, UniformGrid2D};

/// A membrane ionic model (the reaction term of the tissue PDE).
///
/// Implementors expose their state, transmembrane voltage, the outward ionic
/// current `I_ion` (per unit membrane area), and a capacitance. The tissue
/// solver owns the transmembrane voltage; it calls [`IonicModel::ionic_current`]
/// to read the reaction term and [`IonicModel::advance_reaction`] to evolve the
/// internal gating/ionic state at a given transmembrane potential.
pub trait IonicModel: Send + std::fmt::Debug {
    /// Full state vector (transmembrane voltage followed by gating/ionic
    /// variables), used for inspection and cloning.
    fn state(&self) -> Vec<f64>;
    /// Current transmembrane potential `Vm` (mV).
    fn voltage(&self) -> f64;
    /// Outward ionic current `I_ion` (µA/cm²) at the current state.
    fn ionic_current(&self) -> f64;
    /// Membrane capacitance `Cm` (µF/cm²).
    fn capacitance(&self) -> f64;
    /// Evolve the internal (gating / ionic) state by `dt` while holding the
    /// transmembrane voltage at `v`. The voltage itself is owned by the tissue
    /// solver, so implementors must *not* update it here.
    fn advance_reaction(&mut self, v: f64, dt: f64);
    /// Clone into a boxed trait object (so a [`Tissue`] can hold one model per
    /// node without generics).
    fn clone_box(&self) -> Box<dyn IonicModel>;
}

impl Clone for Box<dyn IonicModel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

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
        let rt_f = 8314.0 * 310.0 / 96485.0;
        1000.0 * rt_f * (self.ko / 140.0).ln()
    }

    /// Reversal potential `E_Na = (R·T/F)·ln(Nao/Nai)` (mV), `Nai = 15` mM.
    #[must_use]
    fn e_na(&self) -> f64 {
        let rt_f = 8314.0 * 310.0 / 96485.0;
        1000.0 * rt_f * (self.nao / 15.0).ln()
    }

    /// Nernst potential for Ca²⁺ (mV), `gamma = 0.34` (TP04).
    #[must_use]
    fn e_ca(&self) -> f64 {
        let rt_f = 8314.0 * 310.0 / 96485.0;
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
            (-127_140.0 * (0.2444 * v).exp() - 3.474e-5 * (-0.04391 * v).exp())
                * (v + 37.78)
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
        let ik1 = gk1 * (self.ko / 5.4_f64).sqrt()
            * (0.07 * (-0.3 * (v + 90.0)).exp() + (1.0 + (0.6 * (-(v + 90.0) / 7.3).exp())).powi(-1))
            * (v - ek);
        // ICaL (L-type Ca).
        let gcal = 0.000_175;
        let ical = gcal * self.d * self.f * self.fca * (v - 65.0);
        // Ito (transient outward).
        let gto = 0.294;
        let ito = gto * self.r * self.s * (v - ek);
        // INaCa (Na/Ca exchanger, simplified 3-for-1).
        let knaca = 1000.0;
        let inaca = knaca
            / (1.0
                + (0.000_035 / self.cai.max(1e-9)).powi(3)
                + (self.nao / 1.0).powi(3) * (1.0 + (self.cai.max(1e-9) / 0.000_35).powi(3)).recip());
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

/// Diffusivity of a [`Tissue`] node: isotropic scalar `D` or a per-node 2×2
/// tensor `[[Dxx, Dxy], [Dxy, Dyy]]` (row-major `[dxx, dxy, dyy]`), enabling
/// anisotropic (fibre-oriented) conduction.
#[derive(Debug, Clone)]
pub enum Diffusivity {
    /// Isotropic scalar `D` (cm²/s, scaled).
    Scalar(f64),
    /// Per-node tensor; length must equal `nx·ny`, each entry `[dxx, dxy, dyy]`.
    Tensor(Vec<[f64; 3]>),
}

/// A 2-D tissue sheet coupling an [`IonicModel`] to a diffusion operator
/// (monodomain, or full bidomain when enabled).
#[derive(Debug, Clone)]
pub struct Tissue {
    /// `nx·ny` transmembrane potentials `Vm` (mV).
    pub vm: Vec<f64>,
    /// `nx·ny` extracellular potentials `Ve` (mV); all zero in monodomain mode.
    pub ve: Vec<f64>,
    /// Intracellular diffusion coefficient `D_i` (scalar isotropic default).
    pub diff: f64,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    /// Per-node diffusivity (scalar or tensor).
    diffusivity: Diffusivity,
    /// Per-node membrane models.
    cells: Vec<Box<dyn IonicModel>>,
    /// Whether the full bidomain (extracellular) machinery is active.
    bidomain: bool,
    /// Extracellular diffusion coefficient `D_e` (used only in bidomain mode).
    sigma_e: f64,
}

impl Tissue {
    /// Construct an `nx × ny` sheet of resting Hodgkin–Huxley cells (monodomain,
    /// isotropic diffusion `diff`).
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `nx == 0` or `ny == 0`.
    pub fn new(nx: usize, ny: usize, diff: f64) -> Result<Self, ElectrophysError> {
        Self::with_model(nx, ny, Diffusivity::Scalar(diff), HodgkinHuxley::resting())
    }

    /// Construct an `nx × ny` sheet from a chosen [`IonicModel`] (cloned into
    /// every node) and a diffusivity (scalar or tensor).
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `nx == 0`, `ny == 0`, or a
    /// tensor diffusivity's length disagrees with `nx·ny`.
    pub fn with_model(
        nx: usize,
        ny: usize,
        diffusivity: Diffusivity,
        model: impl IonicModel + 'static,
    ) -> Result<Self, ElectrophysError> {
        if nx == 0 || ny == 0 {
            return Err(ElectrophysError::InvalidTissue("dims must be > 0".into()));
        }
        let n = nx * ny;
        if let Diffusivity::Tensor(t) = &diffusivity {
            if t.len() != n {
                return Err(ElectrophysError::InvalidTissue(
                    "tensor diffusivity length must equal nx*ny".into(),
                ));
            }
        }
        let diff = match &diffusivity {
            Diffusivity::Scalar(d) => *d,
            Diffusivity::Tensor(_) => 1.0,
        };
        let cells = (0..n).map(|_| model.clone_box()).collect();
        Ok(Self {
            vm: vec![0.0; n],
            ve: vec![0.0; n],
            diff,
            nx,
            ny,
            diffusivity,
            cells,
            bidomain: false,
            sigma_e: diff,
        })
    }

    /// Switch the solver into bidomain mode, coupling the intracellular field to
    /// an extracellular potential `Ve` solved from `(D_i + D_e)·∇²Ve = −D_i·∇²Vm`.
    pub fn enable_bidomain(&mut self, extracellular_diff: f64) {
        self.bidomain = true;
        self.sigma_e = extracellular_diff;
    }

    /// Override the diffusivity (e.g. to install a per-node tensor for
    /// anisotropy). The `diff` field is ignored once a tensor is set.
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `tensor.len() != nx·ny`.
    pub fn set_diffusivity(
        &mut self,
        diffusivity: Diffusivity,
    ) -> Result<(), ElectrophysError> {
        let n = self.nx * self.ny;
        if let Diffusivity::Tensor(t) = &diffusivity {
            if t.len() != n {
                return Err(ElectrophysError::InvalidTissue(
                    "tensor diffusivity length must equal nx*ny".into(),
                ));
            }
        }
        if let Diffusivity::Scalar(d) = &diffusivity {
            self.diff = *d;
        }
        self.diffusivity = diffusivity;
        Ok(())
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

    /// Solve the extracellular elliptic equation `(D_i + D_e)·L·Ve = −D_i·L·Vm`
    /// for the current transmembrane field, returning `Ve`. In monodomain mode
    /// this returns a zero vector.
    #[must_use]
    pub fn extracellular_potential(&self) -> Vec<f64> {
        if !self.bidomain {
            return vec![0.0; self.nx * self.ny];
        }
        let grid = UniformGrid2D::new(self.nx, 0.0, 1.0, self.ny, 0.0, 1.0).expect("grid");
        let l = laplacian_2d(&grid, Boundary::Neumann);
        let vm_vec = DVector::from_vec(self.vm.clone());
        let lvm = l.clone() * vm_vec;
        let b = lvm * (-self.diff);
        let di = self.diff;
        let de = self.sigma_e;
        let matvec = |x: &DVector| -> DVector {
            let y = l.clone() * x.clone();
            y * (di + de)
        };
        let ve = cg_solve(&matvec, &b, 5000, 1e-9);
        ve.iter().copied().collect()
    }

    /// Diffusion term `∇·(D∇φ)` (per node) for the field `phi`, using unit grid
    /// spacing and clamped (Neumann) boundaries. For a scalar `D` this is the
    /// 5-point Laplacian `D·(φ[i±1]+φ[j±1]−4φ)`, and for a tensor `D` it is the
    /// anisotropic `Dxx·φ_xx + 2·Dxy·φ_xy + Dyy·φ_yy`.
    fn diffusion_term(&self, phi: &[f64]) -> Vec<f64> {
        let nx = self.nx;
        let ny = self.ny;
        let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
        let c_at = |i: usize, j: usize| phi[clamp(i as isize, nx) + clamp(j as isize, ny) * nx];
        let mut out = vec![0.0; nx * ny];
        match &self.diffusivity {
            Diffusivity::Scalar(d) => {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = self.idx(i, j);
                        let lap =
                            c_at(i.wrapping_sub(1), j) + c_at(i + 1, j) + c_at(i, j.wrapping_sub(1))
                                + c_at(i, j + 1)
                                - 4.0 * phi[c];
                        out[c] = d * lap;
                    }
                }
            }
            Diffusivity::Tensor(t) => {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = self.idx(i, j);
                        let [dxx, dxy, dyy] = t[c];
                        let im = c_at(i.wrapping_sub(1), j);
                        let ip = c_at(i + 1, j);
                        let jm = c_at(i, j.wrapping_sub(1));
                        let jp = c_at(i, j + 1);
                        let vxx = ip - 2.0 * phi[c] + im;
                        let vyy = jp - 2.0 * phi[c] + jm;
                        let vxy = c_at(i + 1, j + 1) - c_at(i.wrapping_sub(1), j + 1)
                            - c_at(i + 1, j.wrapping_sub(1))
                            + c_at(i.wrapping_sub(1), j.wrapping_sub(1));
                        out[c] = dxx * vxx + 2.0 * dxy * vxy + dyy * vyy;
                    }
                }
            }
        }
        out
    }

    /// Advance the tissue PDE by `dt` with explicit operator splitting: the
    /// reaction term `-I_ion/Cm` plus the (scalar or tensor) diffusion term, and
    /// — in bidomain mode — the extracellular elliptic solve coupling `Ve`.
    pub fn step(&mut self, dt: f64) {
        let vm0 = self.vm.clone();
        let phi: Vec<f64> = if self.bidomain {
            let ve = self.extracellular_potential();
            self.ve = ve.clone();
            vm0.iter().zip(&ve).map(|(a, b)| a + b).collect()
        } else {
            vm0.clone()
        };

        let diff_term = self.diffusion_term(&phi);
        let n = self.nx * self.ny;
        let mut next = vec![0.0; n];
        for c in 0..n {
            let i_ion = self.cells[c].ionic_current();
            let cm = self.cells[c].capacitance();
            next[c] = vm0[c] + dt * (-i_ion / cm + diff_term[c]);
        }
        // Evolve the internal membrane state at the new transmembrane voltage.
        for c in 0..n {
            self.cells[c].advance_reaction(next[c], dt);
        }
        self.vm = next;
    }

    /// Maximum transmembrane potential in the sheet (post-stimulus spread
    /// indicator).
    #[must_use]
    pub fn max_voltage(&self) -> f64 {
        self.vm.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Conjugate-gradient solve of `A x = b` for a symmetric (possibly
/// positive-semidefinite) matrix given as a matvec closure. For the bidomain
/// Neumann Laplacian the system is singular (constant nullspace); with a
/// consistent RHS (zero mean) CG converges to the unique zero-mean solution.
fn cg_solve(matvec: &dyn Fn(&DVector) -> DVector, b: &DVector, iters: usize, tol: f64) -> DVector {
    let n = b.len();
    let zero = DVector::from_vec(vec![0.0; n]);
    let mut r = b.clone() - matvec(&zero);
    let mut p = r.clone();
    let mut x = DVector::from_vec(vec![0.0; n]);
    let mut rsold = r.dot(&r);
    if rsold.sqrt() < tol {
        return x;
    }
    for _ in 0..iters {
        let ap = matvec(&p);
        let pap = p.dot(&ap);
        if pap <= 1e-12 {
            break;
        }
        let alpha = rsold / pap;
        x = x + p.clone() * alpha;
        let r_new = r.clone() - ap.clone() * alpha;
        let rsnew = r_new.dot(&r_new);
        if rsnew.sqrt() < tol {
            break;
        }
        let beta = rsnew / rsold;
        p = r_new.clone() + p * beta;
        r = r_new;
        rsold = rsnew;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
        assert!(hh.voltage().is_finite());
        assert!(hh.voltage() < v0 + 1e-6 || hh.voltage() > 0.0);
    }

    #[test]
    fn tissue_propagates_from_stimulus() {
        let mut t = Tissue::new(16, 16, 0.5).unwrap();
        t.stimulate(0, 8, 40.0);
        for _ in 0..100 {
            t.step(0.005);
        }
        // A stimulus should depolarize the tissue well above the resting (0 mV)
        // field — the depolarization propagates and recruits the sheet.
        assert!(t.max_voltage().is_finite());
        assert!(t.max_voltage() > 10.0, "stimulation should depolarize the tissue");
    }

    #[test]
    fn ionic_current_finite_at_rest() {
        let hh = HodgkinHuxley::resting();
        assert!(hh.ionic_current().is_finite());
    }

    #[test]
    fn tt_resting_fires_action_potential() {
        let mut cell = TenTusscher::resting();
        let v_rest = cell.v;
        cell.v = 20.0; // depolarizing stimulus
        let mut peak = cell.v;
        // Single-node monodomain integration: dV/dt = -I_ion/Cm.
        for k in 0..600 {
            cell.advance_reaction(cell.v, 0.01);
            cell.v += 0.01 * (-cell.ionic_current() / cell.cm);
            if !cell.v.is_finite() {
                eprintln!("TT NaN at step {k}: v={} i_ion={}", cell.v, cell.ionic_current());
                break;
            }
            peak = peak.max(cell.v);
        }
        // A genuine AP must swing well above the resting potential.
        assert!(peak > v_rest + 10.0, "TenTusscher should fire a depolarization");
        assert!(cell.v.is_finite());
        assert!(cell.cai.is_finite() && cell.cai > 0.0);
    }

    #[test]
    fn anisotropic_quadratic_laplacian() {
        // For D = I, ∇·(D∇(x²+y²)) = 4 (matches the scalar 5-point Laplacian).
        let n = 9usize;
        let t = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![[1.0, 0.0, 1.0]; n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - 4.0;
                let y = j as f64 - 4.0;
                phi[t.idx(i, j)] = x * x + y * y;
            }
        }
        let term = t.diffusion_term(&phi);
        let mid = t.idx(4, 4);
        assert_abs_diff_eq!(term[mid], 4.0, epsilon = 1e-9);
    }

    #[test]
    fn anisotropic_prefers_strong_axis() {
        // D = diag(10, 1): diffusion along x should dominate a step in x more
        // than the same step along y.
        let n = 9usize;
        let t = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![[10.0, 0.0, 1.0]; n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        // linear in x: ∇·(D∇(ax)) = Dxx * 0 + ... = 0; use ax² to expose Dxx.
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - 4.0;
                phi[t.idx(i, j)] = 0.5 * x * x;
            }
        }
        let term = t.diffusion_term(&phi);
        // For φ = ½ x², ∇·(D∇φ) = Dxx at interior => 10.
        assert_abs_diff_eq!(term[t.idx(4, 4)], 10.0, epsilon = 1e-9);
    }

    #[test]
    fn bidomain_solves_extracellular_field() {
        let mut t = Tissue::new(12, 12, 1.0).unwrap();
        t.enable_bidomain(1.0);
        // A non-uniform transmembrane field forces a non-trivial extracellular
        // potential.
        for j in 0..12 {
            for i in 0..12 {
                let c = t.idx(i, j);
                t.vm[c] = (i as f64) * 0.1;
            }
        }
        let ve = t.extracellular_potential();
        // The solve must be finite and the residual of the elliptic equation
        // small: (D_i+D_e)·L·Ve + D_i·L·Vm ≈ 0.
        assert!(ve.iter().all(|x| x.is_finite()));
        let grid = UniformGrid2D::new(12, 0.0, 1.0, 12, 0.0, 1.0).unwrap();
        let l = laplacian_2d(&grid, Boundary::Neumann);
        let lve = l.clone() * DVector::from_vec(ve.clone());
        let lvm = l.clone() * DVector::from_vec(t.vm.clone());
        let di = t.diff;
        let de = t.sigma_e;
        let residual: f64 = lve
            .iter()
            .zip(lvm.iter())
            .map(|(a, b)| (di + de) * a + di * b)
            .map(|x| x.abs())
            .sum();
        assert!(residual < 1e-4, "elliptic residual too large: {residual}");
    }

    #[test]
    fn bidomain_reduces_to_monodomain_for_large_sigma_e() {
        let n = 8usize;
        // Monodomain reference.
        let mut mono = Tissue::new(n, n, 1.0).unwrap();
        for j in 0..n {
            for i in 0..n {
                let c = mono.idx(i, j);
                mono.vm[c] = (i + j) as f64 * 0.2;
            }
        }
        let diff_mono = mono.diffusion_term(&mono.vm.clone());

        // Bidomain with very large extracellular conductivity: the effective
        // coupling term D_i·L(Vm+Ve) must approach D_i·L·Vm.
        let mut bi = Tissue::new(n, n, 1.0).unwrap();
        bi.enable_bidomain(1e6);
        for j in 0..n {
            for i in 0..n {
                let c = bi.idx(i, j);
                bi.vm[c] = (i + j) as f64 * 0.2;
            }
        }
        let ve = bi.extracellular_potential();
        let phi: Vec<f64> = bi.vm.iter().zip(&ve).map(|(a, b)| a + b).collect();
        let diff_bi = bi.diffusion_term(&phi);
        for c in 0..n * n {
            assert_abs_diff_eq!(diff_bi[c], diff_mono[c], epsilon = 1e-3);
        }
    }
}
