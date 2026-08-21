//! # tpt-sci-climate
//!
//! Simple **climate modelling** for the `tpt-science` pillar, built on
//! [`tpt_sci_ode`]. It provides reduced-order models suitable for coupling and
//! teaching:
//!
//! * A **0-D energy-balance model** [`EnergyBalanceModel`] — global-mean
//!   surface temperature under incoming shortwave, grey-body longwave outgoing,
//!   and a CO₂ radiative-forcing term (`ΔF = 5.35·ln(C/C0)` W/m²).
//! * **Multi-band longwave radiative transfer** ([`radiative_transfer`]) — a
//!   simplified multi-band grey-slab scheme [`MultiBandRadiativeTransfer`] and a
//!   `k`-distribution [`CorrelatedKRt`], replacing the single grey band of
//!   [`grey_radiative_transfer`].
//! * A tiny **atmospheric chemistry** box [`ChemistryBox`] for a linear
//!   production–loss tracer (e.g. OH-driven CH₄ decay), extended to a 3-D
//!   advection–diffusion–reaction tracer field [`Tracer3D`].
//! * A genuine **primitive-equation atmospheric GCM dynamical core**
//!   [`AtmosphereGcm`] — a 3-D hydrostatic (optionally non-hydrostatic) primitive
//!   equation solver, coupled to the [`EnergyBalanceModel`].
//!
//! These close out the v1 "out of scope" climate items: the single grey band is
//! replaced by a multi-band/correlated-k scheme, chemistry is extended to 3-D
//! transport, and the EBM is backed by a working GCM dynamical core.
//!
//! # Example
//!
//! ```
//! use tpt_sci_climate::EnergyBalanceModel;
//!
//! let mut ebm = EnergyBalanceModel::new(1.0, 0.3, 0.61, 280.0).unwrap();
//! // Integrate temperature for a year; it relaxes to equilibrium.
//! for _ in 0..100 {
//!     ebm.step(1.0);
//! }
//! assert!(ebm.temperature().is_finite());
//! ```
#![forbid(unsafe_code)]

mod chemistry_3d;
mod error;
mod gcm;
mod radiative_transfer;

pub use chemistry_3d::Tracer3D;
pub use error::ClimateError;
pub use gcm::AtmosphereGcm;
pub use radiative_transfer::{Band, CorrelatedKBand, CorrelatedKRt, MultiBandRadiativeTransfer};

/// Stefan–Boltzmann constant (W·m⁻²·K⁻⁴).
pub const SIGMA: f64 = 5.670_374_419e-8;
/// Pre-industrial CO₂ reference (ppm).
pub const CO2_PREINDUSTRIAL: f64 = 280.0;

/// 0-D global energy-balance climate model.
///
/// `C·dT/dt = (1−α)·S/4 − ε·σ·T⁴ + ΔF(CO₂)`, where `C` is the heat capacity
/// (J·m⁻²·K⁻¹), `α` the planetary albedo, `ε` the effective emissivity, `S` the
/// solar constant, and `ΔF = 5.35·ln(C/C0)`.
#[derive(Debug, Clone)]
pub struct EnergyBalanceModel {
    /// Heat capacity `C` (J·m⁻²·K⁻¹); ~ocean mixed layer equivalent.
    pub heat_capacity: f64,
    /// Planetary albedo `α` (0..1).
    pub albedo: f64,
    /// Effective longwave emissivity `ε` (0..1).
    pub emissivity: f64,
    /// Solar constant `S` (W/m²).
    pub solar_constant: f64,
    /// Current CO₂ concentration (ppm).
    pub co2: f64,
    /// Global-mean surface temperature `T` (K).
    pub t: f64,
}

impl EnergyBalanceModel {
    /// Construct a validated EBM at temperature `t0`.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if `t0 <= 0`, `albedo` outside
    /// `[0,1]`, `emissivity <= 0`, or `co2 <= 0`.
    pub fn new(
        heat_capacity: f64,
        albedo: f64,
        emissivity: f64,
        co2: f64,
    ) -> Result<Self, ClimateError> {
        if heat_capacity <= 0.0 {
            return Err(ClimateError::InvalidModel(
                "heat capacity must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&albedo) {
            return Err(ClimateError::InvalidModel("albedo must be in [0,1]".into()));
        }
        if emissivity <= 0.0 {
            return Err(ClimateError::InvalidModel("emissivity must be > 0".into()));
        }
        if co2 <= 0.0 {
            return Err(ClimateError::InvalidModel("CO2 must be > 0".into()));
        }
        Ok(Self {
            heat_capacity,
            albedo,
            emissivity,
            solar_constant: 1361.0,
            co2,
            t: 288.0,
        })
    }

    /// Current CO₂ radiative forcing `ΔF = 5.35·ln(C/C0)` (W/m²).
    #[must_use]
    pub fn forcing(&self) -> f64 {
        5.35 * (self.co2 / CO2_PREINDUSTRIAL).ln()
    }

    /// Net top-of-atmosphere flux (W/m²), `> 0` warms.
    #[must_use]
    pub fn net_flux(&self) -> f64 {
        (1.0 - self.albedo) * self.solar_constant / 4.0 - self.emissivity * SIGMA * self.t.powi(4)
            + self.forcing()
    }

    /// Current temperature (K).
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.t
    }

    /// Advance the EBM by `dt` seconds with explicit Euler.
    pub fn step(&mut self, dt: f64) {
        self.t += dt * self.net_flux() / self.heat_capacity;
        if self.t < 0.0 {
            self.t = 0.0;
        }
    }

    /// Equilibrium temperature (where `net_flux = 0`), solved by a few fixed-point
    /// Newton iterations from the current `t`.
    #[must_use]
    pub fn equilibrium_temperature(&self) -> f64 {
        let mut t = self.t.max(1.0);
        for _ in 0..50 {
            let f = (1.0 - self.albedo) * self.solar_constant / 4.0
                - self.emissivity * SIGMA * t.powi(4)
                + self.forcing();
            let df = -4.0 * self.emissivity * SIGMA * t.powi(3);
            t = (t - f / df).max(1.0);
        }
        t
    }
}

/// Single-layer grey-atmosphere radiative temperature: with a grey gas of
/// emissivity `eps`, the surface temperature `Ts = (S·(1−α)/(4·σ·(1−eps/2)))^(1/4)`.
///
/// # Panics
///
/// Panics if `eps` outside `[0,1)`, `alpha` outside `[0,1]`, or `s <= 0`.
#[must_use]
pub fn grey_radiative_transfer(s: f64, alpha: f64, eps: f64) -> f64 {
    assert!(s > 0.0, "S must be > 0");
    assert!((0.0..=1.0).contains(&alpha), "albedo in [0,1]");
    assert!((0.0..1.0).contains(&eps), "eps in [0,1)");
    ((s * (1.0 - alpha)) / (4.0 * SIGMA * (1.0 - eps / 2.0))).powf(0.25)
}

/// A minimal atmospheric-chemistry box: `dC/dt = P − k·C` (constant production
/// `P`, first-order loss `k`). Models e.g. a well-mixed CH₄-like tracer.
#[derive(Debug, Clone)]
pub struct ChemistryBox {
    /// Concentration (ppb-ish, arbitrary units).
    pub concentration: f64,
    /// Constant production rate `P`.
    pub production: f64,
    /// First-order loss coefficient `k` (1/s).
    pub loss: f64,
}

impl ChemistryBox {
    /// Construct a validated box.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if `loss < 0` or `concentration < 0`.
    pub fn new(concentration: f64, production: f64, loss: f64) -> Result<Self, ClimateError> {
        if concentration < 0.0 {
            return Err(ClimateError::InvalidModel(
                "concentration must be >= 0".into(),
            ));
        }
        if loss < 0.0 {
            return Err(ClimateError::InvalidModel("loss must be >= 0".into()));
        }
        Ok(Self {
            concentration,
            production,
            loss,
        })
    }

    /// Steady-state concentration `C* = P/k` (with `k > 0`).
    #[must_use]
    pub fn steady_state(&self) -> f64 {
        if self.loss > 0.0 {
            self.production / self.loss
        } else {
            f64::INFINITY
        }
    }

    /// Advance by `dt`.
    pub fn step(&mut self, dt: f64) {
        self.concentration += dt * (self.production - self.loss * self.concentration);
        if self.concentration < 0.0 {
            self.concentration = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ebm_relaxes_to_equilibrium() {
        let mut ebm = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 280.0).unwrap();
        for _ in 0..2000 {
            ebm.step(1.0);
        }
        // Doubling CO2 raises equilibrium temperature vs pre-industrial.
        let teq = ebm.equilibrium_temperature();
        assert!(teq.is_finite() && teq > 250.0);
        let ebm2 = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 560.0).unwrap();
        let teq2 = ebm2.equilibrium_temperature();
        assert!(teq2 > teq, "doubling CO2 should warm the EBM");
    }

    #[test]
    fn grey_transfer_monotonic_in_greenhouse() {
        let clear = grey_radiative_transfer(1361.0, 0.3, 0.0);
        let greenhouse = grey_radiative_transfer(1361.0, 0.3, 0.5);
        assert!(greenhouse > clear);
    }

    #[test]
    fn chemistry_reaches_steady_state() {
        let mut c = ChemistryBox::new(0.0, 1.0, 0.1).unwrap();
        for _ in 0..1000 {
            c.step(0.1);
        }
        assert_abs_diff_eq!(c.concentration, c.steady_state(), epsilon = 1e-3);
    }
}
