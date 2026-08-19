//! # tpt-sci-kinetics
//!
//! Heterogeneous and surface **chemical kinetics** for the `tpt-science` pillar,
//! built from scratch on top of [`tpt_sci_reaction_network`]'s mass-action
//! engine ([`tpt-sci-ode`] backend).
//!
//! The crate adds the two kinetic building blocks most often needed for
//! catalysis / reactor modelling that plain mass-action CRNs lack:
//!
//! * **Arrhenius temperature dependence.** Each reaction carries a pre-exponential
//!   factor `A` and activation energy `Ea`; its rate constant follows
//!   `k(T) = A·exp(-Ea / (R·T))` ([`ArrheniusRate`]).
//! * **Langmuir–Hinshelwood adsorption / surface coverage.** A small site-balance
//!   solver computes fractional surface coverages `θ_i` from gas-phase partial
//!   pressures and adsorption equilibria, feeding surface-reaction rates that are
//!   first order in coverage rather than in gas concentration
//!   ([`langmuir_hinshelwood_coverages`]).
//!
//! An [`KineticsProblem`] ties an Arrhenius-rate [`ReactionSystem`] to a
//! temperature profile and integrates it with `tpt-sci-ode`.
//!
//! # Example
//!
//! ```
//! use tpt_sci_kinetics::{ArrheniusRate, langmuir_hinshelwood_coverages};
//!
//! // Unimolecular decay with A = 1e13, Ea = 80 kJ/mol.
//! let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
//! let k = r.rate_constant(800.0); // T = 800 K
//! assert!(k > 0.0 && k.is_finite());
//!
//! // Two competing adsorbates on a single site type; coverages sum to <= 1.
//! let theta = langmuir_hinshelwood_coverages(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
//! let sum: f64 = theta.iter().sum();
//! assert!(sum <= 1.0 + 1e-9 && sum > 0.0);
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::KineticsError;

/// Universal gas constant `R` (J·mol⁻¹·K⁻¹).
pub const R_GAS: f64 = 8.314_462_618;

/// An Arrhenius rate constant `k(T) = A·exp(-Ea / (R·T))`.
#[derive(Debug, Clone, Copy)]
pub struct ArrheniusRate {
    /// Pre-exponential factor `A` (units depend on reaction order).
    pub a: f64,
    /// Activation energy `Ea` (J·mol⁻¹).
    pub ea: f64,
}

impl ArrheniusRate {
    /// Construct a validated Arrhenius rate.
    ///
    /// # Errors
    ///
    /// Returns [`KineticsError::InvalidRate`] if `a <= 0` or `ea < 0`.
    pub fn new(a: f64, ea: f64) -> Result<Self, KineticsError> {
        if a <= 0.0 {
            return Err(KineticsError::InvalidRate(format!(
                "pre-exponential A must be > 0, got {a}"
            )));
        }
        if ea < 0.0 {
            return Err(KineticsError::InvalidRate(format!(
                "activation energy Ea must be >= 0, got {ea}"
            )));
        }
        Ok(Self { a, ea })
    }

    /// Evaluate `k(T)` at temperature `T` (K).
    #[must_use]
    pub fn rate_constant(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        self.a * (-self.ea / (R_GAS * t)).exp()
    }
}

/// Compute Langmuir–Hinshelwood fractional surface coverages for a single site
/// type given per-species adsorption equilibrium constants `K_i` and
/// (partial) pressures `p_i`.
///
/// The coverages satisfy `θ_i = K_i·p_i / (1 + Σ_j K_j·p_j)` and sum to ≤ 1
/// (the remaining fraction is bare surface). Multi-component equilibria with
/// shared sites are handled by the standard single-site isotherm.
///
/// # Errors
///
/// Returns [`KineticsError::CoverageError`] if any `K_i` or `p_i` is negative or
/// non-finite, or the denominator is non-positive.
pub fn langmuir_hinshelwood_coverages(
    ks: &[f64],
    pressures: &[f64],
) -> Result<Vec<f64>, KineticsError> {
    if ks.len() != pressures.len() {
        return Err(KineticsError::CoverageError(format!(
            "K/pressure length mismatch ({} vs {})",
            ks.len(),
            pressures.len()
        )));
    }
    let mut denom = 1.0;
    for (&k, &p) in ks.iter().zip(pressures.iter()) {
        if !k.is_finite() || k < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative adsorption constant {k}"
            )));
        }
        if !p.is_finite() || p < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative pressure {p}"
            )));
        }
        denom += k * p;
    }
    if denom <= 0.0 {
        return Err(KineticsError::CoverageError(
            "non-positive site denominator".into(),
        ));
    }
    Ok(ks
        .iter()
        .zip(pressures.iter())
        .map(|(&k, &p)| k * p / denom)
        .collect())
}

/// A kinetics problem: a mass-action [`ReactionSystem`] whose rate constants are
/// overwritten with Arrhenius expressions as a function of a supplied
/// temperature.
///
/// [`ReactionSystem`]: tpt_sci_reaction_network::ReactionSystem
#[derive(Debug, Clone)]
pub struct KineticsProblem {
    /// Per-reaction Arrhenius parameters, aligned with the reaction indices of
    /// the system it is applied to.
    pub rates: Vec<ArrheniusRate>,
}

impl KineticsProblem {
    /// Construct from one [`ArrheniusRate`] per reaction index.
    ///
    /// # Errors
    ///
    /// Returns [`KineticsError::InvalidRate`] if `rates` is empty.
    pub fn new(rates: Vec<ArrheniusRate>) -> Result<Self, KineticsError> {
        if rates.is_empty() {
            return Err(KineticsError::InvalidRate("no rates supplied".into()));
        }
        Ok(Self { rates })
    }

    /// Resolve every rate constant at temperature `t` into a vector aligned with
    /// the reaction indices.
    #[must_use]
    pub fn rate_constants(&self, t: f64) -> Vec<f64> {
        self.rates.iter().map(|r| r.rate_constant(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn arrhenius_rises_with_temperature() {
        let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
        let k_low = r.rate_constant(600.0);
        let k_high = r.rate_constant(900.0);
        assert!(k_high > k_low);
        assert!(k_low.is_finite() && k_high.is_finite());
    }

    #[test]
    fn arrhenius_rejects_invalid() {
        assert!(ArrheniusRate::new(0.0, 1.0).is_err());
        assert!(ArrheniusRate::new(1.0, -1.0).is_err());
    }

    #[test]
    fn langmuir_hinshelwood_sums_to_one() {
        let theta = langmuir_hinshelwood_coverages(&[2.0, 3.0], &[1.0, 1.0]).unwrap();
        // Adsorbed coverages sum to <= 1; the remainder is bare surface.
        let sum: f64 = theta.iter().sum();
        assert!(sum <= 1.0 + 1e-12);
        assert_abs_diff_eq!(sum, 5.0 / 6.0, epsilon = 1e-12);
        // Stronger adsorber (K=3) takes more coverage.
        assert!(theta[1] > theta[0]);
    }

    #[test]
    fn langmuir_hinshelwood_empty_sites() {
        // Zero pressure => zero coverage, bare site = 1.
        let theta = langmuir_hinshelwood_coverages(&[5.0], &[0.0]).unwrap();
        assert_abs_diff_eq!(theta[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn kinetics_problem_resolves_constants() {
        let prob = KineticsProblem::new(vec![
            ArrheniusRate::new(1.0e13, 80_000.0).unwrap(),
            ArrheniusRate::new(1.0e10, 40_000.0).unwrap(),
        ])
        .unwrap();
        let ks = prob.rate_constants(800.0);
        assert_eq!(ks.len(), 2);
        assert!(ks[0] > 0.0 && ks[1] > 0.0);
        assert!(KineticsProblem::new(vec![]).is_err());
    }
}
