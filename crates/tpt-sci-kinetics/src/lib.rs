//! # tpt-sci-kinetics
//!
//! Heterogeneous and surface **chemical kinetics** for the `tpt-science` pillar,
//! built from scratch on top of [`tpt_sci_reaction_network`]'s mass-action
//! engine ([`tpt_sci_ode`] backend).
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
//!   ([`langmuir_hinshelwood_coverages`]), generalized to multiple independent
//!   site types via [`multi_site_langmuir_hinshelwood_coverages`].
//! * **Eley–Rideal mechanism.** [`EleyRideal`] models a step where one
//!   reactant is adsorbed (subject to LH coverage) and the other reacts
//!   directly from the gas phase, `r = k·θ_adsorbed·[gas_phase]`, pluggable
//!   into a `ReactionSystem` via [`EleyRideal::into_rate_law`].
//! * **Coverage-dependent activation energy.** [`CoverageDependentArrheniusRate`]
//!   extends the Arrhenius law with a linear Brønsted–Evans–Polanyi-style
//!   dependence `Ea(θ) = Ea0 + α·θ`, giving `k(T, θ) = A·exp(-Ea(θ)/(R·T))`.
//!
//! An [`KineticsProblem`] ties an Arrhenius-rate
//! [`ReactionSystem`](tpt_sci_reaction_network::ReactionSystem) to a
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

/// Compute Langmuir–Hinshelwood fractional surface coverages for **multiple
/// distinct site types**, e.g. species that adsorb on site type A vs species
/// that adsorb on an independent site type B (each with its own site-balance
/// normalization).
///
/// `ks` and `pressures` are per-species, as in [`langmuir_hinshelwood_coverages`].
/// `site_of` assigns each species (by index) to a site-type index; species
/// sharing a site-type index compete for the *same* pool of sites and their
/// coverages are normalized together, while species on different site types
/// are normalized independently. This reduces to
/// [`langmuir_hinshelwood_coverages`] when every species is assigned to the
/// same (single) site type.
///
/// The returned vector is aligned with `ks`/`pressures`/`site_of` (one
/// coverage per species), and within each site type the coverages sum to ≤ 1.
///
/// # Errors
///
/// Returns [`KineticsError::CoverageError`] if the input lengths mismatch, any
/// `K_i`/`p_i` is negative or non-finite, or any per-site-type denominator is
/// non-positive.
///
/// # Example
///
/// ```
/// use tpt_sci_kinetics::multi_site_langmuir_hinshelwood_coverages;
///
/// // Species 0, 1 adsorb on site type 0; species 2 adsorbs on site type 1.
/// let theta = multi_site_langmuir_hinshelwood_coverages(
///     &[2.0, 3.0, 5.0],
///     &[1.0, 1.0, 1.0],
///     &[0, 0, 1],
/// )
/// .unwrap();
/// // Site type 0 coverages sum with the bare fraction to 1, independent of site 1.
/// assert!((theta[0] + theta[1]) <= 1.0 + 1e-9);
/// assert!(theta[2] <= 1.0 + 1e-9);
/// ```
pub fn multi_site_langmuir_hinshelwood_coverages(
    ks: &[f64],
    pressures: &[f64],
    site_of: &[usize],
) -> Result<Vec<f64>, KineticsError> {
    if ks.len() != pressures.len() || ks.len() != site_of.len() {
        return Err(KineticsError::CoverageError(format!(
            "K/pressure/site-type length mismatch ({} vs {} vs {})",
            ks.len(),
            pressures.len(),
            site_of.len()
        )));
    }
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
    }
    let n_sites = site_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut denoms = vec![1.0_f64; n_sites];
    for ((&k, &p), &site) in ks.iter().zip(pressures.iter()).zip(site_of.iter()) {
        denoms[site] += k * p;
    }
    for (site, &denom) in denoms.iter().enumerate() {
        if denom <= 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-positive site denominator for site type {site}"
            )));
        }
    }
    Ok(ks
        .iter()
        .zip(pressures.iter())
        .zip(site_of.iter())
        .map(|((&k, &p), &site)| k * p / denoms[site])
        .collect())
}

/// An Eley–Rideal surface reaction step: one reactant is adsorbed (its
/// fractional coverage `θ` is subject to Langmuir–Hinshelwood site-balance
/// normalization) while the other reacts directly from the gas phase without
/// itself adsorbing. The rate law is first order in each,
/// `r = k · θ_adsorbed · [gas_phase]`, in contrast to the pure
/// Langmuir–Hinshelwood mechanism `r = k · θ_A · θ_B` where *both* reactants
/// are adsorbed.
#[derive(Debug, Clone, Copy)]
pub struct EleyRideal {
    /// Rate constant `k` for the elementary step.
    pub k: f64,
}

impl EleyRideal {
    /// Construct a validated Eley–Rideal rate law.
    ///
    /// # Errors
    ///
    /// Returns [`KineticsError::InvalidRate`] if `k < 0` or non-finite.
    pub fn new(k: f64) -> Result<Self, KineticsError> {
        if !k.is_finite() || k < 0.0 {
            return Err(KineticsError::InvalidRate(format!(
                "Eley-Rideal rate constant must be finite and >= 0, got {k}"
            )));
        }
        Ok(Self { k })
    }

    /// Evaluate `r = k · θ_adsorbed · [gas_phase_concentration]`.
    ///
    /// Returns `0.0` if either factor is `<= 0` (in particular, zero adsorbed
    /// coverage or zero gas-phase concentration gives zero rate).
    #[must_use]
    pub fn rate(&self, theta_adsorbed: f64, gas_phase_concentration: f64) -> f64 {
        if theta_adsorbed <= 0.0 || gas_phase_concentration <= 0.0 {
            return 0.0;
        }
        self.k * theta_adsorbed * gas_phase_concentration
    }

    /// Wrap this rate law as a `RateLaw::Custom` closure suitable for use
    /// directly in a [`ReactionSystem`](tpt_sci_reaction_network::ReactionSystem):
    /// `theta_index` is the index into the state vector `y` holding the
    /// adsorbed species' surface coverage, and `gas_index` is the index
    /// holding the gas-phase reactant's concentration.
    ///
    /// # Example
    ///
    /// ```
    /// use tpt_sci_kinetics::EleyRideal;
    /// use tpt_sci_reaction_network::RateLaw;
    ///
    /// let er = EleyRideal::new(2.0).unwrap();
    /// let law: RateLaw = er.into_rate_law(0, 1);
    /// // `RateLaw::Custom` only exposes evaluation through a `ReactionSystem`,
    /// // so here we just check it constructs without panicking.
    /// let _ = law;
    /// ```
    #[must_use]
    pub fn into_rate_law(
        self,
        theta_index: usize,
        gas_index: usize,
    ) -> tpt_sci_reaction_network::RateLaw {
        tpt_sci_reaction_network::RateLaw::custom(move |y: &[f64], _p: &[f64]| {
            let theta = y.get(theta_index).copied().unwrap_or(0.0);
            let gas = y.get(gas_index).copied().unwrap_or(0.0);
            self.rate(theta, gas)
        })
    }
}

/// An Arrhenius rate constant with a coverage-dependent activation energy,
/// `k(T, θ) = A·exp(-Ea(θ) / (R·T))` with the linear (Brønsted–Evans–Polanyi
/// style) dependence `Ea(θ) = Ea0 + α·θ`.
///
/// Linear coverage dependence of the activation energy is a standard,
/// well-established approximation in surface kinetics: as a surface fills up,
/// lateral adsorbate-adsorbate interactions shift the transition-state energy
/// approximately linearly in coverage over the accessible range (the same
/// linear-scaling idea behind Brønsted–Evans–Polanyi relations between
/// reaction energy and activation energy). `alpha > 0` models a surface that
/// becomes *more* difficult to react on as it fills (repulsive adsorbate
/// interactions raising the barrier); `alpha < 0` models autocatalytic-like
/// facilitation (attractive/promoting interactions lowering the barrier).
#[derive(Debug, Clone, Copy)]
pub struct CoverageDependentArrheniusRate {
    /// Pre-exponential factor `A` (units depend on reaction order).
    pub a: f64,
    /// Zero-coverage activation energy `Ea0` (J·mol⁻¹).
    pub ea0: f64,
    /// Linear coverage-dependence coefficient `α` (J·mol⁻¹), such that
    /// `Ea(θ) = Ea0 + α·θ`.
    pub alpha: f64,
}

impl CoverageDependentArrheniusRate {
    /// Construct a validated coverage-dependent Arrhenius rate.
    ///
    /// # Errors
    ///
    /// Returns [`KineticsError::InvalidRate`] if `a <= 0`, `ea0 < 0`, or `alpha`
    /// is non-finite.
    pub fn new(a: f64, ea0: f64, alpha: f64) -> Result<Self, KineticsError> {
        if a <= 0.0 {
            return Err(KineticsError::InvalidRate(format!(
                "pre-exponential A must be > 0, got {a}"
            )));
        }
        if ea0 < 0.0 {
            return Err(KineticsError::InvalidRate(format!(
                "zero-coverage activation energy Ea0 must be >= 0, got {ea0}"
            )));
        }
        if !alpha.is_finite() {
            return Err(KineticsError::InvalidRate(format!(
                "coverage coefficient alpha must be finite, got {alpha}"
            )));
        }
        Ok(Self { a, ea0, alpha })
    }

    /// Evaluate the coverage-dependent activation energy `Ea(θ) = Ea0 + α·θ`.
    #[must_use]
    pub fn activation_energy(&self, theta: f64) -> f64 {
        self.ea0 + self.alpha * theta
    }

    /// Evaluate `k(T, θ) = A·exp(-Ea(θ) / (R·T))` at temperature `T` (K) and
    /// fractional surface coverage `θ`. The activation energy is clamped to
    /// `>= 0` before evaluation, since a negative effective barrier is
    /// unphysical.
    #[must_use]
    pub fn rate_constant(&self, t: f64, theta: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let ea = self.activation_energy(theta).max(0.0);
        self.a * (-ea / (R_GAS * t)).exp()
    }
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

    #[test]
    fn multi_site_reduces_to_single_site() {
        // All species on the same site type => identical to the single-site fn.
        let ks = [2.0, 3.0];
        let ps = [1.0, 1.0];
        let single = langmuir_hinshelwood_coverages(&ks, &ps).unwrap();
        let multi = multi_site_langmuir_hinshelwood_coverages(&ks, &ps, &[0, 0]).unwrap();
        for (s, m) in single.iter().zip(multi.iter()) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-12);
        }
    }

    #[test]
    fn multi_site_normalizes_independently_per_site_type() {
        // Species 0,1 on site type 0; species 2 alone on site type 1.
        let theta = multi_site_langmuir_hinshelwood_coverages(
            &[2.0, 3.0, 5.0],
            &[1.0, 1.0, 1.0],
            &[0, 0, 1],
        )
        .unwrap();
        // Site type 0: theta_i = K_i / (1 + 2 + 3) = K_i / 6.
        assert_abs_diff_eq!(theta[0], 2.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(theta[1], 3.0 / 6.0, epsilon = 1e-12);
        // Site type 1: single adsorbate, theta_2 = 5 / (1 + 5) = 5/6, unaffected
        // by site-type-0 occupancy.
        assert_abs_diff_eq!(theta[2], 5.0 / 6.0, epsilon = 1e-12);
        let site0_sum = theta[0] + theta[1];
        assert!(site0_sum <= 1.0 + 1e-12);
        assert!(theta[2] <= 1.0 + 1e-12);
    }

    #[test]
    fn multi_site_rejects_mismatched_lengths_and_bad_values() {
        assert!(multi_site_langmuir_hinshelwood_coverages(&[1.0], &[1.0, 2.0], &[0, 0]).is_err());
        assert!(multi_site_langmuir_hinshelwood_coverages(&[-1.0], &[1.0], &[0]).is_err());
        assert!(multi_site_langmuir_hinshelwood_coverages(&[1.0], &[-1.0], &[0]).is_err());
    }

    #[test]
    fn eley_rideal_zero_when_either_factor_zero() {
        let er = EleyRideal::new(3.0).unwrap();
        assert_eq!(er.rate(0.0, 5.0), 0.0);
        assert_eq!(er.rate(0.5, 0.0), 0.0);
        assert_eq!(er.rate(0.0, 0.0), 0.0);
    }

    #[test]
    fn eley_rideal_linear_in_gas_phase_concentration() {
        let er = EleyRideal::new(2.0).unwrap();
        let r1 = er.rate(0.4, 1.0);
        let r2 = er.rate(0.4, 3.0);
        // Linear in gas-phase concentration for fixed coverage.
        assert_abs_diff_eq!(r2, 3.0 * r1, epsilon = 1e-12);
        assert_abs_diff_eq!(r1, 2.0 * 0.4 * 1.0, epsilon = 1e-12);
    }

    #[test]
    fn eley_rideal_rejects_invalid_rate() {
        assert!(EleyRideal::new(-1.0).is_err());
        assert!(EleyRideal::new(f64::NAN).is_err());
    }

    #[test]
    fn eley_rideal_into_rate_law_matches_direct_rate() {
        use tpt_sci_reaction_network::ReactionNetwork;

        let er = EleyRideal::new(1.5).unwrap();
        let mut net = ReactionNetwork::new();
        let theta_a = net.species("theta_A");
        let gas_b = net.species("B_gas");
        let c = net.species("C");
        net.reaction(
            &[(theta_a, 1.0), (gas_b, 1.0)],
            &[(c, 1.0)],
            er.into_rate_law(0, 1),
        );
        let sys = net.build().unwrap();
        let y = [0.3, 2.0, 0.0];
        let rates = sys.reaction_rates(&y);
        assert_abs_diff_eq!(rates[0], er.rate(0.3, 2.0), epsilon = 1e-12);
    }

    #[test]
    fn coverage_dependent_matches_plain_arrhenius_when_alpha_zero() {
        let plain = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
        let cov = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, 0.0).unwrap();
        for theta in [0.0, 0.3, 1.0] {
            assert_abs_diff_eq!(
                cov.rate_constant(800.0, theta),
                plain.rate_constant(800.0),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn coverage_dependent_rate_constant_monotonic_in_theta() {
        // alpha > 0: Ea increases with theta => k decreases with theta.
        let repulsive = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, 20_000.0).unwrap();
        let k_low = repulsive.rate_constant(800.0, 0.0);
        let k_high = repulsive.rate_constant(800.0, 1.0);
        assert!(k_high < k_low);

        // alpha < 0: Ea decreases with theta => k increases with theta.
        let promoting = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, -20_000.0).unwrap();
        let k_low = promoting.rate_constant(800.0, 0.0);
        let k_high = promoting.rate_constant(800.0, 1.0);
        assert!(k_high > k_low);
    }

    #[test]
    fn coverage_dependent_rejects_invalid() {
        assert!(CoverageDependentArrheniusRate::new(0.0, 1.0, 0.0).is_err());
        assert!(CoverageDependentArrheniusRate::new(1.0, -1.0, 0.0).is_err());
        assert!(CoverageDependentArrheniusRate::new(1.0, 1.0, f64::NAN).is_err());
    }
}
