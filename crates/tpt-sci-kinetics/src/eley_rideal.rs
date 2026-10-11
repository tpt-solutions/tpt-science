//! Eley–Rideal surface mechanism: adsorbed species reacting with gas-phase molecules.

use super::*;

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
