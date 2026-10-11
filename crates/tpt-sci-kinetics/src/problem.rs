//! Kinetics problems: assembling rate laws into an ODE right-hand side.

use super::*;

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
