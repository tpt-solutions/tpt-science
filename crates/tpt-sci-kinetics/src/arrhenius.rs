//! Arrhenius rate constants, `k(T) = A·exp(−Eₐ / (R·T))`.

use super::*;

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

impl ArrheniusRate {
    /// Typed constructor: pre-exponential factor `a` and activation energy `ea`.
    ///
    /// # Errors
    ///
    /// As [`ArrheniusRate::new`].
    pub fn with_activation_energy(
        a: f64,
        ea: tpt_sci_units::JoulePerMole,
    ) -> Result<Self, KineticsError> {
        Self::new(a, ea.value())
    }

    /// Rate constant at a typed temperature. Equivalent to
    /// [`ArrheniusRate::rate_constant`] with the unit carried in the type.
    #[must_use]
    pub fn rate_constant_at(&self, t: tpt_sci_units::Kelvin) -> f64 {
        self.rate_constant(t.value())
    }
}
