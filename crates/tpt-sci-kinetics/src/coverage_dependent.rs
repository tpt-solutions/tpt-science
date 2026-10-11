//! Arrhenius rates whose activation energy depends on surface coverage.

use super::*;

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
