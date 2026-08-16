use tpt_sci_ode::{OdeProblem, OdeProblemBuilder};

use crate::error::ReactionNetworkError;
use crate::model::ReactionSystem;

impl ReactionSystem {
    /// Build an [`OdeProblemBuilder`] for `dy/dt = S · r(y)` starting from
    /// `y0` at time `t0`. The returned builder can be tuned (tolerances, initial
    /// step) before being finalised into a solvable [`OdeProblem`].
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::StateDimension`] if `y0` does not have
    /// one entry per species, or wraps an [`tpt_sci_ode::OdeError`] if the
    /// underlying problem is invalid.
    pub fn ode_builder(
        &self,
        y0: &[f64],
        t0: f64,
    ) -> Result<OdeProblemBuilder, ReactionNetworkError> {
        if y0.len() != self.n_species() {
            return Err(ReactionNetworkError::StateDimension {
                got: y0.len(),
                expected: self.n_species(),
            });
        }
        let sys = self.clone();
        let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            sys.eval_rhs(y, dydt);
        };
        Ok(OdeProblemBuilder::new(rhs, y0.to_vec(), t0))
    }

    /// Build a ready-to-solve [`OdeProblem`] for `dy/dt = S · r(y)` with default
    /// tolerances, starting from `y0` at time `t0`.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::StateDimension`] if `y0` does not have
    /// one entry per species, or wraps an [`tpt_sci_ode::OdeError`] if the
    /// underlying problem is invalid.
    pub fn to_ode_problem(&self, y0: &[f64], t0: f64) -> Result<OdeProblem, ReactionNetworkError> {
        Ok(self.ode_builder(y0, t0)?.build()?)
    }
}
