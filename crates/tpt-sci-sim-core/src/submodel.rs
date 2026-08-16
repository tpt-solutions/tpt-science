/// A sub-model in a multi-scale [`Simulation`](crate::Simulation).
///
/// A sub-model owns some internal state and knows how to advance it by a fixed
/// internal step `dt`. Sub-models in a single simulation may run at very
/// different time scales (a fast chemical kinetics ODE vs. a slow diffusion
/// field), which is exactly the cross-scale situation the orchestrator is
/// designed for.
pub trait SubModel {
    /// Stable identifier, unique within a [`Simulation`](crate::Simulation).
    fn id(&self) -> &str;

    /// The model's current internal simulation time.
    fn time(&self) -> f64;

    /// The largest internal step this model can safely take in one `advance`
    /// call. The orchestrator never asks a model to advance by more than this.
    fn max_step(&self) -> f64;

    /// Advance the model's internal state by exactly `dt` (with
    /// `0 < dt <= max_step`).
    ///
    /// # Errors
    ///
    /// Returns an error if the integration step fails (e.g. the underlying
    /// solver errors or a model-specific invariant is violated).
    fn advance(&mut self, dt: f64) -> Result<(), crate::SimError>;

    /// Read-only view of the model's current state vector (used as coupling
    /// output).
    fn state(&self) -> &[f64];

    /// Mutable view of the model's input buffer, if it accepts coupled inputs.
    /// Returns `None` for models that are driven purely internally. The
    /// orchestrator writes cross-scale inputs here after every sub-step.
    fn input_mut(&mut self) -> Option<&mut [f64]> {
        None
    }

    /// Restore the model's state vector and internal time from a checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if `state`'s length does not match the model's expected
    /// state length.
    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), crate::SimError>;
}
