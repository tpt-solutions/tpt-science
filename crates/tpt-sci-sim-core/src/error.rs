use thiserror::Error;

/// Errors raised while orchestrating a [`Simulation`](crate::Simulation) or
/// advancing a [`SubModel`](crate::SubModel).
#[derive(Debug, Error)]
pub enum SimError {
    /// A sub-model was registered with an id that is already in use.
    #[error("duplicate sub-model id: {0}")]
    DuplicateModel(String),

    /// A referenced sub-model id does not exist in the simulation.
    #[error("unknown sub-model id: {0}")]
    UnknownModel(String),

    /// The target time for `step_until` was not strictly greater than the
    /// current global time.
    #[error("step_until target {0} is not ahead of current time {1}")]
    NonAdvancingStep(f64, f64),

    /// A sub-model failed while advancing its internal state.
    #[error("sub-model '{0}' failed to advance: {1}")]
    Advance(String, String),

    /// A coupling referenced a model that does not expose an input buffer.
    #[error("coupling target '{0}' exposes no input buffer")]
    CouplingTargetNoInput(String),
}
