use thiserror::Error;

/// Errors raised while building or solving an [`OdeProblem`](crate::OdeProblem).
#[derive(Debug, Error)]
pub enum OdeError {
    /// Wraps an error returned by the underlying `diffsol` solver.
    #[error("diffsol solver error: {0}")]
    Diffsol(String),

    /// The problem definition was invalid (e.g. empty initial state).
    #[error("invalid ODE problem: {0}")]
    Invalid(String),
}

impl From<diffsol::DiffsolError> for OdeError {
    fn from(e: diffsol::DiffsolError) -> Self {
        OdeError::Diffsol(e.to_string())
    }
}
