//! Errors raised while building or running a probabilistic-programming model.

use thiserror::Error;

/// Errors raised by [`mod@crate::nuts`] and the [`crate::Model`] DSL.
#[derive(Debug, Error)]
pub enum PplError {
    /// A model/parameter had the wrong number of dimensions for its data.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// An initial point or parameter produced a non-finite log-density (e.g. it
    /// lay outside the prior support).
    #[error("non-finite log-density at initial point (likely outside prior support)")]
    NonFiniteInitial,

    /// A sampler option was invalid (e.g. zero samples or non-positive step
    /// size).
    #[error("invalid sampler option: {0}")]
    InvalidOption(String),
}
