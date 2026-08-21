//! Error types for `tpt-sci-ocean`.

use thiserror::Error;

/// Errors raised by the ocean model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OceanError {
    /// The model configuration was invalid.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// Two arrays/matrices that must agree in dimension did not.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    /// A dense linear-algebra routine failed (non-finite or non-SPD input).
    #[error("linear algebra failure: {0}")]
    LinAlg(String),
}
