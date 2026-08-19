//! Error types for `tpt-sci-climate`.

use thiserror::Error;

/// Errors raised by the climate model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ClimateError {
    /// The model configuration was invalid.
    #[error("invalid model: {0}")]
    InvalidModel(String),
}
