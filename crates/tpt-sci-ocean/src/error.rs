//! Error types for `tpt-sci-ocean`.

use thiserror::Error;

/// Errors raised by the ocean model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OceanError {
    /// The model configuration was invalid.
    #[error("invalid model: {0}")]
    InvalidModel(String),
}
