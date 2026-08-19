//! Error types for `tpt-sci-hemodynamics`.

use thiserror::Error;

/// Errors raised by the hemodynamics model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum HemodynamicsError {
    /// A vessel was invalid.
    #[error("invalid vessel: {0}")]
    InvalidVessel(String),
    /// A network was invalid.
    #[error("invalid network: {0}")]
    InvalidNetwork(String),
}
