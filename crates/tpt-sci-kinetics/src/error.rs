//! Error types for `tpt-sci-kinetics`.

use thiserror::Error;

/// Errors raised by the kinetics engine.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum KineticsError {
    /// An Arrhenius rate was invalid.
    #[error("invalid rate: {0}")]
    InvalidRate(String),
    /// Surface-coverage computation failed.
    #[error("coverage error: {0}")]
    CoverageError(String),
}
