//! Error types for `tpt-sci-cfd-core`.

use thiserror::Error;

/// Errors raised by the CFD core solver.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CfdError {
    /// The grid configuration was invalid.
    #[error("invalid grid: {0}")]
    InvalidGrid(String),
}
