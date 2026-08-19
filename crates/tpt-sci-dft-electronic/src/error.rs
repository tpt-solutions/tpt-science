//! Error types for `tpt-sci-dft-electronic`.

use thiserror::Error;

/// Errors raised by the electronic-structure DFT solver.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DftError {
    /// The 1-D grid was invalid.
    #[error("invalid grid: {0}")]
    InvalidGrid(String),
    /// The Kohn–Sham setup was invalid.
    #[error("invalid setup: {0}")]
    InvalidSetup(String),
}
