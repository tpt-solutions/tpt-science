//! Error types for `tpt-sci-dft-classical`.

use thiserror::Error;

/// Errors raised by the classical-DFT wrapper.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DftError {
    /// A functional/parameter could not be constructed.
    #[error("functional error: {0}")]
    Functional(String),
    /// A DFT profile could not be built or solved.
    #[error("profile error: {0}")]
    Profile(String),
}
