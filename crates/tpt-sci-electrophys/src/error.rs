//! Error types for `tpt-sci-electrophys`.

use thiserror::Error;

/// Errors raised by the electrophysiology model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ElectrophysError {
    /// The tissue configuration was invalid.
    #[error("invalid tissue: {0}")]
    InvalidTissue(String),
}
