//! Error types for `tpt-sci-md`.

use thiserror::Error;

/// Errors raised by the molecular-dynamics engine.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MdError {
    /// A particle failed validation on construction.
    #[error("invalid particle: {0}")]
    InvalidParticle(String),
    /// The integrator configuration was invalid.
    #[error("invalid integrator: {0}")]
    InvalidIntegrator(String),
    /// The radial-distribution-function computation failed.
    #[error("rdf error: {0}")]
    RdfError(String),
    /// The Ewald summation configuration was invalid.
    #[error("invalid ewald parameters: {0}")]
    InvalidEwald(String),
    /// The EAM potential configuration was invalid.
    #[error("invalid eam parameters: {0}")]
    InvalidEam(String),
    /// A SHAKE bond-constraint specification or solve was invalid.
    #[error("shake error: {0}")]
    ShakeError(String),
    /// A cell-list configuration was invalid.
    #[error("invalid cell list: {0}")]
    InvalidCellList(String),
}
