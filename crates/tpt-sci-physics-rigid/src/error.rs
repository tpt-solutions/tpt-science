//! Error types for the rigid-body physics crate.

use thiserror::Error;

/// Errors produced by the physics world and body construction.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PhysicsError {
    /// A [`crate::Body`] was constructed with invalid parameters.
    #[error("invalid body: {0}")]
    InvalidBody(String),
    /// An attempt was made to add a body whose `id` already exists in the
    /// [`crate::World`].
    #[error("duplicate body id {0}")]
    DuplicateId(usize),
    /// Two vectors (gravity, bounds, or a body) disagreed on dimensionality.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension that the world expects.
        expected: usize,
        /// The dimension that was supplied.
        got: usize,
    },
}
