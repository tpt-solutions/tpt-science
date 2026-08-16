//! Error types for `tpt-sci-astro`.

use thiserror::Error;

/// Errors produced by the astrodynamics primitives.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AstroError {
    /// An input failed a physical/mathematical invariant (e.g. `a ≤ 0`,
    /// `e ∉ [0, 1)`, non-finite angle, or `μ ≤ 0`).
    #[error("invalid orbital elements: {0}")]
    InvalidElements(String),
    /// The input geometry was degenerate and could not be transformed
    /// (e.g. zero position magnitude, zero angular momentum, or a
    /// non-elliptical orbit).
    #[error("degenerate geometry: {0}")]
    DegenerateGeometry(String),
}
