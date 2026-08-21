//! Error types for `tpt-sci-cfd-core`.

use thiserror::Error;

/// Errors raised by the CFD core solver.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CfdError {
    /// The grid configuration was invalid.
    #[error("invalid grid: {0}")]
    InvalidGrid(String),
    /// The unstructured mesh was invalid (bad connectivity, degenerate cell,
    /// out-of-range node index, …).
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),
    /// A solver/model parameter was outside its admissible range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// A supplied field had the wrong length for the grid/mesh it was used with.
    #[error("dimension mismatch: expected {expected} entries, got {actual}")]
    DimensionMismatch {
        /// Length required by the grid/mesh.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// The solution became non-finite (the scheme went unstable).
    #[error("solver diverged: {0}")]
    Diverged(String),
    /// An iterative solver hit its iteration cap before reaching the tolerance.
    #[error("solver did not converge: residual {residual:e} after {iterations} iterations")]
    NotConverged {
        /// Residual reached when the iteration cap was hit.
        residual: f64,
        /// Number of iterations performed.
        iterations: usize,
    },
}
