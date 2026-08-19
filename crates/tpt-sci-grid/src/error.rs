use thiserror::Error;

/// Errors raised while constructing grids or assembling operators.
#[derive(Debug, Error)]
pub enum GridError {
    /// A grid needs at least this many points.
    #[error("grid needs at least {0} points, got {1}")]
    TooFewPoints(usize, usize),

    /// The domain endpoints were not strictly ordered.
    #[error("invalid domain: x0 ({0}) >= x1 ({1})")]
    InvalidDomain(f64, f64),
}
