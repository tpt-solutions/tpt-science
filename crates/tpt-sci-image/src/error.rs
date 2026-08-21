//! Errors for the `tpt-sci-image` crate.
//!
//! Every public reconstruction entry point validates its inputs and returns a
//! [`Result`], matching the `Result`-returning convention of the other
//! `tpt-science` crates. This replaces the previous silent zero-padding of
//! malformed inputs, which could mask caller bugs.

use thiserror::Error;

/// Errors returned by the image reconstruction API.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ImageError {
    /// The input image had zero rows or columns.
    #[error("image must be non-empty, got {nrows}x{ncols}")]
    EmptyImage {
        /// Number of rows in the offending image.
        nrows: usize,
        /// Number of columns in the offending image.
        ncols: usize,
    },

    /// The angle list supplied for a reconstruction was empty.
    #[error("angle list must be non-empty")]
    EmptyAngles,

    /// The sinogram row count did not match the number of projection angles.
    #[error("sinogram has {sino_rows} rows but {n_angles} angles were supplied")]
    AngleCountMismatch {
        /// Number of rows in the supplied sinogram.
        sino_rows: usize,
        /// Number of projection angles supplied.
        n_angles: usize,
    },

    /// A sinogram dimension did not match the expected volume geometry
    /// (`n_bins = max(nx, ny)`, or `n_z`).
    #[error("expected dimension {expected} but got {got}")]
    DimensionMismatch {
        /// Expected extent (voxels or detector bins).
        expected: usize,
        /// Supplied extent.
        got: usize,
    },

    /// A [`crate::cone_beam::ConeBeamGeometry`] parameter was invalid: a
    /// non-positive source/detector distance or pixel spacing, or a zero
    /// detector pixel count / angle count.
    #[error("invalid cone-beam geometry: {reason}")]
    InvalidGeometry {
        /// Human-readable description of what was invalid.
        reason: String,
    },
}
