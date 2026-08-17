//! # tpt-sci-grid
//!
//! Structured finite-difference grids and stencils for the `tpt-science`
//! pillar: uniform 1-D, 2-D and 3-D tensor-product grids, finite-difference
//! stencil coefficients, and assembly of discrete PDE operators (Laplacians)
//! on those grids.
//!
//!
//! This crate is built from scratch (no wrap target — see `spec.txt`). It is
//! motivated by the compartmental-ODE / structured-grid spatial-model needs of
//! `tpt-soma` and `tpt-cerebrum` (reaction–diffusion, cable-equation /
//! cortical-sheet style models).
//!
//! The assembled operators are returned as [`DMatrix`] / [`DVector`] from the
//! `tpt-math` dense linear-algebra substrate ([`linalg`]), which is an in-house,
//! dual-licensed (MIT OR Apache-2.0) backend with no `nalgebra`/`faer` license
//! exposure, per the workspace policy in ADR 0007.
//!
//! ## Example
//!
//! ```rust
//! use tpt_sci_grid::{laplacian_1d, UniformGrid1D, Boundary};
//!
//! let g = UniformGrid1D::new(11, 0.0, 1.0).unwrap();
//! let l = laplacian_1d(&g, Boundary::Dirichlet);
//! assert_eq!(l.nrows(), 11);
//! assert_eq!(l.ncols(), 11);
//! ```

pub mod error;
pub mod grid;
pub mod operator;
#[cfg(feature = "sparse")]
pub mod sparse;
pub mod stencil;

pub use error::GridError;
pub use grid::{Boundary, UniformGrid1D, UniformGrid2D, UniformGrid3D, linspace};
pub use operator::{derivative_1d, kron, laplacian_1d, laplacian_2d, laplacian_3d};
pub use stencil::Stencil;

#[cfg(feature = "sparse")]
pub use sparse::{
    CsrMatrix, diffuse_step, laplacian_1d_sparse, laplacian_2d_sparse, laplacian_3d_sparse,
};

/// The `tpt-math` linear-algebra substrate this crate builds on.
pub use tpt_math_linalg as linalg;

/// Dense matrix / vector types re-exported from the `tpt-math` substrate so
/// callers can consume assembled operators without a separate dependency.
pub use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};
