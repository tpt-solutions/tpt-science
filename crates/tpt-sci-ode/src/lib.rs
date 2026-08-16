//! # tpt-sci-ode
//!
//! ODE/DAE solving substrate for the `tpt-science` pillar.
//!
//! This crate is a thin, ergonomic wrapper over
//! [`diffsol`](https://crates.io/crates/diffsol) (JOSS 2026, Enzyme-based
//! autodiff, LLVM/Cranelift JIT) — see `spec.txt` for the rationale: the
//! ecosystem gap is well-covered by a maintained, dual-licensed crate, so we
//! wrap rather than reimplement. Every crate in this pillar is dual-licensed
//! `MIT OR Apache-2.0`.
//!
//! ## Example
//!
//! ```rust
//! use tpt_sci_ode::{OdeProblem, Method};
//!
//! // dy/dt = -y,  y(0) = 1  ->  y(1) = e^-1
//! let prob = OdeProblem::new(
//!     |_t, y, dydt| { dydt[0] = -y[0]; },
//!     vec![1.0],
//!     0.0,
//! ).unwrap();
//! let y = prob.solve(Method::Bdf, 1.0).unwrap();
//! assert!((y[0] - std::f64::consts::E.recip()).abs() < 1e-5);
//! ```

pub mod error;
pub mod problem;
pub mod solver;

pub use error::OdeError;
pub use problem::{OdeProblem, OdeProblemBuilder, Rhs};
pub use solver::Method;

// Re-export the numeric scalar trait from the tpt-math substrate that this
// crate depends on, so downstream science crates share one scalar vocabulary.
pub use tpt_math_numeric::Scalar;
