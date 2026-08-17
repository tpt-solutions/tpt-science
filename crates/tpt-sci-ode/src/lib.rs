//! # tpt-sci-ode
//!
//! ODE/DAE solving substrate for the `tpt-science` pillar.
//!
//! This crate implements its own ODE integrators **from scratch** — no
//! [`diffsol`](https://crates.io/crates/diffsol), `nalgebra`, or `faer` in the
//! shipped dependency graph — so the whole crate is TPT-owned code under
//! `MIT OR Apache-2.0`. It builds on the dual-licensed `tpt-math` dense linear
//! algebra for the implicit-method linear solves.
//!
//! Four methods are provided (see [`Method`]):
//! * [`Method::Tsit45`] — explicit Runge–Kutta 4(5), non-stiff.
//! * [`Method::TrBdf2`] — 2-stage SDIRK (TR-BDF2), A-stable, stiff.
//! * [`Method::Esdirk34`] — 4-stage ESDIRK order 3(4), A-/L-stable, stiff.
//! * [`Method::Bdf`] — variable-order (1–5) backward differentiation, stiff.
//!
//! All use a shared adaptive-step driver with Hermite dense output, so
//! [`OdeProblem::solve_dense`] returns exact states at the requested times.
//! `diffsol` is retained only as an optional, dev-only verification oracle
//! (feature `verify-diffsol`, excluded from license scanning).
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
pub mod linalg;
pub mod problem;
pub mod solver;

pub use error::OdeError;
pub use problem::{OdeProblem, OdeProblemBuilder, Rhs};
pub use solver::Method;

// Re-export the numeric scalar trait from the tpt-math substrate that this
// crate depends on, so downstream science crates share one scalar vocabulary.
pub use tpt_math_numeric::Scalar;
