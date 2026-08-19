//! # tpt-sci-dft-classical
//!
//! **Classical / soft-matter density functional theory (DFT)** for the
//! `tpt-science` pillar, provided by wrapping the [`feos`] framework
//! ([`feos_dft`], MIT OR Apache-2.0). This crate is a thin, opinionated layer
//! over `feos` so downstream science consumers get a stable, TPT-flavoured
//! entry point without reaching into the full `feos` API:
//!
//! * Re-exports the `feos` DFT pieces ([`feos_dft`]) — profiles, solvers,
//!   convolutions, geometries.
//! * Wraps the PC-SAFT Helmholtz functional ([`feos::pcsaft::PcSaft`]) behind a
//!   small [`ClassicalDft`] builder that solves a **1-D density profile** for a
//!   pure component given temperature and bulk density.
//!
//! The heavy lifting (functional derivatives, Picard / Anderson mixing, FFT
//! convolutions) is done by `feos` itself; this crate only assembles the
//! inputs and exposes a tidy result.
//!
//! # Example
//!
//! ```ignore
//! use tpt_sci_dft_classical::ClassicalDft;
//! use feos::pcsaft::{PcSaft, PcSaftParameters};
//! use feos_core::parameter::IdentifierOption;
//!
//! // Build a PC-SAFT functional for a single pure component from its
//! // parameters, then solve a planar density profile at (T, rho_bulk).
//! let parameters = PcSaftParameters::from_json(
//!     vec!["propane"], "../../parameters/pcsaft/esper2023.json",
//!     None, IdentifierOption::Name,
//! ).unwrap();
//! let dft = ClassicalDft::with_functional(PcSaft::new(parameters));
//! ```
#![forbid(unsafe_code)]

mod error;
mod profile;

pub use error::DftError;
pub use profile::ClassicalDft;

/// Re-export of the wrapped `feos` framework so consumers can use the full
/// underlying DFT machinery directly if they need to.
pub use feos;
/// Re-export of the wrapped `feos_dft` crate (profiles, solvers, geometries).
pub use feos_dft;
