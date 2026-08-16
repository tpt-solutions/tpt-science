//! # tpt-sci-sim-core
//!
//! Multi-scale simulation orchestration for the `tpt-science` pillar: the
//! layer *above* [`tpt_sci_ode`] and [`tpt_sci_grid`] that a "multi-scale
//! computational platform" (the motivating use case for `tpt-soma` and
//! `tpt-cerebrum`) actually needs and that neither solver crate provides
//! alone.
//!
//! It provides:
//!
//! - **Time-stepping across heterogeneous sub-models.** Each [`SubModel`]
//!   advances on its own internal time scale (`max_step`) while a
//!   [`Simulation`] drives them all forward to a shared target time, taking
//!   the largest sub-step that no model would overshoot.
//! - **Cross-scale coupling.** A [`Coupling`] maps the state of one model onto
//!   the input buffer of another after every sub-step, so a fast model's
//!   output can drive a slow one (or vice versa).
//! - **Checkpointing.** [`Simulation::snapshot`] / [`Simulation::restore`]
//!   capture and restore every model's state and the global clock, for
//!   resumable / reproducible runs.
//!
//! ## Example
//!
//! ```rust
//! use tpt_sci_sim_core::{OdeSubModel, Simulation, SubModel};
//!
//! // A single fast exponential-decay sub-model: dy/dt = -y, y(0) = 1.
//! let decay = OdeSubModel::new(
//!     "decay",
//!     |_t, y, dydt| dydt[0] = -y[0],
//!     vec![1.0],
//!     0.0,
//! );
//! let mut sim = Simulation::new();
//! sim.add_model(decay).unwrap();
//! sim.step_until(1.0).unwrap();
//! let y = sim.model("decay").unwrap().state()[0];
//! assert!((y - std::f64::consts::E.recip()).abs() < 1e-3);
//! ```

pub mod coupling;
pub mod diffusion_model;
pub mod error;
pub mod ode_model;
pub mod sim;
pub mod submodel;

pub use coupling::{Coupling, CouplingFn};
pub use diffusion_model::DiffusionSubModel;
pub use error::SimError;
pub use ode_model::OdeSubModel;
pub use sim::{Checkpoint, Simulation};
pub use submodel::SubModel;
