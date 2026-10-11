//! Umbrella crate for the tpt-science simulation substrate.
//!
//! Depend on `tpt-sci` once and reach every domain crate through a short module
//! path, instead of listing each `tpt-sci-*` crate separately:
//!
//! ```
//! use tpt_sci::ode::{Method, OdeProblem};
//!
//! // dy/dt = -y  ->  y(1) = e^{-1}
//! let prob = OdeProblem::new(|_t, y, dy| dy[0] = -y[0], vec![1.0], 0.0).unwrap();
//! let y = prob.solve(Method::Tsit45, 1.0).unwrap();
//! assert!((y[0] - (-1.0f64).exp()).abs() < 1e-5);
//! ```
//!
//! Each domain is behind a Cargo feature of the same name (`ode`, `grid`,
//! `sim-core`, …). The default features cover the ODE, grid and orchestration
//! crates; enable `full` for everything.
//!
//! | Module | Crate | Feature |
//! |--------|-------|---------|
//! | `ode` | `tpt-sci-ode` | `ode` (default) |
//! | `grid` | `tpt-sci-grid` | `grid` (default) |
//! | `sim_core` | `tpt-sci-sim-core` | `sim-core` (default) |
//! | `reaction_network` | `tpt-sci-reaction-network` | `reaction-network` |
//! | `ppl` | `tpt-sci-ppl` | `ppl` |
//! | `image` | `tpt-sci-image` | `image` |
//! | `physics_rigid` | `tpt-sci-physics-rigid` | `physics-rigid` |
//! | `quantum` | `tpt-sci-quantum` | `quantum` |
//! | `astro` | `tpt-sci-astro` | `astro` |
//! | `md` | `tpt-sci-md` | `md` |
//! | `dft_classical` | `tpt-sci-dft-classical` | `dft-classical` |
//! | `kinetics` | `tpt-sci-kinetics` | `kinetics` |
//! | `cfd_core` | `tpt-sci-cfd-core` | `cfd-core` |
//! | `hemodynamics` | `tpt-sci-hemodynamics` | `hemodynamics` |
//! | `electrophys` | `tpt-sci-electrophys` | `electrophys` |
//! | `climate` | `tpt-sci-climate` | `climate` |
//! | `ocean` | `tpt-sci-ocean` | `ocean` |
//! | `dft_electronic` | `tpt-sci-dft-electronic` | `dft-electronic` |

/// Re-export of [`tpt-sci-astro`](https://docs.rs/tpt-sci-astro).
#[cfg(feature = "astro")]
pub use tpt_sci_astro as astro;
/// Re-export of [`tpt-sci-cfd-core`](https://docs.rs/tpt-sci-cfd-core).
#[cfg(feature = "cfd-core")]
pub use tpt_sci_cfd_core as cfd_core;
/// Re-export of [`tpt-sci-climate`](https://docs.rs/tpt-sci-climate).
#[cfg(feature = "climate")]
pub use tpt_sci_climate as climate;
/// Re-export of [`tpt-sci-dft-classical`](https://docs.rs/tpt-sci-dft-classical).
#[cfg(feature = "dft-classical")]
pub use tpt_sci_dft_classical as dft_classical;
/// Re-export of [`tpt-sci-dft-electronic`](https://docs.rs/tpt-sci-dft-electronic).
#[cfg(feature = "dft-electronic")]
pub use tpt_sci_dft_electronic as dft_electronic;
/// Re-export of [`tpt-sci-electrophys`](https://docs.rs/tpt-sci-electrophys).
#[cfg(feature = "electrophys")]
pub use tpt_sci_electrophys as electrophys;
/// Re-export of [`tpt-sci-grid`](https://docs.rs/tpt-sci-grid).
#[cfg(feature = "grid")]
pub use tpt_sci_grid as grid;
/// Re-export of [`tpt-sci-hemodynamics`](https://docs.rs/tpt-sci-hemodynamics).
#[cfg(feature = "hemodynamics")]
pub use tpt_sci_hemodynamics as hemodynamics;
/// Re-export of [`tpt-sci-image`](https://docs.rs/tpt-sci-image).
#[cfg(feature = "image")]
pub use tpt_sci_image as image;
/// Re-export of [`tpt-sci-kinetics`](https://docs.rs/tpt-sci-kinetics).
#[cfg(feature = "kinetics")]
pub use tpt_sci_kinetics as kinetics;
/// Re-export of [`tpt-sci-md`](https://docs.rs/tpt-sci-md).
#[cfg(feature = "md")]
pub use tpt_sci_md as md;
/// Re-export of [`tpt-sci-ocean`](https://docs.rs/tpt-sci-ocean).
#[cfg(feature = "ocean")]
pub use tpt_sci_ocean as ocean;
/// Re-export of [`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).
#[cfg(feature = "ode")]
pub use tpt_sci_ode as ode;
/// Re-export of [`tpt-sci-physics-rigid`](https://docs.rs/tpt-sci-physics-rigid).
#[cfg(feature = "physics-rigid")]
pub use tpt_sci_physics_rigid as physics_rigid;
/// Re-export of [`tpt-sci-ppl`](https://docs.rs/tpt-sci-ppl).
#[cfg(feature = "ppl")]
pub use tpt_sci_ppl as ppl;
/// Re-export of [`tpt-sci-quantum`](https://docs.rs/tpt-sci-quantum).
#[cfg(feature = "quantum")]
pub use tpt_sci_quantum as quantum;
/// Re-export of [`tpt-sci-reaction-network`](https://docs.rs/tpt-sci-reaction-network).
#[cfg(feature = "reaction-network")]
pub use tpt_sci_reaction_network as reaction_network;
/// Re-export of [`tpt-sci-sim-core`](https://docs.rs/tpt-sci-sim-core).
#[cfg(feature = "sim-core")]
pub use tpt_sci_sim_core as sim_core;

/// Typed SI quantities and the physical constants shared across the crates.
pub use tpt_sci_units as units;
