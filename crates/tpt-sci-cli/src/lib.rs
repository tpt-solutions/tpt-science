//! Declarative runs of reaction networks.
//!
//! A run is described by a small JSON spec: the network in the textual
//! chemical-notation DSL, its rate constants, the initial amounts, the time
//! window, and the solver. [`run_to_csv`] integrates it and returns the
//! trajectory as CSV, which the `tpt-sci` binary writes to a file or stdout.
//!
//! ```json
//! {
//!   "network": "beta, S + I --> 2 I\ngamma, I --> R",
//!   "parameters": { "beta": 0.0004, "gamma": 0.2 },
//!   "initial": { "S": 990, "I": 10, "R": 0 },
//!   "t_final": 60,
//!   "points": 60
//! }
//! ```
//!
//! Rates follow mass-action kinetics on molecule counts: a bimolecular rate
//! constant is per count, so `beta = 0.0004` with 1000 individuals is the
//! per-capita rate 0.4.
//!
//! Optional fields: `t0` (default `0`), `method` (`"bdf"` by default; also
//! `"tsit45"`, `"trbdf2"`, `"esdirk34"`), `rtol` (`1e-6`) and `atol` (`1e-9`).

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use tpt_sci_ode::{Method, Trajectory};
use tpt_sci_reaction_network::ReactionNetwork;

/// A declarative reaction-network run. Unknown fields are rejected so that a
/// misspelt option fails loudly instead of being ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// The network, in the textual DSL (`rate, reactants --> products` per line).
    pub network: String,
    /// Rate constants by name. Each must be declared in `network`.
    #[serde(default)]
    pub parameters: BTreeMap<String, f64>,
    /// Initial amount of each species. Species not listed start at zero.
    pub initial: BTreeMap<String, f64>,
    /// Start time.
    #[serde(default)]
    pub t0: f64,
    /// End time. Must be after `t0`.
    pub t_final: f64,
    /// Number of evenly spaced output times in `(t0, t_final]`.
    #[serde(default = "default_points")]
    pub points: usize,
    /// Integrator: `"tsit45"`, `"trbdf2"`, `"esdirk34"` or `"bdf"`.
    #[serde(default = "default_method")]
    pub method: String,
    /// Relative tolerance.
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    /// Absolute tolerance.
    #[serde(default = "default_atol")]
    pub atol: f64,
}

fn default_points() -> usize {
    100
}
fn default_method() -> String {
    "bdf".to_string()
}
fn default_rtol() -> f64 {
    1e-6
}
fn default_atol() -> f64 {
    1e-9
}

/// Map a method name from a spec to a solver.
///
/// # Errors
///
/// Returns an error naming the accepted methods if `name` is not one of them.
pub fn parse_method(name: &str) -> Result<Method> {
    match name {
        "tsit45" => Ok(Method::Tsit45),
        "trbdf2" => Ok(Method::TrBdf2),
        "esdirk34" => Ok(Method::Esdirk34),
        "bdf" => Ok(Method::Bdf),
        other => bail!("unknown method {other:?}; expected tsit45, trbdf2, esdirk34 or bdf"),
    }
}

/// Integrate a spec and return the trajectory.
///
/// # Errors
///
/// Returns an error if the network does not parse, a parameter or species is
/// undeclared, the time window or tolerances are invalid, or the solver fails.
pub fn run(spec: &Spec) -> Result<Trajectory> {
    if spec.t_final.is_nan() || spec.t_final <= spec.t0 {
        bail!("t_final ({}) must be after t0 ({})", spec.t_final, spec.t0);
    }
    if spec.points == 0 {
        bail!("points must be at least 1");
    }
    let method = parse_method(&spec.method)?;

    let mut sys = ReactionNetwork::from_dsl(&spec.network).context("parsing the network")?;
    for (name, value) in &spec.parameters {
        sys.set_parameter(name, *value)
            .with_context(|| format!("setting parameter {name:?}"))?;
    }
    let amounts: Vec<(&str, f64)> = spec.initial.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let y0 = sys
        .initial_state(&amounts)
        .context("building the initial state")?;

    let t_eval: Vec<f64> = (1..=spec.points)
        .map(|k| spec.t0 + (spec.t_final - spec.t0) * k as f64 / spec.points as f64)
        .collect();
    let names: Vec<String> = sys.species_names().to_vec();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let problem = sys
        .ode_builder(&y0, spec.t0)
        .context("building the ODE problem")?
        .rtol(spec.rtol)
        .atol(spec.atol)
        .build()
        .map_err(|e| anyhow!("invalid tolerances: {e}"))?;
    problem
        .solve_trajectory(method, &t_eval, &name_refs)
        .context("integrating the network")
}

/// Parse a JSON spec, run it, and return the trajectory as CSV.
///
/// # Errors
///
/// Returns an error for malformed JSON, or for anything [`run`] rejects.
pub fn run_to_csv(json: &str) -> Result<String> {
    let spec: Spec = serde_json::from_str(json).context("parsing the JSON spec")?;
    Ok(run(&spec)?.to_csv())
}
