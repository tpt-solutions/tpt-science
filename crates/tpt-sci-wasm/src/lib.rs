//! WebAssembly bindings for tpt-science.
//!
//! Build for the browser with `wasm-pack build --target web` (or
//! `cargo build --target wasm32-unknown-unknown -p tpt-sci-wasm`). Functions
//! return `Result`, which surfaces as a thrown JavaScript `Error`.
//!
//! Exports:
//!
//! * [`run_spec`]: run a declarative reaction-network spec (see `tpt-sci-cli`)
//!   and return the trajectory as CSV.
//! * [`ebm_equilibrium_temperature`]: the equilibrium surface temperature of
//!   the 0-D energy-balance model at a CO₂ concentration.

use wasm_bindgen::prelude::*;

/// Run a reaction-network spec given as JSON and return its trajectory as CSV.
///
/// # Errors
///
/// Throws a JavaScript error if the spec is malformed or the integration fails.
#[wasm_bindgen]
pub fn run_spec(json: &str) -> Result<String, JsError> {
    tpt_sci_cli::run_to_csv(json).map_err(|e| JsError::new(&format!("{e:#}")))
}

/// Equilibrium surface temperature (K) of the 0-D energy-balance model at
/// `co2_ppm`, with the Earth-like parameters used in the climate examples.
///
/// # Errors
///
/// Throws a JavaScript error if `co2_ppm` is not a positive number.
#[wasm_bindgen]
pub fn ebm_equilibrium_temperature(co2_ppm: f64) -> Result<f64, JsError> {
    // Heat capacity 1.0, albedo 0.3, emissivity 0.61 (see `EnergyBalanceModel`).
    let model = tpt_sci_climate::EnergyBalanceModel::new(1.0, 0.3, 0.61, co2_ppm)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(model.equilibrium_temperature())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_runner_produces_csv() {
        let csv = tpt_sci_cli::run_to_csv(
            r#"{ "network": "k, A --> B", "parameters": {"k": 1.0},
                 "initial": {"A": 1.0}, "t_final": 1.0, "points": 2 }"#,
        )
        .unwrap();
        assert!(csv.starts_with("t,A,B\n"));
    }

    #[test]
    fn doubling_co2_warms_the_model() {
        let t1 = ebm_equilibrium_temperature(280.0).unwrap();
        let t2 = ebm_equilibrium_temperature(560.0).unwrap();
        assert!(t2 > t1);
    }
}
