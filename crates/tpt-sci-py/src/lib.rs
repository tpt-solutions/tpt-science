//! Python bindings for tpt-science.
//!
//! Build a wheel with `maturin build --release` (or `maturin develop` for an
//! editable install). Then, from Python:
//!
//! ```python
//! import tpt_sci
//! csv = tpt_sci.run_spec('{"network": "k, A --> B", "parameters": {"k": 1.0}, '
//!                        '"initial": {"A": 1.0}, "t_final": 1.0}')
//! tpt_sci.ebm_equilibrium_temperature(560.0)
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Run a reaction-network spec given as JSON and return its trajectory as CSV.
///
/// Raises `ValueError` if the spec is malformed or the integration fails.
#[pyfunction]
fn run_spec(json: &str) -> PyResult<String> {
    tpt_sci_cli::run_to_csv(json).map_err(|e| PyValueError::new_err(format!("{e:#}")))
}

/// Equilibrium surface temperature (K) of the 0-D energy-balance model at a CO₂
/// concentration in ppm.
///
/// Raises `ValueError` if the concentration is not positive.
#[pyfunction]
fn ebm_equilibrium_temperature(co2_ppm: f64) -> PyResult<f64> {
    // Heat capacity 1.0, albedo 0.3, emissivity 0.61 (see `EnergyBalanceModel`).
    let model = tpt_sci_climate::EnergyBalanceModel::new(1.0, 0.3, 0.61, co2_ppm)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(model.equilibrium_temperature())
}

#[pymodule]
fn tpt_sci(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_spec, m)?)?;
    m.add_function(wrap_pyfunction!(ebm_equilibrium_temperature, m)?)?;
    Ok(())
}
