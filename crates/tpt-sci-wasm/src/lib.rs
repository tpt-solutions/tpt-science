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

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_cfd_core::{Boundary, CollocatedGrid, Step};
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

/// Two overlapping discs on an `n × n` grid, the phantom of the CT tutorial.
/// Returned row-major (`out[i * n + j]`). The discs keep their size relative to
/// the 64-pixel tutorial grid, whatever `n` is.
#[wasm_bindgen]
pub fn ct_phantom(n: usize) -> Vec<f64> {
    let scale = 64.0 / n.max(1) as f64;
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = (i as f64 - n as f64 / 2.0) * scale;
            let y = (j as f64 - n as f64 / 2.0) * scale;
            let in_a = x * x + y * y < 36.0;
            let in_b = (x - 12.0).powi(2) + (y - 10.0).powi(2) < 36.0;
            out.push(if in_a || in_b { 1.0 } else { 0.0 });
        }
    }
    out
}

/// Reconstruct the CT phantom from `angles` projections over `[0, π)`, row-major.
/// `filtered` selects filtered back-projection (ramp filter) over the naive
/// back-projection, which smears.
///
/// # Errors
///
/// Throws a JavaScript error if `n` or `angles` is zero.
#[wasm_bindgen]
pub fn ct_reconstruct(n: usize, angles: usize, filtered: bool) -> Result<Vec<f64>, JsError> {
    if n == 0 || angles == 0 {
        return Err(JsError::new("n and angles must be positive"));
    }
    let phantom = ct_phantom(n);
    let truth = DMatrix::from_fn(n, n, |i, j| phantom[i * n + j]);
    let angle_list = tpt_sci_image::linspace(0.0, std::f64::consts::PI, angles);
    let sinogram = tpt_sci_image::radon_transform(&truth, &angle_list)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let image = if filtered {
        tpt_sci_image::filtered_back_projection(&sinogram, &angle_list)
    } else {
        tpt_sci_image::naive_back_projection(&sinogram, &angle_list)
    }
    .map_err(|e| JsError::new(&e.to_string()))?;
    Ok((0..n * n).map(|k| image[(k / n, k % n)]).collect())
}

/// Speed `|(u, v)|` of the lid-driven cavity on an `n × n` grid after `steps`
/// time steps, row-major. The lid moves at unit speed and the Reynolds number is
/// `reynolds`. Uses the explicit step of the CFD example (`dt = 5e-3`), so large
/// `n` with a high Reynolds number may blow up.
///
/// # Errors
///
/// Throws a JavaScript error if the grid is empty or the solver diverges.
#[wasm_bindgen]
pub fn lid_cavity_speed(n: usize, reynolds: f64, steps: usize) -> Result<Vec<f64>, JsError> {
    if n < 4 || steps == 0 || reynolds.is_nan() || reynolds <= 0.0 {
        return Err(JsError::new("need n >= 4, steps >= 1, reynolds > 0"));
    }
    let grid = CollocatedGrid::new(n, n, 1.0, 1.0).map_err(|e| JsError::new(&e.to_string()))?;
    let mut step = Step::new(grid, 1.0 / reynolds, 5e-3, 1.0);
    step.set_boundary(Boundary::Top, 1.0);
    for k in 0..steps {
        if !step.advance() {
            return Err(JsError::new(&format!("cavity diverged at step {k}")));
        }
    }
    let g = step.grid();
    let mut out = Vec::with_capacity(g.nx * g.ny);
    for j in 0..g.ny {
        for i in 0..g.nx {
            let c = g.idx(i, j);
            out.push(step.u[c].hypot(step.v[c]));
        }
    }
    Ok(out)
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
    fn filtered_back_projection_beats_naive() {
        let rmse = |img: &[f64]| {
            let truth = ct_phantom(32);
            (img.iter()
                .zip(&truth)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / truth.len() as f64)
                .sqrt()
        };
        let naive = ct_reconstruct(32, 60, false).unwrap();
        let fbp = ct_reconstruct(32, 60, true).unwrap();
        assert!(rmse(&fbp) < rmse(&naive));
    }

    #[test]
    fn cavity_speed_field_is_bounded_by_the_lid() {
        let speed = lid_cavity_speed(16, 100.0, 50).unwrap();
        assert_eq!(speed.len(), 16 * 16);
        assert!(speed.iter().all(|v| v.is_finite() && *v <= 1.0 + 1e-9));
    }

    #[test]
    fn doubling_co2_warms_the_model() {
        let t1 = ebm_equilibrium_temperature(280.0).unwrap();
        let t2 = ebm_equilibrium_temperature(560.0).unwrap();
        assert!(t2 > t1);
    }
}
