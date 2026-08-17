use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_sci_grid::{Boundary, UniformGrid1D, laplacian_1d};

use crate::SimError;
use crate::submodel::SubModel;

/// A [`SubModel`] integrating the 1-D diffusion equation on a uniform grid
/// with explicit Euler time stepping.
///
/// The discrete semi-discrete system is
/// `∂u/∂t = D · L u + s`, where `L` is the [`laplacian_1d`] assembled by
/// [`tpt_sci_grid`], `D` is the diffusion coefficient, and `s` is an optional
/// per-node forcing term. The forcing is held in the model's **input buffer**,
/// so a coupling can drive the field from another sub-model's state — the
/// canonical cross-scale pattern (e.g. a fast reaction ODE feeding a slow
/// diffusion field).
pub struct DiffusionSubModel {
    id: String,
    laplacian: DMatrix,
    coeff: f64,
    state: Vec<f64>,
    input: Vec<f64>,
    time: f64,
    max_step: f64,
}

impl DiffusionSubModel {
    /// Build a diffusion sub-model on `grid` with diffusion coefficient `coeff`
    /// and boundary condition `bc`. The initial field is `u0` (one value per
    /// grid node).
    ///
    /// # Errors
    ///
    /// Returns [`SimError::Advance`] if `u0`'s length does not match the number
    /// of grid nodes.
    pub fn new(
        id: &str,
        grid: UniformGrid1D,
        coeff: f64,
        bc: Boundary,
        u0: Vec<f64>,
    ) -> Result<Self, SimError> {
        if u0.len() != grid.n() {
            return Err(SimError::Advance(
                id.to_string(),
                format!(
                    "initial field length {} != grid size {}",
                    u0.len(),
                    grid.n()
                ),
            ));
        }
        // Stability-limited explicit-Euler step for 1-D diffusion on a uniform
        // grid: dt <= dx^2 / (2 D). Use a safe fraction of that.
        let stability = grid.dx() * grid.dx() / (2.0 * coeff.max(f64::MIN_POSITIVE));
        Ok(Self {
            id: id.to_string(),
            laplacian: laplacian_1d(&grid, bc),
            coeff,
            state: u0.clone(),
            input: vec![0.0; u0.len()],
            time: 0.0,
            max_step: 0.9 * stability,
        })
    }

    /// Diffusion coefficient.
    pub fn coeff(&self) -> f64 {
        self.coeff
    }

    /// Largest stable internal step (derived from the explicit-Euler stability
    /// limit on construction; can be overridden with `set_max_step`).
    pub fn set_max_step(&mut self, max_step: f64) {
        self.max_step = max_step;
    }
}

impl SubModel for DiffusionSubModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn max_step(&self) -> f64 {
        self.max_step
    }

    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        let u = DVector::from_vec(self.state.clone());
        let lap = self.laplacian.clone() * u;
        let mut next = self.state.clone();
        for i in 0..next.len() {
            next[i] += dt * (self.coeff * lap[i] + self.input[i]);
        }
        self.state = next;
        self.time += dt;
        Ok(())
    }

    fn state(&self) -> &[f64] {
        &self.state
    }

    fn input_mut(&mut self) -> Option<&mut [f64]> {
        Some(&mut self.input)
    }

    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        if state.len() != self.state.len() {
            return Err(SimError::Advance(
                self.id.clone(),
                format!(
                    "checkpoint state length {} != model state length {}",
                    state.len(),
                    self.state.len()
                ),
            ));
        }
        self.state = state.to_vec();
        self.time = time;
        Ok(())
    }
}
