use tpt_sci_ode::{Method, OdeProblem, OdeProblemBuilder};

use crate::SimError;
use crate::submodel::SubModel;

/// A [`SubModel`] wrapping a `tpt-sci-ode` [`OdeProblem`].
///
/// The model keeps its current state and time; each `advance(dt)` integrates
/// the ODE forward by `dt` from the current state using a fixed integration
/// [`Method`] (default [`Method::Bdf`]). Because [`OdeProblem`] is
/// immutable once built, the model respawns a fresh problem rooted at the
/// current `(time, state)` before each step (see
/// [`OdeProblem::respawn`]). This makes it a drop-in slow/fast sub-model in a
/// multi-scale [`Simulation`](crate::Simulation).
pub struct OdeSubModel {
    id: String,
    problem: OdeProblem,
    method: Method,
    state: Vec<f64>,
    time: f64,
    max_step: f64,
}

impl OdeSubModel {
    /// Build a sub-model from an RHS closure, initial state `y0` at time `t0`.
    /// Uses default solver tolerances and `Bdf` integration.
    pub fn new<F>(id: &str, rhs: F, y0: Vec<f64>, t0: f64) -> Self
    where
        F: Fn(f64, &[f64], &mut [f64]) + 'static,
    {
        Self::with_builder(id, OdeProblemBuilder::new(rhs, y0, t0), Method::Bdf)
    }

    /// Build a sub-model from a fully configured [`OdeProblemBuilder`].
    ///
    /// # Panics
    ///
    /// Panics if `builder.build()` fails (e.g. an empty initial state or
    /// non-positive tolerances); callers should construct a valid builder.
    pub fn with_builder(id: &str, builder: OdeProblemBuilder, method: Method) -> Self {
        let problem = builder
            .build()
            .expect("OdeSubModel builder must produce a valid OdeProblem");
        let time = problem.t0();
        let state = problem.y0().to_vec();
        Self {
            id: id.to_string(),
            problem,
            method,
            state,
            time,
            max_step: f64::INFINITY,
        }
    }

    /// Integration method used for each `advance`.
    pub fn method(&self) -> Method {
        self.method
    }

    /// Set the integration method.
    pub fn set_method(&mut self, method: Method) {
        self.method = method;
    }

    /// Largest internal step the model will take in one `advance`. Defaults to
    /// unbounded; set a finite value to force the orchestrator to sub-divide
    /// (e.g. for coupling a fast ODE into a slow simulation).
    pub fn set_max_step(&mut self, max_step: f64) {
        self.max_step = max_step;
    }
}

impl SubModel for OdeSubModel {
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
        let next = self.time + dt;
        let step = self
            .problem
            .respawn(self.state.clone(), self.time)
            .map_err(|e| SimError::Advance(self.id.clone(), e.to_string()))?;
        let out = step
            .solve(self.method, next)
            .map_err(|e| SimError::Advance(self.id.clone(), e.to_string()))?;
        self.state = out;
        self.time = next;
        Ok(())
    }

    fn state(&self) -> &[f64] {
        &self.state
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
