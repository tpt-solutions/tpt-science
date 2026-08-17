use std::collections::HashMap;

use crate::coupling::Coupling;
use crate::error::SimError;
use crate::submodel::SubModel;

/// An immutable snapshot of a [`Simulation`]'s state, suitable for
/// checkpoint/restart.
///
/// `Checkpoint` stores the global clock plus each sub-model's state vector and
/// internal time. It is intentionally plain data (no trait objects) so it can
/// be serialized by downstream consumers.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Global simulation time at snapshot.
    pub global_time: f64,
    /// `(id, state, internal_time)` per sub-model, in registration order.
    pub models: Vec<(String, Vec<f64>, f64)>,
}

/// A multi-scale simulation orchestrating a set of [`SubModel`]s.
///
/// The orchestrator drives all sub-models forward to a shared target time with
/// adaptive sub-stepping: each sub-step is the largest `dt` that advances no
/// model past the target and exceeds no model's `max_step`. After every
/// sub-step, all registered [`Coupling`]s fire (writing cross-scale inputs),
/// so models of differing time scales stay coupled without overshooting.
pub struct Simulation {
    models: HashMap<String, Box<dyn SubModel>>,
    order: Vec<String>,
    couplings: Vec<Coupling>,
    global_time: f64,
}

impl Simulation {
    /// Create an empty simulation at global time `0.0`.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            order: Vec::new(),
            couplings: Vec::new(),
            global_time: 0.0,
        }
    }

    /// Register a sub-model. Returns an error if the id is already taken.
    ///
    /// # Errors
    ///
    /// Returns [`SimError::DuplicateModel`] if a model with the same id is
    /// already registered.
    pub fn add_model(&mut self, model: impl SubModel + 'static) -> Result<(), SimError> {
        let id = model.id().to_string();
        if self.models.contains_key(&id) {
            return Err(SimError::DuplicateModel(id));
        }
        self.order.push(id.clone());
        self.models.insert(id, Box::new(model));
        Ok(())
    }

    /// Register a cross-scale [`Coupling`]. The source and target models must
    /// already be registered (or will be — couplings are applied by id at
    /// `step_until` time, so order does not matter).
    pub fn add_coupling(&mut self, coupling: Coupling) {
        self.couplings.push(coupling);
    }

    /// Borrow a registered model by id.
    pub fn model(&self, id: &str) -> Option<&dyn SubModel> {
        self.models.get(id).map(|m| m.as_ref())
    }

    /// Current global simulation time (the minimum model time, i.e. how far
    /// all models have been driven together).
    pub fn global_time(&self) -> f64 {
        self.global_time
    }

    /// Advance all sub-models together until the global time reaches `target`.
    ///
    /// Sub-steps are chosen so that no model overshoots `target` and no model
    /// exceeds its `max_step`. After each sub-step every coupling is applied.
    ///
    /// # Errors
    ///
    /// Returns [`SimError::NonAdvancingStep`] if `target <= global_time`, or
    /// propagates any error returned by a sub-model's `advance`/`restore_state`
    /// or by a coupling (e.g. [`SimError::UnknownModel`]).
    ///
    /// # Panics
    ///
    /// Panics if an internally registered model is missing from the registry
    /// (a logic invariant; should never happen in practice).
    pub fn step_until(&mut self, target: f64) -> Result<(), SimError> {
        if target <= self.global_time {
            return Err(SimError::NonAdvancingStep(target, self.global_time));
        }
        const EPS: f64 = 1e-12;
        while self.global_time < target - EPS {
            let mut dt = target - self.global_time;
            for id in &self.order {
                let m = self.models.get(id).expect("registered model");
                let remaining = target - m.time();
                if remaining > EPS {
                    dt = dt.min(remaining).min(m.max_step());
                }
            }
            if dt <= 0.0 {
                break;
            }
            for id in &self.order {
                let m = self.models.get_mut(id).expect("registered model");
                if m.time() < target - EPS {
                    m.advance(dt)?;
                }
            }
            self.apply_couplings()?;
            self.global_time += dt;
        }
        self.global_time = target;
        Ok(())
    }

    fn apply_couplings(&mut self) -> Result<(), SimError> {
        for c in &self.couplings {
            let src = self
                .model(c.source())
                .ok_or_else(|| SimError::UnknownModel(c.source().to_string()))?
                .state()
                .to_vec();
            let target = self
                .models
                .get_mut(c.target())
                .ok_or_else(|| SimError::UnknownModel(c.target().to_string()))?;
            let input = target
                .input_mut()
                .ok_or_else(|| SimError::CouplingTargetNoInput(c.target().to_string()))?;
            c.apply(&src, input);
        }
        Ok(())
    }

    /// Capture a [`Checkpoint`] of the current global time and every model's
    /// state and internal time.
    ///
    /// # Panics
    ///
    /// Panics if an internally registered model is missing from the registry
    /// (a logic invariant; should never happen in practice).
    pub fn snapshot(&self) -> Checkpoint {
        let models = self
            .order
            .iter()
            .map(|id| {
                let m = self.models.get(id).expect("registered model");
                (id.clone(), m.state().to_vec(), m.time())
            })
            .collect();
        Checkpoint {
            global_time: self.global_time,
            models,
        }
    }

    /// Restore every model (and the global clock) from a [`Checkpoint`]. Models
    /// present in the checkpoint but not in the simulation are ignored; models
    /// in the simulation but missing from the checkpoint keep their current
    /// state.
    ///
    /// # Errors
    ///
    /// Returns [`SimError::UnknownModel`] if a model id present in the
    /// checkpoint is not registered in the simulation, or propagates any error
    /// from the sub-model's `restore_state` (e.g. a state-length mismatch).
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), SimError> {
        for (id, state, time) in &checkpoint.models {
            let m = self
                .models
                .get_mut(id)
                .ok_or_else(|| SimError::UnknownModel(id.clone()))?;
            m.restore_state(state, *time)?;
        }
        self.global_time = checkpoint.global_time;
        Ok(())
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::Coupling;
    use crate::diffusion_model::DiffusionSubModel;
    use crate::ode_model::OdeSubModel;

    fn decay(id: &str) -> OdeSubModel {
        OdeSubModel::new(id, |_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0)
    }

    #[test]
    fn single_ode_matches_analytic() {
        let mut sim = Simulation::new();
        sim.add_model(decay("decay")).unwrap();
        sim.step_until(1.0).unwrap();
        let y = sim.model("decay").unwrap().state()[0];
        assert_abs_diff_eq!(y, std::f64::consts::E.recip(), epsilon = 1e-3);
    }

    #[test]
    fn non_advancing_step_is_rejected() {
        let mut sim = Simulation::new();
        sim.add_model(decay("decay")).unwrap();
        assert!(matches!(
            sim.step_until(0.0),
            Err(SimError::NonAdvancingStep(_, _))
        ));
    }

    #[test]
    fn duplicate_model_is_rejected() {
        let mut sim = Simulation::new();
        sim.add_model(decay("decay")).unwrap();
        assert!(matches!(
            sim.add_model(decay("decay")),
            Err(SimError::DuplicateModel(_))
        ));
    }

    #[test]
    fn two_scale_substepping_respects_max_step() {
        // Two fast/slow decay ODEs; the orchestrator must sub-divide to honour
        // the smaller max_step while still landing both at the target time.
        let mut fast = decay("fast");
        fast.set_max_step(0.05);
        let mut slow = decay("slow");
        slow.set_max_step(0.1);

        let mut sim = Simulation::new();
        sim.add_model(fast).unwrap();
        sim.add_model(slow).unwrap();
        sim.step_until(0.2).unwrap();

        assert_abs_diff_eq!(sim.model("fast").unwrap().time(), 0.2, epsilon = 1e-9);
        assert_abs_diff_eq!(sim.model("slow").unwrap().time(), 0.2, epsilon = 1e-9);
        assert_eq!(sim.global_time(), 0.2);
    }

    #[test]
    fn cross_scale_coupling_drives_target() {
        // Source: y' = 1, y(0)=0  ->  drives a diffusion field via coupling.
        // A small max_step forces the orchestrator to sub-step, so the
        // one-step-lagged coupling actually feeds the field across sub-steps.
        let mut source = OdeSubModel::new("src", |_t, _y, dydt| dydt[0] = 1.0, vec![0.0], 0.0);
        source.set_max_step(0.1);
        let grid = tpt_sci_grid::UniformGrid1D::new(9, 0.0, 1.0).unwrap();
        let target = DiffusionSubModel::new(
            "field",
            grid,
            1e-3,
            tpt_sci_grid::Boundary::Neumann,
            vec![0.0; 9],
        )
        .unwrap();

        let mut sim = Simulation::new();
        sim.add_model(source).unwrap();
        sim.add_model(target).unwrap();
        // Broadcast the source scalar onto every diffusion node.
        sim.add_coupling(Coupling::new("src", "field", |src, input| {
            for x in input.iter_mut() {
                *x = src[0];
            }
        }));
        sim.step_until(0.5).unwrap();

        // The field must have been driven positive by the source forcing.
        let field = sim.model("field").unwrap().state();
        assert!(field.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn checkpoint_round_trips() {
        let mut sim = Simulation::new();
        sim.add_model(decay("decay")).unwrap();
        sim.step_until(0.5).unwrap();
        let checkpoint = sim.snapshot();
        let at_half = sim.model("decay").unwrap().state()[0];

        sim.step_until(1.0).unwrap();
        assert_abs_diff_eq!(
            sim.model("decay").unwrap().state()[0],
            std::f64::consts::E.recip(),
            epsilon = 1e-3
        );

        sim.restore(&checkpoint).unwrap();
        assert_abs_diff_eq!(
            sim.model("decay").unwrap().state()[0],
            at_half,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(sim.global_time(), 0.5, epsilon = 1e-12);
    }
}
