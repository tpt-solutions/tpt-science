use std::rc::Rc;

/// Mapping from one sub-model's state onto another's input buffer, applied by
/// the orchestrator after every sub-step. This is how cross-scale coupling is
/// expressed: the `source` model's `state()` is read and written into the
/// `target` model's `input_mut()` buffer via `apply`.
pub type CouplingFn = dyn Fn(&[f64], &mut [f64]);

/// A directed coupling `source -> target`.
///
/// After each orchestration sub-step, the simulation calls
/// `apply(source.state(), target.input_mut())`. The `target` model then
/// consumes the written input on its *next* `advance`. This one-step-lagged
/// exchange is the standard, stable pattern for coupling sub-models running
/// at different time scales (the `target` advances using the *latest*
/// `source` value, never a future one).
pub struct Coupling {
    source: String,
    target: String,
    apply: Rc<CouplingFn>,
}

impl Coupling {
    /// Create a coupling from `source` (whose `state()` is read) to `target`
    /// (whose `input_mut()` buffer is written). The closure `apply` maps the
    /// source state onto the target input; it must handle differing lengths
    /// (e.g. by sampling, averaging, or projection).
    pub fn new<F>(source: &str, target: &str, apply: F) -> Self
    where
        F: Fn(&[f64], &mut [f64]) + 'static,
    {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            apply: Rc::new(apply),
        }
    }

    /// Identifier of the source sub-model.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Identifier of the target sub-model.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Apply the coupling: read `src` (the source model's state) and write into
    /// `input` (the target model's input buffer).
    pub fn apply(&self, src: &[f64], input: &mut [f64]) {
        (self.apply)(src, input);
    }
}
