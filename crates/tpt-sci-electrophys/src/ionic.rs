//! The ionic-model interface shared by HH and Ten Tusscher cells.

/// A membrane ionic model (the reaction term of the tissue PDE).
///
/// Implementors expose their state, transmembrane voltage, the outward ionic
/// current `I_ion` (per unit membrane area), and a capacitance. The tissue
/// solver owns the transmembrane voltage; it calls [`IonicModel::ionic_current`]
/// to read the reaction term and [`IonicModel::advance_reaction`] to evolve the
/// internal gating/ionic state at a given transmembrane potential.
pub trait IonicModel: Send + std::fmt::Debug {
    /// Full state vector (transmembrane voltage followed by gating/ionic
    /// variables), used for inspection and cloning.
    fn state(&self) -> Vec<f64>;
    /// Current transmembrane potential `Vm` (mV).
    fn voltage(&self) -> f64;
    /// Outward ionic current `I_ion` (µA/cm²) at the current state.
    fn ionic_current(&self) -> f64;
    /// Membrane capacitance `Cm` (µF/cm²).
    fn capacitance(&self) -> f64;
    /// Evolve the internal (gating / ionic) state by `dt` while holding the
    /// transmembrane voltage at `v`. The voltage itself is owned by the tissue
    /// solver, so implementors must *not* update it here.
    fn advance_reaction(&mut self, v: f64, dt: f64);
    /// Clone into a boxed trait object (so a [`crate::Tissue`] can hold one model per
    /// node without generics).
    fn clone_box(&self) -> Box<dyn IonicModel>;
}

impl Clone for Box<dyn IonicModel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
