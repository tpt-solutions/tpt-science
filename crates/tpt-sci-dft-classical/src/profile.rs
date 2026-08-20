//! High-level classical-DFT entry point wrapping `feos`/`feos_dft`.
//!
//! This layer stores a concrete Helmholtz energy functional (e.g. `PcSaft`
//! from `feos::pcsaft`) behind a `ClassicalDft<F>` handle, generic over the
//! functional type `F`. The heavy numerical work — building the
//! [`feos_dft::DFTProfile`], the FFT convolver, Picard / Anderson mixing, and the
//! grand-potential solve — is performed by `feos` itself; see
//! `examples/adsorption.rs` for a complete PC-SAFT density-profile solve that
//! wires `ClassicalDft` into [`feos_dft`].

use std::sync::Arc;

use feos_dft::HelmholtzEnergyFunctional;

/// A classical-DFT problem handle, generic over the wrapped Helmholtz energy
/// functional `F` (any type implementing [`HelmholtzEnergyFunctional`], e.g.
/// `PcSaft` from `feos::pcsaft`).
pub struct ClassicalDft<F: HelmholtzEnergyFunctional> {
    /// The Helmholtz energy functional (PC-SAFT, PeTS, …).
    pub functional: Arc<F>,
}

impl<F: HelmholtzEnergyFunctional> ClassicalDft<F> {
    /// Construct from any concrete `feos` Helmholtz energy functional.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Requires feos' built-in parameter JSON (see `examples/adsorption.rs`).
    /// use tpt_sci_dft_classical::ClassicalDft;
    /// use feos::pcsaft::{PcSaft, PcSaftParameters};
    /// use feos_core::parameter::IdentifierOption;
    ///
    /// let parameters = PcSaftParameters::from_json(
    ///     vec!["propane"],
    ///     "../../parameters/pcsaft/esper2023.json",
    ///     None,
    ///     IdentifierOption::Name,
    /// ).unwrap();
    /// let dft = ClassicalDft::with_functional(PcSaft::new(parameters));
    /// ```
    #[must_use]
    pub fn with_functional(functional: F) -> Self {
        Self {
            functional: Arc::new(functional),
        }
    }

    /// Borrow the wrapped functional.
    #[must_use]
    pub fn functional_ref(&self) -> &F {
        self.functional.as_ref()
    }
}
