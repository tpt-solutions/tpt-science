//! High-level classical-DFT entry point wrapping `feos`/`feos_dft`.
//!
//! This layer stores a concrete Helmholtz energy functional (e.g. `PcSaft`
//! from [`feos::pcsaft`]) behind a `ClassicalDft` handle. The heavy numerical
//! work — building the [`feos_dft::DFTProfile`], the FFT convolver, Picard /
//! Anderson mixing, and the grand-potential solve — is performed by `feos`
//! itself; see `examples/adsorption.rs` for a complete PC-SAFT density-profile
//! solve that wires `ClassicalDft` into [`feos_dft`].

use std::sync::Arc;

use feos_dft::HelmholtzEnergyFunctional;

/// A classical-DFT problem handle: owns a Helmholtz energy functional and the
/// bulk thermodynamic state the profiles are solved against.
#[derive(Clone)]
pub struct ClassicalDft {
    /// The Helmholtz energy functional (PC-SAFT, PeTS, …).
    pub functional: Arc<dyn HelmholtzEnergyFunctional>,
}

impl ClassicalDft {
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
    pub fn with_functional<F: HelmholtzEnergyFunctional + 'static>(functional: F) -> Self {
        Self {
            functional: Arc::new(functional),
        }
    }

    /// Borrow the wrapped functional.
    #[must_use]
    pub fn functional_ref(&self) -> &dyn HelmholtzEnergyFunctional {
        self.functional.as_ref()
    }
}
