//! Classical DFT demo: build a PC-SAFT functional and wrap it in `ClassicalDft`.
//!
//! This shows the `tpt-sci-dft-classical` entry point. A full 1-D/2-D/3-D
//! density-profile solve (grid construction, FFT convolver, Picard/Anderson
//! mixing) is performed by `feos_dft` once a functional is wrapped — see the
//! `feos` project's adsorption examples for the complete solve recipe that
//! consumes the functional held by [`ClassicalDft`].
//!
//! Requires feos' built-in PC-SAFT parameters (`../../parameters/pcsaft/
//! esper2023.json`, vendored with the `feos` repo). If the file is absent this
//! example prints a notice and exits cleanly.
//!
//! Run with: `cargo run --example adsorption -p tpt-sci-dft-classical`

use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use tpt_sci_dft_classical::ClassicalDft;

fn main() {
    let param_path = "../../parameters/pcsaft/esper2023.json";
    let params = match PcSaftParameters::from_json(
        vec!["propane"],
        param_path,
        None,
        IdentifierOption::Name,
    ) {
        Ok(p) => p,
        Err(_) => {
            println!("feos PC-SAFT parameters not found at {param_path}; skipping.");
            return;
        }
    };

    let functional = PcSaft::new(params);
    let _dft = ClassicalDft::with_functional(functional);
    println!("Wrapped a PC-SAFT Helmholtz functional.");
    println!("Ready for a feos_dft profile solve against this functional.");
}
