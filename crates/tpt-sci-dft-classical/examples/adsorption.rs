//! Classical DFT demo: build a PC-SAFT functional (via the `feos` wrap) and show
//! it is ready to drive a `feos_dft` density-profile solve.
//!
//! Requires feos' built-in PC-SAFT parameters (`../../parameters/pcsaft/
//! esper2023.json`, vendored with the `feos` repo). If the file is absent this
//! example prints a notice and exits cleanly.
//!
//! Run with: `cargo run --example adsorption -p tpt-sci-dft-classical`

use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;

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

    let _functional = PcSaft::new(params);
    println!("Wrapped a PC-SAFT Helmholtz functional from feos.");
    println!(
        "A full 1-D/2-D/3-D density-profile solve consumes this functional via \
         feos_dft (DFTProfile + DFTSolver), per the tpt-sci-dft-classical README."
    );
}
