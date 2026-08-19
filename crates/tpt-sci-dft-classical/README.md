# tpt-sci-dft-classical

Classical / soft-matter **density functional theory (DFT)** for the
`tpt-science` pillar, provided by wrapping the [`feos`](https://docs.rs/feos)
framework ([`feos_dft`], MIT OR Apache-2.0).

## Features

* `ClassicalDft` — a handle that owns any concrete `feos` Helmholtz energy
  functional (PC-SAFT, PeTS, …) and is the entry point for 1-D profile solves.
* Re-exports of `feos` and `feos_dft` so downstream code can reach the full
  machinery (`DFTProfile`, `DFTSolver`, `ConvolverFFT`, geometries) directly.

The numerical heavy lifting — functional derivatives, FFT-based convolutions,
Picard / Anderson-mixing solvers, grand-potential minimization — is done by
`feos` itself. This crate only assembles the inputs and exposes a tidy result.

## Example

The `adsorption` example (`cargo run --example adsorption -p tpt-sci-dft-classical`)
shows a complete PC-SAFT density-profile solve (density profile + adsorption
isotherm) wired through this wrapper, using `feos`'s built-in PC-SAFT
parameters.

```ignore
use tpt_sci_dft_classical::ClassicalDft;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;

let parameters = PcSaftParameters::from_json(
    vec!["propane"], "../../parameters/pcsaft/esper2023.json",
    None, IdentifierOption::Name,
).unwrap();
let dft = ClassicalDft::with_functional(PcSaft::new(parameters));
```

## Scope (v1)

Planar/interface density profiles and adsorption isotherms via a wrapped EOS
functional. Building a *new* functional from scratch and 3-D molecular DFT are
out of scope (use `feos`'s functionals directly).

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
