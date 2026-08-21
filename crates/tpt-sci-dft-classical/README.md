# tpt-sci-dft-classical

Classical / soft-matter **density functional theory (DFT)** for the
`tpt-science` pillar, provided by wrapping the [`feos`](https://docs.rs/feos)
framework ([`feos_dft`], MIT OR Apache-2.0).

## Features

* `ClassicalDft` — a handle that owns any concrete `feos` Helmholtz energy
  functional (PC-SAFT, PeTS, …) and is the entry point for 1-D profile solves.
* `SquareGradientDft` — a from-scratch van der Waals square-gradient local
  density functional with 1-D planar and 3-D real-space solvers (no `feos`
  dependency required).
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

This crate provides two complementary classical-DFT paths:

* **Wrapped `feos`/`feos-dft`** — `ClassicalDft` owns any concrete `feos`
  Helmholtz energy functional (PC-SAFT, PeTS, …) for 1-D planar density
  profiles, adsorption isotherms, and surface tension. The numerical heavy
  lifting (FFT convolutions, Picard/Anderson mixing) is done by `feos`.
* **From-scratch square-gradient / local density-functional**
  (`SquareGradientDft`) — a self-contained van der Waals square-gradient
  functional `F[n] = ∫ (f_bulk(n) + κ/2 (∇n)²) dr` for a simple inhomogeneous
  fluid, minimised by gradient relaxation of the Euler–Lagrange equation
  `μ = ∂f_bulk/∂n − κ∇²n` to a constant chemical potential. It ships a 1-D
  planar solve (bulk, hard wall, capillary length) and a 3-D generalisation that
  reuses `tpt-sci-grid`'s sparse 3-D Laplacian per node. See
  `examples/square_gradient.rs`.

Building a *molecular* (hard-sphere / fundamental-measure) functional from
scratch and full 3-D electronic DFT remain out of scope (use `feos`'s
functionals directly).

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
