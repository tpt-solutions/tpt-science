# tpt-sci-climate

Reduced-order **climate modelling** for the `tpt-science` pillar, built from
scratch (`EnergyBalanceModel`/`ChemistryBox` use a hand-rolled explicit-Euler
stepper, not `tpt-sci-ode`/`tpt-math-linalg`).

## Features

* `EnergyBalanceModel` — 0-D global energy balance with CO₂ forcing
  `ΔF = 5.35·ln(C/C0)` (W/m²). Integrate with `step(dt)` (explicit Euler) or jump
  straight to the steady state with `equilibrium_temperature()` (Newton fixpoint).
* `grey_radiative_transfer` — single-layer grey-atmosphere surface temperature
  `Ts = (S·(1−α)/(4·σ·(1−ε/2)))^(1/4)`.
* `radiative_transfer` — **multi-band longwave** schemes replacing the single
  grey band: [`MultiBandRadiativeTransfer`] (a stack of grey slabs, band OLR
  `F_b = w_b·(τ_b·σ·Ts⁴ + ε_b·σ·Ta⁴)`) and a `k`-distribution [`CorrelatedKRt`].
* `ChemistryBox` — constant-production / first-order-loss tracer
  (`dC/dt = P − k·C`) with `steady_state()` (C* = P/k).
* `Tracer3D` — the 3-D advection–diffusion–reaction analogue of `ChemistryBox`
  on a `tpt-sci-grid::UniformGrid3D` (upwind advection + sparse Laplacian + a
  per-cell `P − k·c` source/sink).
* `AtmosphereGcm` — a genuine **primitive-equation atmospheric GCM dynamical
  core** (hydrostatic, with an optional non-hydrostatic pressure correction),
  coupled to the `EnergyBalanceModel` via `couple_to_ebm`.
* Constants: `SIGMA` (Stefan–Boltzmann), `CO2_PREINDUSTRIAL` (280 ppm).

## Example

```rust
use tpt_sci_climate::EnergyBalanceModel;

// Heat capacity, albedo, emissivity, CO2 (ppm).
let mut ebm = EnergyBalanceModel::new(1.0, 0.3, 0.61, 280.0).unwrap();
let t0 = ebm.equilibrium_temperature();
ebm.co2 = 560.0; // doubled
let t2 = ebm.equilibrium_temperature();
println!("equilibrium warming = {:.2} K", t2 - t0);

// Time-march toward the new equilibrium.
for _ in 0..100 {
    ebm.step(1.0);
}
assert!(ebm.temperature().is_finite());
```

The `warming` example (`cargo run --example warming -p tpt-sci-climate`) runs the
full CO₂-doubling relaxation.

## Scope (v1)

0-D EBM, multi-band/correlated-k longwave radiative transfer, single- and 3-D
tracer atmospheric chemistry, and a hydrostatic (optionally non-hydrostatic)
primitive-equation GCM dynamical core coupled to the EBM.

Clouds, moist convection, and spectral-dynamics (spherical-harmonic) GCM cores
remain out of scope — the GCM here is a reduced-order Cartesian primitive-equation
core, not a fully-coupled global climate model.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
