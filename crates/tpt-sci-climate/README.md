# tpt-sci-climate

Reduced-order **climate modelling** for the `tpt-science` pillar, built on
[`tpt-sci-ode`](https://docs.rs/tpt-sci-ode) and `tpt-math-linalg`.

## Features

* `EnergyBalanceModel` — 0-D global energy balance with CO₂ forcing
  `ΔF = 5.35·ln(C/C0)` (W/m²). Integrate with `step(dt)` (explicit Euler) or jump
  straight to the steady state with `equilibrium_temperature()` (Newton fixpoint).
* `grey_radiative_transfer` — single-layer grey-atmosphere surface temperature
  `Ts = (S·(1−α)/(4·σ·(1−ε/2)))^(1/4)`.
* `ChemistryBox` — constant-production / first-order-loss tracer
  (`dC/dt = P − k·C`) with `steady_state()` (C* = P/k).
* Constants: `SIGMA` (Stefan–Boltzmann), `CO2_PREINDUSTRIAL` (280 ppm).

## Example

```rust
use tpt_sci_climate::EnergyBalanceModel;

// Heat capacity, albedo, emissivity, CO2 (ppm).
let mut ebm = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 280.0).unwrap();
let t0 = ebm.equilibrium_temperature();
ebm.co2 = 560.0; // doubled
let t2 = ebm.equilibrium_temperature();
println!("equilibrium warming = {:.2} K", t2 - t0);

// Time-march toward the new equilibrium.
for _ in 0..5000 {
    ebm.step(1.0);
}
assert!(ebm.temperature().is_finite());
```

The `warming` example (`cargo run --example warming -p tpt-sci-climate`) runs the
full CO₂-doubling relaxation.

## Scope (v1)

0-D EBM, simple grey radiative transfer, single-tracer atmospheric chemistry.
GCMs, full radiative-transfer bands, and 3-D atmospheric chemistry are out of v1
scope.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
