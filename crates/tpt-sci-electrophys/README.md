# tpt-sci-electrophys

Cardiac **electrophysiology** for the `tpt-science` pillar, built from
scratch (`HodgkinHuxley`/`Tissue` use a hand-rolled explicit-Euler stepper
and a hand-rolled 5-point Laplacian, not `tpt-sci-ode`/`tpt-sci-grid`).

## Features

* `HodgkinHuxley` — classic giant-axon model: gating `m, h, n` with
  voltage-dependent rate laws and ionic current
  `I_ion = ḡNa·m³h·(V−E_Na) + ḡK·n⁴·(V−E_K) + ḡL·(V−E_L)`. Integrate the
  membrane ODE with `step(dt)`; inspect `state()` / `voltage()`.
* `Tissue` — 2-D **monodomain** sheet coupling the HH membrane to a 5-point
  Laplacian diffusion (`dVm/dt = −I_ion/Cm + D·∇²Vm`), so an action potential
  launched at one node propagates through the tissue. Bidomain is not
  implemented (see Scope below).

## Example

```rust
use tpt_sci_electrophys::HodgkinHuxley;

let mut hh = HodgkinHuxley::resting();
// Depolarize; integrate the membrane ODE for a short time.
let y = hh.state();
assert!(y.len() == 4);
hh.step(0.01);
assert!(hh.voltage().is_finite());
```

The `ap_wave` example (`cargo run --example ap_wave -p tpt-sci-electrophys`)
propagates an action potential across a 2-D tissue sheet.

## Scope (v1)

Single-cell HH + 2-D monodomain propagation. Full bidomain (intra/extra split),
ionic models beyond HH (e.g. Ten Tusscher), and anisotropy are out of scope for
v1.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
