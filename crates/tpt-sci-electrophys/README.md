# tpt-sci-electrophys

Cardiac **electrophysiology** for the `tpt-science` pillar, built on
[`tpt_sci_grid`] (the structured-grid diffusion / sparse elliptic solver) and a
hand-rolled explicit-Euler membrane stepper.

## Features

* `HodgkinHuxley` — classic giant-axon model: gating `m, h, n` with
  voltage-dependent rate laws and ionic current
  `I_ion = ḡNa·m³h·(V−E_Na) + ḡK·n⁴·(V−E_K) + ḡL·(V−E_L)`. Integrate the
  membrane ODE with `step(dt)`; inspect `state()` / `voltage()`.
* `TenTusscher` — Ten Tusscher–Panfilov (2004) human-ventricular myocyte
  (epicardial variant): a genuine ionic model with fast Na, `I_Kr`, `I_Ks`,
  `I_K1`, `I_CaL`, `I_to`, the Na/Ca exchanger and a reduced calcium balance.
  Both models implement the `IonicModel` trait, so the same tissue solver can
  drive either membrane.
* `Tissue` — 2-D **monodomain** sheet coupling a membrane model to a diffusion
  operator: `dVm/dt = −I_ion/Cm + ∇·(D∇Vm)`, so an action potential launched at
  one node propagates through the tissue.
* **Bidomain** — `Tissue::enable_bidomain(σ_e)` couples the intracellular field
  to an extracellular potential `Ve` governed by the elliptic equation
  `(σ_i + σ_e)·∇²Ve = −σ_i·∇²Vm`. The conductivity-weighted Laplacian is
  assembled as a sparse [`tpt_sci_grid`] `CsrMatrix` and solved each step with
  `tpt_sci_grid::sparse::conjugate_gradient`. `Tissue::bidomain_step` is the
  explicitly-named entry point; `extracellular_potential` exposes the raw `Ve`
  field. When `σ_i = σ_e` the bidomain reduces to the monodomain with effective
  diffusion `σ_i·σ_e/(σ_i+σ_e)`, and as `σ_e → ∞` it reduces to the monodomain
  with `D = σ_i`.
* **Anisotropic (tensor) diffusion** — per-node 2×2 symmetric positive-definite
  diffusivity tensors via `DiffusionTensor` (alias `ConductivityTensor`) and the
  free `tensor_diffusion_2d` operator, so fibre-orientation effects (faster
  conduction along fibres) are captured by `∇·(D∇Vm)`.

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
propagates an action potential across a 2-D tissue sheet, `cable_propagation`
(`cargo run --example cable_propagation -p tpt-sci-electrophys`) shows
1-D cable conduction, and `bidomain_demo`
(`cargo run --example bidomain_demo -p tpt-sci-electrophys`) demonstrates the
full bidomain extracellular solve alongside the monodomain.

## Scope (v1)

Single-cell `HodgkinHuxley` + `TenTusscher` ionic models; 2-D monodomain and
full **bidomain** propagation with **anisotropic (tensor) diffusion** on a
structured grid. The original v1 "out of scope" items (full bidomain, a second
ionic model, and anisotropy) are all implemented. Remaining simplifications:
hand-rolled explicit-Euler integration (not the `tpt-sci-ode` adaptive driver),
structured 2-D grids only (no 3-D/unstructured), and a single-cation-reduced
calcium balance in the Ten Tusscher cell.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
