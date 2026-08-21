# tpt-sci-sim-core

Multi-scale simulation orchestration for the `tpt-science` pillar: the layer
*above* `tpt-sci-ode` and `tpt-sci-grid` that a "multi-scale computational
platform" actually needs and that neither solver crate provides alone.

It provides:

- **Time-stepping across heterogeneous sub-models.** Each `SubModel` advances on
  its own internal time scale (`max_step`) while a `Simulation` drives them all
  forward to a shared target time, taking the largest sub-step that no model
  would overshoot.
- **Cross-scale coupling.** A `Coupling` maps the state of one model onto the
  input buffer of another after every sub-step.
- **Checkpointing.** `Simulation::snapshot` / `Simulation::restore` capture and
  restore every model's state and the global clock, for resumable /
  reproducible runs.

Two ready-made `SubModel` implementations are provided: `OdeSubModel` wraps a
`tpt-sci-ode` problem, and `DiffusionSubModel` wraps a 1-D diffusion field
built on a `tpt-sci-grid` Laplacian.

Depends on `tpt-sci-ode`, `tpt-sci-grid`. The `multi_scale_cookbook` example
additionally dev-depends on `tpt-sci-reaction-network` to drive an
`OdeSubModel` from a reaction-network model.

## Example

```rust
use tpt_sci_sim_core::{OdeSubModel, Simulation, SubModel};

// A single fast exponential-decay sub-model: dy/dt = -y, y(0) = 1.
let decay = OdeSubModel::new(
    "decay",
    |_t, y, dydt| dydt[0] = -y[0],
    vec![1.0],
    0.0,
);
let mut sim = Simulation::new();
sim.add_model(decay).unwrap();
sim.step_until(1.0).unwrap();
let y = sim.model("decay").unwrap().state()[0];
assert!((y - std::f64::consts::E.recip()).abs() < 1e-3);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
