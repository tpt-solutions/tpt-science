# tpt-sci-reaction-network

Species / rate / stoichiometry DSL for compartmental models — the Rust
equivalent of Julia's [Catalyst.jl](https://github.com/SciML/Catalyst.jl) —
built from scratch for the `tpt-science` pillar.

Given a chemical reaction network (species, reactions, and rate laws), the crate
compiles the **law of mass action** into the deterministic ODE

```text
dy/dt = S · r(y, p)
```

(where `S` is the stoichiometry matrix and `r` the per-reaction rate vector) and
hands the result to [`tpt-sci-ode`](https://crates.io/crates/tpt-sci-ode) for
integration. It also exposes the stoichiometry matrix and rate vector directly
for inspection, parameter estimation, or coupling into a larger multi-scale
model.

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE).

## Example

Michaelis–Menten enzyme kinetics (`S + E ⇌ SE → P + E`):

```rust
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::ReactionNetwork;

let mut model = ReactionNetwork::from_dsl(
    "kB, S + E --> SE
     kD, SE --> S + E
     kP, SE --> P + E",
).unwrap();
model.set_parameter("kB", 0.01).unwrap();
model.set_parameter("kD", 0.1).unwrap();
model.set_parameter("kP", 0.1).unwrap();

// E + SE is conserved: total enzyme stays at its initial 10.
let y0 = model.initial_state(&[("S", 50.0), ("E", 10.0)]).unwrap();
let y = model.to_ode_problem(&y0, 0.0).unwrap().solve(Method::Bdf, 200.0).unwrap();

let e = model.species_index("E").unwrap();
let se = model.species_index("SE").unwrap();
assert!((y[e] + y[se] - 10.0).abs() < 1e-6);
```

## Scope

**v1 (this crate):** a CRN IR (species / parameters / reactions), mass-action
rate laws (plus custom non-mass-action rate closures), a stoichiometry matrix
and reaction-rate builder, and an `OdeProblem` bridge into `tpt-sci-ode`. A
textual Catalyst.jl-style DSL (`kB, S + E --> SE`) is provided for convenience.

**Out of v1 (documented, not built):** the stochastic Gillespie / SSA backend
(defer to wrapping [`rebop`](https://crates.io/crates/rebop), MIT), SDE/jump
models, SBML I/O, network analysis, and conservation-law elimination.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or [Apache-2.0](../../LICENSE-APACHE)
at your option.
