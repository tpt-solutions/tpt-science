# tpt-sci-kinetics

Surface and heterogeneous chemical kinetics for the `tpt-science` pillar, built
from scratch on top of [`tpt-sci-reaction-network`](https://docs.rs/tpt-sci-reaction-network)
(the mass-action CRN engine) and [`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).

## Features

* `ArrheniusRate` — `k(T) = A·exp(-Ea/(R·T))`, validated (`new`), temperature
  driven (`rate_constant`). `R_GAS` is the universal gas constant.
* `langmuir_hinshelwood_coverages` — single-site fractional surface coverages
  `θ_i = K_i·p_i / (1 + Σ K_j·p_j)` from adsorption equilibria + partial
  pressures (coverages sum to ≤ 1).
* `KineticsProblem` — binds an Arrhenius `ReactionSystem` to a temperature
  (`rate_constants`), integrating with `tpt-sci-ode`.

These are the two kinetic building blocks most reactor/catalysis models need on
top of plain mass-action CRNs.

## Example

```rust
use tpt_sci_kinetics::{ArrheniusRate, langmuir_hinshelwood_coverages};

// Unimolecular decay with A = 1e13, Ea = 80 kJ/mol.
let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
let k = r.rate_constant(800.0); // T = 800 K
assert!(k > 0.0 && k.is_finite());

// Two competing adsorbates on a single site type; coverages sum to <= 1.
let theta = langmuir_hinshelwood_coverages(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
let sum: f64 = theta.iter().sum();
assert!(sum <= 1.0 + 1e-9 && sum > 0.0);
```

The `reactor` example (`cargo run --example reactor -p tpt-sci-kinetics`) wires
an Arrhenius-rate reaction system into a temperature-driven solve.

## Scope (v1)

Arrhenius temperature dependence and Langmuir–Hinshelwood surface coverage, the
two building blocks most reactor/catalysis models need on top of plain
mass-action CRNs. Detailed micro-kinetic mechanisms (multiple site types, Eley–
Rideal, coverage-dependent `Ea`) are out of scope for v1.

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
