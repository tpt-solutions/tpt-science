# tpt-sci-kinetics

Surface and heterogeneous chemical kinetics for the `tpt-science` pillar, built
from scratch on top of [`tpt-sci-reaction-network`](https://docs.rs/tpt-sci-reaction-network)
(the mass-action CRN engine) and [`tpt-sci-ode`](https://docs.rs/tpt-sci-ode).

## What's here

- `ArrheniusRate` — `k(T) = A·exp(-Ea/(R·T))`, validated, temperature-driven.
- `langmuir_hinshelwood_coverages` — single-site fractional coverages from
  adsorption equilibria + partial pressures.
- `KineticsProblem` — binds an Arrhenius `ReactionSystem` to a temperature.

## Scope (v1)

Arrhenius temperature dependence and Langmuir–Hinshelwood surface coverage, the
two building blocks most reactor/catalysis models need on top of plain
mass-action CRNs. Detailed micro-kinetic mechanisms (multiple site types, Eley–
Rideal, coverage-dependent `Ea`) are out of scope for v1.

Dual-licensed under MIT OR Apache-2.0.
