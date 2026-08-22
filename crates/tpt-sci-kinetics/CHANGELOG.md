# Changelog

All notable changes to `tpt-sci-kinetics` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-kinetics` to crates.io (spec2.txt expanded vision).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- Surface and heterogeneous chemical kinetics built from scratch on
  `tpt-sci-reaction-network` (mass-action CRN engine) and `tpt-sci-ode`.
- `ArrheniusRate` — `k(T) = A·exp(-Ea/(R·T))`, validated, temperature-driven
  (`rate_constant`).
- `langmuir_hinshelwood_coverages` — single-site fractional surface coverages
  from adsorption equilibria + partial pressures (sum to ≤ 1).
- `KineticsProblem` — binds an Arrhenius `ReactionSystem` to a temperature
  (`rate_constants`).
- `R_GAS` universal gas constant constant.

### Scope (v1)

- Arrhenius temperature dependence and Langmuir–Hinshelwood surface coverage, the
  two building blocks most reactor/catalysis models need on top of plain
  mass-action CRNs. Detailed micro-kinetic mechanisms (multiple site types,
  Eley–Rideal, coverage-dependent `Ea`) are out of scope for v1.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
