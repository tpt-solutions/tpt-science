# Changelog

All notable changes to `tpt-sci-reaction-network` are documented here. This
project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [Unreleased]

### Added

- `simulate_ssa_ensemble`: runs independent SSA trajectories, each seeded from a
  stream derived from a base seed. Runs execute in parallel under the `rayon`
  feature. The output is the same either way.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-reaction-network` to crates.io. Promoted from
`flagged-deferred` after a dedicated ecosystem research pass found no
dual-licensed Rust crate that generates mass-action ODE RHS from a CRN IR
(Catalyst.jl equivalent does not exist in Rust).

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- From-scratch Catalyst.jl-style species/rate/stoichiometry DSL and mass-action
  ODE builder, depending on `tpt-sci-ode`:
  - Programmatic `ReactionNetwork` builder and a textual (`kB, S + E --> SE`)
    DSL.
  - Compiled `ReactionSystem` exposing the stoichiometry matrix `S` and
    per-reaction rate vector `r`, with custom (non-mass-action) rate laws.
  - `OdeProblem` bridge into `tpt-sci-ode` (`ReactionSystem::to_ode_problem`).
  - Stochastic backend `ReactionSystem::simulate_ssa` (Gillespie's direct
    method) with `SsaTrajectory`.

### Changed

- Removed dead `ReactionNetworkError::DuplicateSpecies` / `::DuplicateParameter`
  variants (species/parameter registration is idempotent).

### Out of v1 (documented, not built)

- SDE/jump models, SBML I/O, network analysis, and conservation-law elimination.
  (`rebop` is the intended wrap target for a future stochastic backend; it is now
  implemented from scratch instead.)

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
