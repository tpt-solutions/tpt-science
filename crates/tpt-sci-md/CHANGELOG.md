# Changelog

All notable changes to `tpt-sci-md` are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- `Particle` — point particle with position/velocity/force/mass/species,
  validated construction (`new`, `new_with_species`).
- `lennard_jones` / `Forces::lennard_jones` — pairwise Lennard-Jones 12-6
  interactions with cut-off + shift and minimum-image periodic boundaries.
- `Integrator` — velocity-Verlet stepping with kinetic-energy, temperature, and a
  Berendsen-style thermostat; `velocity_verlet`, `temperature`, `thermostat`.
- `rdf` — radial distribution function `g(r)` for structural analysis.

## [0.1.0] — 2026-08-16

Initial implementation of `tpt-sci-md` (spec2.txt expanded vision).

### Added

- From-scratch classical molecular dynamics engine (no wrapped engine; `lumol` was
  audited and rejected, BSD-3-Clause and alpha/stale) on `tpt-math-linalg`.
  Models mono-/few-species Lennard-Jones fluids in a cubic periodic box.

### Scope (v1)

- Mono/few-species Lennard-Jones fluids in a cubic periodic box. EAM, long-range
  electrostatics (PPPM), constrained bonds, and neighbour lists are out of scope
  for v1.

### Notes

- Crate is `publish = false` and consumed as a path/workspace dependency; no
  crates.io release has been cut yet.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
