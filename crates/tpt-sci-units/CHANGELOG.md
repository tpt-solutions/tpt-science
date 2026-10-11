# Changelog

All notable changes to `tpt-sci-units` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Initial release: typed quantities (`Kelvin`, `JoulePerMole`, `JoulePerMoleKelvin`,
  `WattPerSquareMetre`, `WattPerSquareMetreKelvin4`, `Kilometre`,
  `KilometreCubedPerSecondSquared`), the shared constants `R_GAS`,
  `STEFAN_BOLTZMANN`, `EARTH_MU` and `EARTH_RADIUS_EQ`, and the helpers
  `arrhenius_factor` and `radiative_exitance`.
- `tpt-sci-kinetics`, `tpt-sci-climate` and `tpt-sci-astro` take their constants from
  this crate.

## [0.1.0]

- Not released (`publish = false`).
