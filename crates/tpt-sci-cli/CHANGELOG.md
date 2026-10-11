# Changelog

All notable changes to `tpt-sci-cli` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Initial release of the `tpt-sci` command: `tpt-sci run <spec.json> [--out file]`
  integrates a reaction network described in a JSON spec and writes the
  trajectory as CSV.
- Library functions `run`, `run_to_csv` and `parse_method`, and the `Spec` type.
- `examples/sir.json`, an SIR run.

## [0.1.0]

- Not released (`publish = false`).
