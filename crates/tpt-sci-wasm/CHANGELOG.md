# Changelog

All notable changes to `tpt-sci-wasm` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Initial WebAssembly bindings: `run_spec` (declarative reaction-network runs) and
  `ebm_equilibrium_temperature` (0-D energy-balance model).
- `ct_phantom`, `ct_reconstruct` (naive and filtered back-projection) and
  `lid_cavity_speed` (lid-driven cavity speed field).
- Browser playground at `www/index.html`: runs a reaction-network spec, the
  energy-balance model, the CT reconstruction and the lid-driven cavity. Build the glue with
  `wasm-bindgen --target web --out-dir www/pkg` from the wasm32 release build.

## [0.1.0]

- Not released (`publish = false`).
