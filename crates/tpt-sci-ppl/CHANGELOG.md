# Changelog

All notable changes to `tpt-sci-ppl` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-ppl` to crates.io.

### Added

- Criterion benchmark suite (/benches/) covering the crate's core hot path.
- From-scratch NUTS (No-U-Turn Sampler) Hamiltonian Monte Carlo backend on
  `tpt-math-autodiff-rev` (reverse-mode gradients) and `tpt-math-prob`
  (randomness). The `nuts-rs` wrap planned in `spec.txt` was dropped per the
  "build our own internals" direction.
- `Model` / `ModelBuilder` model/DSL layer: differentiable priors and likelihood
  closures, `Model::build`, `Model::fit`.
- `Trace` convergence diagnostics: split-R-hat (`Trace::rhat`), Geyer ESS
  (`Trace::ess`), and divergence rate / count (`Trace::divergence_rate`,
  `Trace::n_divergences`). `Model::fit` / `Model::fit_chains` return a `Trace`.
- `fit_chains` for multi-chain fits; `Model::fit` now delegates to
  `fit_chains(2, …)` so every returned trace carries a meaningful split-R-hat.
- Typed priors from `tpt-math-prob-bayes` (`Gaussian`, `Beta`, `Gamma`) via
  `ModelBuilder::gaussian_prior` / `beta_prior` / `gamma_prior`.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
