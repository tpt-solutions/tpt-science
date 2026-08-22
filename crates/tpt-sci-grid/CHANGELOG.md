# Changelog

All notable changes to `tpt-sci-grid` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-grid` to crates.io.

### Added

- Uniform 1-D, 2-D and 3-D tensor-product grids (`UniformGrid1D` /
  `UniformGrid2D` / `UniformGrid3D`), `linspace` helper.
- Discrete Laplacians in 1-D, 2-D and 3-D (`laplacian_1d` / `laplacian_2d` /
  `laplacian_3d`), with homogeneous Dirichlet or Neumann boundaries.
- `UniformGrid3D` plus dense `laplacian_3d` and sparse `laplacian_3d_sparse`
  (shared row assembly so dense == sparse), with both Dirichlet and Neumann
  boundaries.
- Feature-gated sparse backend (`sparse` feature): `CsrMatrix`,
  `laplacian_1d_sparse` / `laplacian_2d_sparse` / `laplacian_3d_sparse`, and an
  explicit-Euler `diffuse_step`.
- Finite-difference `Stencil` enum and `derivative_1d`, plus a `kron` tensor
  product for building higher-dimensional operators.
- Assembled operators returned as `DMatrix` / `DVector` from the in-house,
  dual-licensed `tpt-math` dense linear-algebra substrate (no `nalgebra`/`faer`).

### Changed

- `operator::derivative_1d` now consumes the `Stencil` enum (previously dead
  code), so the advertised finite-difference stencils are actually available.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
