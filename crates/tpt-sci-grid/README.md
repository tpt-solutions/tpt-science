# tpt-sci-grid

Structured finite-difference grids and stencils for the `tpt-science` pillar:
uniform 1-D and tensor-product N-D grids, finite-difference stencil
coefficients, and assembly of discrete PDE operators (Laplacians) on those
grids.

Built from scratch (no wrap target). Motivated by the compartmental-ODE /
structured-grid spatial-model needs of `tpt-soma` and `tpt-cerebrum`
(reaction–diffusion, cable-equation / cortical-sheet style models).

The assembled operators are returned as `DMatrix` / `DVector` from the
`tpt-math` dense linear-algebra substrate, an in-house, dual-licensed
(MIT OR Apache-2.0) backend with no `nalgebra`/`faer` license exposure (ADR 0007).

Depends on `tpt-math-linalg` (published).

## Features

* Uniform 1-D, 2-D and 3-D tensor-product grids
  (`UniformGrid1D` / `UniformGrid2D` / `UniformGrid3D`).
* Discrete Laplacians in 1-D, 2-D and 3-D (`laplacian_1d` / `laplacian_2d` /
  `laplacian_3d`), with homogeneous Dirichlet or Neumann boundaries.
* Feature-gated sparse backend (enable the `sparse` feature): a CSR
  `CsrMatrix` plus sparse Laplacian assemblers (`laplacian_1d_sparse`,
  `laplacian_2d_sparse`, `laplacian_3d_sparse`) and an explicit-Euler
  `diffuse_step`, so realistically sized 2-D/3-D PDE grids avoid the O(n²)
  memory cost of the dense operators.

## Example

```rust
use tpt_sci_grid::{laplacian_1d, UniformGrid1D, Boundary};

let g = UniformGrid1D::new(11, 0.0, 1.0).unwrap();
let l = laplacian_1d(&g, Boundary::Dirichlet);
assert_eq!(l.nrows(), 11);
assert_eq!(l.ncols(), 11);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
