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
