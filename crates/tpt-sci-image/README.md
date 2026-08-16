# tpt-sci-image

2-D tomographic reconstruction built from scratch.

This crate implements the core of parallel-beam computed tomography on top of
the in-house `tpt-math` substrate:

* `radon_transform` — the forward Radon transform (a sinogram).
* `filtered_back_projection` — the classic ram-lak filtered back projection
  (FBP) inverse.
* `naive_back_projection` — the unfiltered adjoint, shown for comparison.

Everything is hand-rolled: rotations/interpolation are done by coordinate
remapping with bilinear sampling, and the ramp filter lives in the Fourier
domain via `tpt-math-signal-fft`. There are no external image or tomography
dependencies.

Scope is **2-D parallel-beam CT** rather than fully n-dimensional (the
"n-dimensional" wording in the original plan is not met; revisit if a vertical
needs 3-D volumes).

Depends on `tpt-math-signal-fft`, `tpt-math-linalg` (published).

## Example

```rust
use tpt_sci_image::{linspace, radon_transform};
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

// A 4x4 image with a single bright pixel at the centre.
let image = DMatrix::from_fn(4, 4, |i, j| if i == 1 && j == 1 { 1.0 } else { 0.0 });
let angles = linspace(0.0, std::f64::consts::PI, 8);
let sinogram = radon_transform(&image, &angles);

// The sinogram has one row per angle and one column per detector bin.
assert_eq!(sinogram.nrows(), 8);
assert_eq!(sinogram.ncols(), 4);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
