# tpt-sci-image

2-D tomographic reconstruction built from scratch.

This crate implements the core of parallel-beam computed tomography on top of
the in-house `tpt-math` substrate:

* `radon_transform` — the forward Radon transform (a sinogram).
* `filtered_back_projection` — the classic ram-lak filtered back projection
  (FBP) inverse.
* `naive_back_projection` — the unfiltered adjoint, shown for comparison.

In addition, the `volume` module extends the same machinery to 3-D parallel-beam
CT:

* `volume::radon_transform_3d` — the forward Radon transform of a `Volume`
  (one `n_z × n_bins` sinogram per angle).
* `volume::filtered_back_projection_3d` — the ram-lak FBP inverse, returning a
  reconstructed `Volume`.

The 3-D acquisition rotates the beam about the `z` axis, so each `z` slice is
reconstructed independently (a correct parallel-beam volume geometry; it does
not model a general cone beam).

Everything is hand-rolled: rotations/interpolation are done by coordinate
remapping with bilinear sampling, and the ramp filter lives in the Fourier
domain via `tpt-math-signal-fft`. There are no external image or tomography
dependencies.

Scope for the 2-D API is **2-D parallel-beam CT**; the `volume` module provides
the 3-D volume support that was previously deferred.

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
