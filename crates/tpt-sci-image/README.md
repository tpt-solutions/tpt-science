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
reconstructed independently (a correct parallel-beam volume geometry).

A separate `cone_beam` module provides general cone-beam forward/back
projection alongside that parallel-beam volume path (not a replacement for
it):

* `cone_beam::ConeBeamGeometry` — a point-source/flat-detector acquisition
  geometry (source-to-isocenter and isocenter-to-detector distances,
  detector pixel dimensions/spacing, angle count).
* `cone_beam::cone_beam_forward_projection` — ray-marches, with trilinear
  interpolation, from the source through every detector pixel to produce the
  divergent-ray projections.
* `cone_beam::fdk_reconstruction` — the Feldkamp-Davis-Kress (FDK) algorithm:
  cosine-weight each projection for ray divergence, ram-lak filter it along
  the detector row (reusing the same filter machinery as the parallel-beam
  paths), and back-project along the divergent rays with an inverse-square
  distance weight (Feldkamp, Davis & Kress, "Practical cone-beam algorithm,"
  *J. Opt. Soc. Am. A* 1(6), 612-619, 1984).

Everything is hand-rolled: rotations/interpolation are done by coordinate
remapping with bilinear sampling, and the ramp filter lives in the Fourier
domain via `tpt-math-signal-fft`. There are no external image or tomography
dependencies.

All the transform/reconstruction entry points (`radon_transform`,
`filtered_back_projection`, `naive_back_projection`, the `volume::*`
equivalents, and the `cone_beam::*` cone-beam/FDK functions) return
`Result<_, ImageError>` — see `error.rs` — rather than silently accepting
malformed input (empty images, mismatched angle counts, invalid geometry).

Scope for the 2-D API is **2-D parallel-beam CT**; the `volume` module
provides the 3-D parallel-beam volume support, and `cone_beam` provides
general cone-beam (FDK) reconstruction as a separate code path.

Depends on `tpt-math-signal-fft`, `tpt-math-linalg` (published).

## Example

```rust
use tpt_sci_image::{linspace, radon_transform};
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

// A 4x4 image with a single bright pixel at the centre.
let image = DMatrix::from_fn(4, 4, |i, j| if i == 1 && j == 1 { 1.0 } else { 0.0 });
let angles = linspace(0.0, std::f64::consts::PI, 8);
let sinogram = radon_transform(&image, &angles).unwrap();

// The sinogram has one row per angle and one column per detector bin.
assert_eq!(sinogram.nrows(), 8);
assert_eq!(sinogram.ncols(), 4);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
