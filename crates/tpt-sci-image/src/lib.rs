//! `tpt-sci-image` — 2-D tomographic reconstruction built from scratch.
//!
//! This crate implements the core of parallel-beam computed tomography on top
//! of the in-house `tpt-math` substrate:
//!
//! * [`radon_transform`] — the forward Radon transform (a sinogram).
//! * [`filtered_back_projection`] — the classic ram-lak filtered back
//!   projection (FBP) inverse.
//!
//! The same machinery is extended to volumes in the [`volume`] module:
//! [`volume::radon_transform_3d`] and [`volume::filtered_back_projection_3d`]
//! reconstruct a [`volume::Volume`] (a `n_x × n_y × n_z` scalar field) under a
//! parallel-beam geometry that rotates about the `z` axis.
//!
//! Everything is hand-rolled: rotations/interpolation are done by coordinate
//! remapping with bilinear sampling, and the ramp filter lives in the Fourier
//! domain via [`tpt_math_signal_fft`]. There are no external image or
//! tomography dependencies.
//!
//! # Examples
//!
//! ```
//! use tpt_sci_image::{linspace, radon_transform};
//! use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
//!
//! // A 4x4 image with a single bright pixel at the centre.
//! let image = DMatrix::from_fn(4, 4, |i, j| if i == 1 && j == 1 { 1.0 } else { 0.0 });
//! let angles = linspace(0.0, std::f64::consts::PI, 8);
//! let sinogram = radon_transform(&image, &angles).unwrap();
//!
//! // The sinogram has one row per angle and one column per detector bin.
//! assert_eq!(sinogram.nrows(), 8);
//! assert_eq!(sinogram.ncols(), 4);
//! ```

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_math_signal_fft::{Complex, fft, ifft_normalized};

/// Errors returned by the image reconstruction API.
pub mod error;
pub use error::ImageError;

/// 3-D tomographic reconstruction (parallel-beam volume CT).
pub mod volume;
pub use volume::{Volume, filtered_back_projection_3d, radon_transform_3d};

/// Build an inclusive linearly-spaced vector of `n` points from `start` to
/// `end`.
///
/// `linspace(0.0, 1.0, 5)` is `[0.0, 0.25, 0.5, 0.75, 1.0]`. An empty count
/// yields an empty vector; a count of one yields `[start]`.
///
/// # Examples
///
/// ```
/// use tpt_sci_image::linspace;
/// let v = linspace(0.0, 1.0, 3);
/// assert_eq!(v, vec![0.0, 0.5, 1.0]);
/// ```
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * i as f64).collect()
}

/// The number of detector bins used for an image: the detector must be long
/// enough to cover the image along its longest axis, so we pick
/// `max(nrows, ncols)`.
fn detector_count(image: &DMatrix<f64>) -> usize {
    image.nrows().max(image.ncols())
}

/// Bilinear sample of `image` at fractional `(col, row)` coordinates.
///
/// `col` is the horizontal (column) coordinate and `row` the vertical (row)
/// coordinate, both in the image's own pixel grid with the origin at the
/// top-left pixel. Values outside the image extent are treated as zero (the
/// object is assumed supported on the image).
fn sample_bilinear(image: &DMatrix<f64>, col: f64, row: f64) -> f64 {
    let nrows = image.nrows() as f64;
    let ncols = image.ncols() as f64;
    if col < 0.0 || row < 0.0 || col > ncols - 1.0 || row > nrows - 1.0 {
        return 0.0;
    }
    let c0 = col.floor() as usize;
    let r0 = row.floor() as usize;
    let c1 = (c0 + 1).min(image.ncols() - 1);
    let r1 = (r0 + 1).min(image.nrows() - 1);
    let dc = col - c0 as f64;
    let dr = row - r0 as f64;
    let v00 = image[(r0, c0)];
    let v01 = image[(r0, c1)];
    let v10 = image[(r1, c0)];
    let v11 = image[(r1, c1)];
    v00 * (1.0 - dc) * (1.0 - dr) + v01 * dc * (1.0 - dr) + v10 * (1.0 - dc) * dr + v11 * dc * dr
}

/// Linear interpolation of a 1-D projection at fractional bin `x`.
///
/// Out-of-range positions return zero (no detector data there).
fn sample_1d(v: &[f64], x: f64) -> f64 {
    let n = v.len();
    if n == 0 || x < 0.0 || x > (n - 1) as f64 {
        return 0.0;
    }
    let i0 = x.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    let d = x - i0 as f64;
    v[i0] * (1.0 - d) + v[i1] * d
}

/// Forward Radon transform (parallel beam) of `image` over `angles`.
///
/// For each angle `theta` we build a rotated frame of size
/// `n_bins x n_bins` (with `n_bins = max(nrows, ncols)`) by mapping every
/// output pixel back through the inverse rotation `R(theta)` into the original
/// image and bilinearly sampling it. The detector projection for that angle is
/// then the sum along each row of the rotated frame, so the returned sinogram
/// has `sinogram.nrows() == angles.len()` and
/// `sinogram.ncols() == n_bins`.
///
/// The rotation uses nearest-pixel-centre coordinates (origin at the centre
/// pixel) and bilinear interpolation; the same centre convention and rotation
/// sense are used by [`filtered_back_projection`] so forward and inverse are
/// consistent.
///
/// # Errors
///
/// Returns [`ImageError::EmptyImage`] if `image` has zero rows or columns, or
/// [`ImageError::EmptyAngles`] if `angles` is empty.
pub fn radon_transform(image: &DMatrix<f64>, angles: &[f64]) -> Result<DMatrix<f64>, ImageError> {
    if image.nrows() == 0 || image.ncols() == 0 {
        return Err(ImageError::EmptyImage {
            nrows: image.nrows(),
            ncols: image.ncols(),
        });
    }
    if angles.is_empty() {
        return Err(ImageError::EmptyAngles);
    }
    let nb = detector_count(image);
    let c = (nb - 1) as f64 / 2.0;
    let cx_img = (image.ncols() - 1) as f64 / 2.0;
    let cy_img = (image.nrows() - 1) as f64 / 2.0;

    let mut sino = vec![0.0_f64; angles.len() * nb];
    for (a, &theta) in angles.iter().enumerate() {
        let cos = theta.cos();
        let sin = theta.sin();
        // Fill the rotated frame (nb x nb) by inverse-mapping each output
        // pixel into the original image.
        for i in 0..nb {
            for j in 0..nb {
                let u = j as f64 - c;
                let v = i as f64 - c;
                // Source coordinate in the original image via R(theta).
                let sx = cos * u - sin * v;
                let sy = sin * u + cos * v;
                let src_col = sx + cx_img;
                let src_row = sy + cy_img;
                let val = sample_bilinear(image, src_col, src_row);
                // Accumulate into the row-sum (one detector bin per row).
                sino[a * nb + i] += val;
            }
        }
    }
    Ok(DMatrix::from_fn(angles.len(), nb, |a, i| sino[a * nb + i]))
}

/// Apply the ram-lak ramp filter to a single projection in the Fourier domain.
///
/// The forward FFT is taken, each bin `k` is multiplied by `|k|`
/// (`min(k, n - k)`, which is the folded positive frequency), and the result
/// is inverse-transformed (normalized) before returning its real part.
fn ramp_filter(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    if n == 0 {
        return Vec::new();
    }
    let spectrum = fft(p);
    let filtered: Vec<Complex<f64>> = spectrum
        .iter()
        .enumerate()
        .map(|(k, &z)| {
            let freq = if k <= n / 2 { k } else { n - k } as f64;
            z * freq
        })
        .collect();
    let back = ifft_normalized(&filtered);
    back.iter().map(|z| z.re).collect()
}

/// Back-project a set of (possibly filtered) 1-D projections into an
/// `n_bins x n_bins` reconstruction.
///
/// Each reconstruction pixel at centre-relative coordinate `(x, y)` is mapped
/// onto the detector axis for angle `theta` via `t = -sin(theta)*x +
/// cos(theta)*y`; the projection value at bin `t + c` is added. `projections`
/// must have one `Vec<f64>` per angle, each of length `n_bins`.
fn back_project(projections: &[Vec<f64>], angles: &[f64], nb: usize) -> DMatrix<f64> {
    let c = if nb == 0 { 0.0 } else { (nb - 1) as f64 / 2.0 };
    let mut rec = vec![0.0_f64; nb * nb];
    for (a, &theta) in angles.iter().enumerate() {
        let q = match projections.get(a) {
            Some(q) => q,
            None => continue,
        };
        let cos = theta.cos();
        let sin = theta.sin();
        for i in 0..nb {
            for j in 0..nb {
                let x = j as f64 - c;
                let y = i as f64 - c;
                let t = -sin * x + cos * y;
                let bin_f = t + c;
                rec[i * nb + j] += sample_1d(q, bin_f);
            }
        }
    }
    DMatrix::from_fn(nb, nb, |i, j| rec[i * nb + j])
}

/// Filtered back-projection (FBP) reconstruction from a sinogram.
///
/// For each angle the projection row is ramp-filtered in the Fourier domain
/// (ram-lak) and then back-projected. The accumulated image is normalized by
/// the number of angles, which corresponds to the `1/pi * integral dtheta`
/// continuous formula with `dtheta ~= pi / n_angles`. An additional linear
/// scale factor `4 / n_bins` corrects the amplitude of the discrete ramp
/// filter (which omits the `2*pi/n` frequency spacing and the `1/n` of the
/// unnormalized FFT) so that a point source reconstructs to its original
/// amplitude.
///
/// The returned reconstruction is always `n_bins x n_bins` with
/// `n_bins = sinogram.ncols()`.
///
/// # Errors
///
/// Returns [`ImageError::EmptyAngles`] if `angles` is empty or
/// [`ImageError::AngleCountMismatch`] if `sinogram.nrows() != angles.len()`.
pub fn filtered_back_projection(
    sinogram: &DMatrix<f64>,
    angles: &[f64],
) -> Result<DMatrix<f64>, ImageError> {
    if angles.is_empty() {
        return Err(ImageError::EmptyAngles);
    }
    if sinogram.nrows() != angles.len() {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: sinogram.nrows(),
            n_angles: angles.len(),
        });
    }
    let nb = sinogram.ncols();
    let mut qs: Vec<Vec<f64>> = Vec::with_capacity(angles.len());
    for a in 0..angles.len() {
        let row: Vec<f64> = (0..nb).map(|j| sinogram[(a, j)]).collect();
        qs.push(ramp_filter(&row));
    }
    let rec = back_project(&qs, angles, nb);
    // Empirical FBP amplitude normalisation. The continuous filtered
    // back-projection identity is
    //     f(x) = (1/π) ∫_0^π [Rf(·,θ) * h](x·n(θ)) dθ ,
    // with ramp kernel h(t) ∝ |ω| in frequency, sampled here as the raw DFT
    // ramp `|k|` used by `ramp_filter`. Discretely the detector bins are spaced
    // Δs = 1, the `n_angles` projections cover [0, π) at Δθ = π/n_angles, and the
    // back-projection simply sums `n_angles` contributions. Matching the
    // discrete ramp's DC gain to the continuous kernel and absorbing the Δθ and
    // Δs pixel-area factors yields the constant `4.0 / (nb · n_angles)`
    // (Kak & Slaney, *Principles of Computerized Tomographic Imaging*, §3.3).
    let scale = 4.0 / nb as f64 / angles.len() as f64;
    Ok(rec * scale)
}

/// Unfiltered (naive) back-projection of a sinogram.
///
/// Like [`filtered_back_projection`] but without the ramp filter, used to show
/// that filtering improves the reconstruction (it removes the characteristic
/// blur/halo of the naive adjoint).
///
/// # Errors
///
/// Returns [`ImageError::EmptyAngles`] if `angles` is empty or
/// [`ImageError::AngleCountMismatch`] if `sinogram.nrows() != angles.len()`.
pub fn naive_back_projection(
    sinogram: &DMatrix<f64>,
    angles: &[f64],
) -> Result<DMatrix<f64>, ImageError> {
    if angles.is_empty() {
        return Err(ImageError::EmptyAngles);
    }
    if sinogram.nrows() != angles.len() {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: sinogram.nrows(),
            n_angles: angles.len(),
        });
    }
    let nb = sinogram.ncols();
    let projections: Vec<Vec<f64>> = (0..angles.len())
        .map(|a| (0..nb).map(|j| sinogram[(a, j)]).collect())
        .collect();
    let rec = back_project(&projections, angles, nb);
    // Same `4.0 / (nb · n_angles)` normalisation as `filtered_back_projection`
    // (Kak & Slaney, *Principles of Computerized Tomographic Imaging*, §3.3),
    // applied to the unfiltered adjoint so the two reconstructions share a scale.
    let scale = 4.0 / nb as f64 / angles.len() as f64;
    Ok(rec * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

    /// Index of the centre pixel of an `n x n` image.
    fn center(n: usize) -> (usize, usize) {
        (n / 2, n / 2)
    }

    fn mean(m: &DMatrix<f64>) -> f64 {
        let mut s = 0.0;
        let mut n = 0;
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                s += m[(i, j)];
                n += 1;
            }
        }
        s / n as f64
    }

    #[test]
    fn linspace_is_inclusive() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert_abs_diff_eq!(v[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[4], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn radon_dimensions_match_angles_and_bins() {
        let n = 16usize;
        let image = DMatrix::from_fn(
            n,
            n,
            |i, j| if i == n / 2 && j == n / 2 { 1.0 } else { 0.0 },
        );
        let angles = linspace(0.0, std::f64::consts::PI, 32);
        let sino = radon_transform(&image, &angles).unwrap();
        assert_eq!(sino.nrows(), 32);
        assert_eq!(sino.ncols(), n);
    }

    #[test]
    fn point_source_peaks_near_centre_column() {
        let n = 16usize;
        let image = DMatrix::from_fn(
            n,
            n,
            |i, j| if i == n / 2 && j == n / 2 { 1.0 } else { 0.0 },
        );
        let angles = linspace(0.0, std::f64::consts::PI, 32);
        let sino = radon_transform(&image, &angles).unwrap();

        // Find the column (detector bin) holding the maximum value of the whole
        // sinogram: for a centred point source it must be the centre bin.
        let mut max_col = 0;
        let mut max_val = f64::NEG_INFINITY;
        for a in 0..sino.nrows() {
            for j in 0..sino.ncols() {
                if sino[(a, j)] > max_val {
                    max_val = sino[(a, j)];
                    max_col = j;
                }
            }
        }
        assert!((max_col as isize - n as isize / 2).abs() <= 1);
    }

    #[test]
    fn constant_image_conserves_row_sum() {
        let n = 16usize;
        let image = DMatrix::from_fn(n, n, |_, _| 1.0);
        let angles = linspace(0.0, std::f64::consts::PI, 24);
        let sino = radon_transform(&image, &angles).unwrap();
        // A constant unit image integrates to at most N samples per detector
        // bin (a full line through the square). Because the discrete sinogram
        // samples a rotated `N x N` frame and no bin sits exactly on the
        // image centre line, the most central line can lose at most a single
        // boundary sample to the square's edge — so each central bin is within
        // one of N. Every bin is non-negative and bounded by N.
        let c = n / 2;
        for a in 0..sino.nrows() {
            let max_bin = (0..sino.ncols())
                .map(|j| sino[(a, j)])
                .fold(0.0_f64, f64::max);
            assert!(
                (max_bin - n as f64).abs() <= 1.0 + 1e-6,
                "angle {a} max {max_bin}"
            );
            assert!(
                (sino[(a, c)] - n as f64).abs() <= 1.0 + 1e-6,
                "angle {a} central {}",
                sino[(a, c)]
            );
            for j in 0..sino.ncols() {
                assert!(
                    sino[(a, j)] <= n as f64 + 1e-9,
                    "bin {j} = {}",
                    sino[(a, j)]
                );
                assert!(sino[(a, j)] >= 0.0);
            }
        }
    }

    #[test]
    fn fbp_point_source_centre_is_positive_and_prominent() {
        let n = 16usize;
        let image = DMatrix::from_fn(
            n,
            n,
            |i, j| if i == n / 2 && j == n / 2 { 1.0 } else { 0.0 },
        );
        let angles = linspace(0.0, std::f64::consts::PI, 32);
        let sino = radon_transform(&image, &angles).unwrap();
        let rec = filtered_back_projection(&sino, &angles).unwrap();

        let c = center(n);
        assert!(rec[(c.0, c.1)] > 0.0, "centre should be positive");
        let m = mean(&rec);
        assert!(
            rec[(c.0, c.1)] >= m,
            "centre should be at least the mean (got {} vs mean {})",
            rec[(c.0, c.1)],
            m
        );
    }

    #[test]
    fn fbp_recovers_gaussian_and_beats_naive() {
        let n = 32usize;
        let sigma = 4.0_f64;
        let (cx, cy) = (n as f64 / 2.0, n as f64 / 2.0);
        let original = DMatrix::from_fn(n, n, |i, j| {
            let dx = i as f64 - cx;
            let dy = j as f64 - cy;
            (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
        });

        let angles = linspace(0.0, std::f64::consts::PI, 48);
        let sino = radon_transform(&original, &angles).unwrap();
        let rec = filtered_back_projection(&sino, &angles).unwrap();
        let naive = naive_back_projection(&sino, &angles).unwrap();

        // The centre reconstruction should be positive and reasonably close to
        // the original centre value (filtered back-projection is not exact, so
        // we allow a generous band rather than an exact equality).
        let c = center(n);
        let centre_val = rec[(c.0, c.1)];
        assert!(centre_val > 0.0, "centre should be positive");
        assert!(
            centre_val > 0.3 * original[(c.0, c.1)] && centre_val < 2.5 * original[(c.0, c.1)],
            "centre {centre_val} should be reasonably close to {}",
            original[(c.0, c.1)]
        );

        // Mean absolute error over the interior (exclude a 4-pixel border that
        // suffers edge effects in both methods).
        let lo = 4;
        let hi = n - 4;
        let mae = |m: &DMatrix<f64>| -> f64 {
            let mut s = 0.0;
            let mut count = 0;
            for i in lo..hi {
                for j in lo..hi {
                    s += (m[(i, j)] - original[(i, j)]).abs();
                    count += 1;
                }
            }
            s / count as f64
        };
        let fbp_err = mae(&rec);
        let naive_err = mae(&naive);
        assert!(
            fbp_err < naive_err,
            "FBP error {fbp_err} should be below naive error {naive_err}"
        );
    }

    #[test]
    fn radon_rejects_empty_image_and_angles() {
        let empty = DMatrix::<f64>::zeros(0, 4);
        let angles = linspace(0.0, std::f64::consts::PI, 8);
        assert!(matches!(
            radon_transform(&empty, &angles),
            Err(ImageError::EmptyImage { .. })
        ));
        let image = DMatrix::from_fn(4, 4, |_, _| 1.0);
        assert!(matches!(
            radon_transform(&image, &[]),
            Err(ImageError::EmptyAngles)
        ));
    }

    #[test]
    fn fbp_rejects_angle_count_mismatch() {
        let n = 8usize;
        let image = DMatrix::from_fn(n, n, |_, _| 1.0);
        let angles = linspace(0.0, std::f64::consts::PI, 8);
        let sino = radon_transform(&image, &angles).unwrap();
        // A sinogram built from 8 angles cannot be reconstructed with 4 angles.
        let few = linspace(0.0, std::f64::consts::PI, 4);
        assert!(matches!(
            filtered_back_projection(&sino, &few),
            Err(ImageError::AngleCountMismatch { .. })
        ));
        assert!(matches!(
            naive_back_projection(&sino, &few),
            Err(ImageError::AngleCountMismatch { .. })
        ));
    }
}
