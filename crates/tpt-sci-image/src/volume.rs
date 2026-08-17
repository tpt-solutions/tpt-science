//! 3-D tomographic reconstruction: parallel-beam computed tomography of a volume.
//!
//! This extends the 2-D machinery in the crate root to a [`Volume`]. The
//! acquisition rotates the beam about the `z` axis, so each `z` slice is
//! independently Radon-transformed and reconstructed — the 3-D filtered
//! back-projection is exactly a stack of independent 2-D FBP reconstructions
//! (the `z` coordinate is invariant under rotation about `z`). That is a
//! standard, correct parallel-beam volume geometry (it does not model a generic
//! cone beam).
//!
//! * [`radon_transform_3d`] — the forward Radon transform of a [`Volume`]
//!   (one `n_z × n_bins` sinogram [`DMatrix`] per projection angle).
//! * [`filtered_back_projection_3d`] — the ram-lak FBP inverse, returning a
//!   reconstructed [`Volume`].

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_math_signal_fft::{Complex, fft, ifft_normalized};

use crate::error::ImageError;

/// A dense 3-D scalar field stored row-major as a flat buffer.
///
/// Voxels are addressed `index = ix + iy·nx + iz·nx·ny` (so `x` is the
/// fastest-varying axis and `z` the slowest), matching the layout of a 3-D
/// structured grid (`nx·ny·nz` nodes).
#[derive(Debug, Clone)]
pub struct Volume {
    /// Node count along `x`.
    pub nx: usize,
    /// Node count along `y`.
    pub ny: usize,
    /// Node count along `z`.
    pub nz: usize,
    /// Flat buffer of `nx·ny·nz` values in `ix + iy·nx + iz·nx·ny` order.
    pub data: Vec<f64>,
}

impl Volume {
    /// Create a zero-filled volume.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            data: vec![0.0; nx * ny * nz],
        }
    }

    /// Build a volume from a closure `f(ix, iy, iz) -> value`.
    #[must_use]
    pub fn from_fn(
        nx: usize,
        ny: usize,
        nz: usize,
        f: impl Fn(usize, usize, usize) -> f64,
    ) -> Self {
        let mut data = vec![0.0; nx * ny * nz];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    data[ix + iy * nx + iz * nx * ny] = f(ix, iy, iz);
                }
            }
        }
        Self { nx, ny, nz, data }
    }

    /// Lexicographic voxel index `ix + iy·nx + iz·nx·ny`.
    #[must_use]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.nx + iz * self.nx * self.ny
    }

    /// Total voxel count `nx·ny·nz`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Whether the volume is empty (`nx·ny·nz == 0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Bilinear sample of the `z` slice of `vol` at fractional `(col, row)` in the
/// `x`/`y` plane. Out-of-extent coordinates return `0.0`.
fn sample_slice(vol: &Volume, z: usize, col: f64, row: f64) -> f64 {
    let nrows = vol.ny as f64;
    let ncols = vol.nx as f64;
    if col < 0.0 || row < 0.0 || col > ncols - 1.0 || row > nrows - 1.0 {
        return 0.0;
    }
    let c0 = col.floor() as usize;
    let r0 = row.floor() as usize;
    let c1 = (c0 + 1).min(vol.nx - 1);
    let r1 = (r0 + 1).min(vol.ny - 1);
    let dc = col - c0 as f64;
    let dr = row - r0 as f64;
    let base = vol.index(0, 0, z);
    let v00 = vol.data[base + c0 + r0 * vol.nx];
    let v01 = vol.data[base + c1 + r0 * vol.nx];
    let v10 = vol.data[base + c0 + r1 * vol.nx];
    let v11 = vol.data[base + c1 + r1 * vol.nx];
    v00 * (1.0 - dc) * (1.0 - dr) + v01 * dc * (1.0 - dr) + v10 * (1.0 - dc) * dr + v11 * dc * dr
}

/// Linear interpolation of a 1-D projection at fractional bin `x`. Returns `0.0`
/// for out-of-range positions.
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

/// Apply the ram-lak ramp filter to a single 1-D projection in the Fourier
/// domain (see the crate-root [`crate::filtered_back_projection`] for the
/// motivation). The returned real part is the filtered projection.
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

/// Forward 3-D Radon transform (parallel beam, rotation about `z`) of `volume`
/// over `angles`.
///
/// Returns one sinogram [`DMatrix`] per angle. Each sinogram has `n_z` rows (one
/// per `z` slice) and `n_bins = max(nx, ny)` columns (the beam-direction
/// detector bins), following the same centre convention and rotation sense as
/// the crate-root [`crate::radon_transform`].
///
/// # Errors
///
/// Returns [`ImageError::EmptyImage`] if the volume has no voxels, or
/// [`ImageError::EmptyAngles`] if `angles` is empty.
pub fn radon_transform_3d(
    volume: &Volume,
    angles: &[f64],
) -> Result<Vec<DMatrix<f64>>, ImageError> {
    if volume.is_empty() {
        return Err(ImageError::EmptyImage {
            nrows: volume.ny,
            ncols: volume.nx,
        });
    }
    if angles.is_empty() {
        return Err(ImageError::EmptyAngles);
    }
    let nb = volume.nx.max(volume.ny);
    let c = (nb - 1) as f64 / 2.0;
    let cx = (volume.nx - 1) as f64 / 2.0;
    let cy = (volume.ny - 1) as f64 / 2.0;

    let mut out = Vec::with_capacity(angles.len());
    for &theta in angles {
        let cos = theta.cos();
        let sin = theta.sin();
        let mut proj = vec![0.0_f64; volume.nz * nb];
        for z in 0..volume.nz {
            for i in 0..nb {
                let u = i as f64 - c;
                for j in 0..nb {
                    let v = j as f64 - c;
                    let sx = cos * u - sin * v;
                    let sy = sin * u + cos * v;
                    proj[z * nb + i] += sample_slice(volume, z, sx + cx, sy + cy);
                }
            }
        }
        out.push(DMatrix::from_fn(volume.nz, nb, |z, i| proj[z * nb + i]));
    }
    Ok(out)
}

/// Filtered back-projection (FBP) reconstruction of a [`Volume`] from a 3-D
/// sinogram stack.
///
/// Each angle's sinogram (one `[n_z × n_bins]` [`DMatrix`] from
/// [`radon_transform_3d`]) is ramp-filtered per `z` row and back-projected. The
/// reconstruction has dimensions `nx × ny × nz`, which must match the volume
/// that produced the sinograms (`n_bins = max(nx, ny)`).
///
/// # Errors
///
/// Returns [`ImageError::EmptyAngles`] if `angles` is empty,
/// [`ImageError::EmptyImage`] if `nx`/`ny`/`nz` is zero,
/// [`ImageError::AngleCountMismatch`] if the sinogram count disagrees with the
/// angle count, or [`ImageError::DimensionMismatch`] if a sinogram's dimensions
/// are not `n_z × max(nx, ny)`.
pub fn filtered_back_projection_3d(
    sinograms: &[DMatrix<f64>],
    angles: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Volume, ImageError> {
    if angles.is_empty() {
        return Err(ImageError::EmptyAngles);
    }
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(ImageError::EmptyImage {
            nrows: ny,
            ncols: nx,
        });
    }
    if sinograms.len() != angles.len() {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: sinograms.len(),
            n_angles: angles.len(),
        });
    }
    let nb = nx.max(ny);
    for s in sinograms {
        if s.ncols() != nb {
            return Err(ImageError::DimensionMismatch {
                expected: nb,
                got: s.ncols(),
            });
        }
        if s.nrows() != nz {
            return Err(ImageError::DimensionMismatch {
                expected: nz,
                got: s.nrows(),
            });
        }
    }

    let c = (nb - 1) as f64 / 2.0;
    let cx = (nx - 1) as f64 / 2.0;
    let cy = (ny - 1) as f64 / 2.0;

    // Pre-filter each sinogram row (per z) once.
    let mut filtered: Vec<Vec<Vec<f64>>> = Vec::with_capacity(sinograms.len());
    for s in sinograms {
        let mut rows = Vec::with_capacity(s.nrows());
        for z in 0..s.nrows() {
            let row: Vec<f64> = (0..s.ncols()).map(|j| s[(z, j)]).collect();
            rows.push(ramp_filter(&row));
        }
        filtered.push(rows);
    }

    let mut rec = vec![0.0_f64; nx * ny * nz];
    for (a, &theta) in angles.iter().enumerate() {
        let cos = theta.cos();
        let sin = theta.sin();
        for iz in 0..nz {
            for iy in 0..ny {
                let y = iy as f64 - cy;
                for ix in 0..nx {
                    let x = ix as f64 - cx;
                    let t = -sin * x + cos * y;
                    let bin = t + c;
                    let val = sample_1d(&filtered[a][iz], bin);
                    rec[ix + iy * nx + iz * nx * ny] += val;
                }
            }
        }
    }

    // Same empirical FBP normalisation as the 2-D API: `4.0 / (nb · n_angles)`
    // matches the discrete ramp filter's DC gain to the continuous kernel and
    // absorbs the Δθ / Δs pixel-area factors (Kak & Slaney, *Principles of
    // Computerized Tomographic Imaging*, §3.3). Each `z` slice is reconstructed
    // independently, so the 1-D detector-bin count `nb` is the slice width.
    let scale = 4.0 / nb as f64 / angles.len() as f64;
    for v in &mut rec {
        *v *= scale;
    }
    Ok(Volume {
        nx,
        ny,
        nz,
        data: rec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radon_3d_dimensions() {
        let v = Volume::from_fn(16, 16, 8, |ix, iy, iz| {
            if ix == 8 && iy == 8 && iz == 4 {
                1.0
            } else {
                0.0
            }
        });
        let angles = crate::linspace(0.0, std::f64::consts::PI, 16);
        let sino = radon_transform_3d(&v, &angles).unwrap();
        assert_eq!(sino.len(), 16);
        for s in &sino {
            assert_eq!(s.nrows(), 8);
            assert_eq!(s.ncols(), 16);
        }
    }

    #[test]
    fn fbp_3d_recovers_point_source() {
        let n = 16usize;
        let v = Volume::from_fn(n, n, n, |ix, iy, iz| {
            if ix == n / 2 && iy == n / 2 && iz == n / 2 {
                1.0
            } else {
                0.0
            }
        });
        let angles = crate::linspace(0.0, std::f64::consts::PI, 32);
        let sino = radon_transform_3d(&v, &angles).unwrap();
        let rec = filtered_back_projection_3d(&sino, &angles, n, n, n).unwrap();

        let c = rec.data[rec.index(n / 2, n / 2, n / 2)];
        assert!(c > 0.0, "centre should be positive");
        let mean = rec.data.iter().sum::<f64>() / rec.data.len() as f64;
        assert!(
            c >= mean,
            "centre should be at least the mean (c={c}, mean={mean})"
        );
    }

    #[test]
    fn fbp_3d_beats_naive_3d() {
        let n = 24usize;
        let sigma = 3.0_f64;
        let (cx, cy, cz) = (n as f64 / 2.0, n as f64 / 2.0, n as f64 / 2.0);
        let orig = Volume::from_fn(n, n, n, |ix, iy, iz| {
            let dx = ix as f64 - cx;
            let dy = iy as f64 - cy;
            let dz = iz as f64 - cz;
            (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
        });
        let angles = crate::linspace(0.0, std::f64::consts::PI, 40);
        let sino = radon_transform_3d(&orig, &angles).unwrap();
        let rec = filtered_back_projection_3d(&sino, &angles, n, n, n).unwrap();

        let naive = naive_back_projection_3d(&sino, &angles, n, n, n);
        let mut fbp_err = 0.0;
        let mut naive_err = 0.0;
        let mut count = 0;
        let b = 3;
        for iz in b..n - b {
            for iy in b..n - b {
                for ix in b..n - b {
                    let k = orig.index(ix, iy, iz);
                    fbp_err += (rec.data[k] - orig.data[k]).abs();
                    naive_err += (naive.data[k] - orig.data[k]).abs();
                    count += 1;
                }
            }
        }
        fbp_err /= count as f64;
        naive_err /= count as f64;
        assert!(
            fbp_err < naive_err,
            "FBP error {fbp_err} should be below naive error {naive_err}"
        );
    }

    #[test]
    fn radon_3d_rejects_empty() {
        let empty = Volume::new(0, 4, 4);
        let angles = crate::linspace(0.0, std::f64::consts::PI, 8);
        assert!(matches!(
            radon_transform_3d(&empty, &angles),
            Err(ImageError::EmptyImage { .. })
        ));
        let v = Volume::from_fn(4, 4, 4, |_, _, _| 1.0);
        assert!(matches!(
            radon_transform_3d(&v, &[]),
            Err(ImageError::EmptyAngles)
        ));
    }

    /// Unfiltered (naive) 3-D back-projection, used only to show that ramp
    /// filtering improves the reconstruction.
    fn naive_back_projection_3d(
        sinograms: &[DMatrix<f64>],
        angles: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Volume {
        let nb = nx.max(ny);
        let c = (nb - 1) as f64 / 2.0;
        let cx = (nx - 1) as f64 / 2.0;
        let cy = (ny - 1) as f64 / 2.0;
        let mut rec = vec![0.0_f64; nx * ny * nz];
        for (a, &theta) in angles.iter().enumerate() {
            let cos = theta.cos();
            let sin = theta.sin();
            for iz in 0..nz {
                let row: Vec<f64> = (0..sinograms[a].ncols())
                    .map(|j| sinograms[a][(iz, j)])
                    .collect();
                for iy in 0..ny {
                    let y = iy as f64 - cy;
                    for ix in 0..nx {
                        let x = ix as f64 - cx;
                        let t = -sin * x + cos * y;
                        let bin = t + c;
                        let val = sample_1d(&row, bin);
                        rec[ix + iy * nx + iz * nx * ny] += val;
                    }
                }
            }
        }
        let scale = 4.0 / nb as f64 / angles.len() as f64;
        for v in &mut rec {
            *v *= scale;
        }
        Volume {
            nx,
            ny,
            nz,
            data: rec,
        }
    }
}
