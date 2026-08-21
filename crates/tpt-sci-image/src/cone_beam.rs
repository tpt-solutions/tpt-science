//! General cone-beam (divergent point-source) acquisition geometry and the
//! Feldkamp-Davis-Kress (FDK) reconstruction.
//!
//! This is a separate code path from the parallel-beam [`crate::volume`]
//! module, not a replacement for it: a point X-ray source at distance
//! [`ConeBeamGeometry::source_to_iso`] from the volume's isocenter
//! illuminates a flat 2-D detector [`ConeBeamGeometry::iso_to_detector`]
//! beyond the isocenter, and the whole source/detector rig rotates around
//! the volume (about the `z` axis) through a set of projection angles. Rays
//! diverge from the source, so (unlike the parallel-beam path) a ray's `z`
//! coordinate changes as it crosses the volume — reconstruction cannot be
//! decomposed into independent `z` slices.
//!
//! * [`cone_beam_forward_projection`] ray-marches, with trilinear
//!   interpolation, from the source through every detector pixel to produce
//!   one 2-D projection image per angle.
//! * [`fdk_reconstruction`] inverts that model: it cosine-weights each
//!   projection for ray divergence, ram-lak filters it along the detector
//!   row direction (reusing [`crate::ramp_filter`], the same machinery
//!   behind [`crate::filtered_back_projection`] and
//!   [`crate::volume::filtered_back_projection_3d`]), and back-projects
//!   along the divergent rays with an inverse-square distance weight.
//!
//! Reference: L. A. Feldkamp, L. C. Davis, J. W. Kress, "Practical cone-beam
//! algorithm," *J. Opt. Soc. Am. A* 1(6), 612-619 (1984).

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

use crate::error::ImageError;
use crate::volume::{Volume, sample_trilinear};
use crate::{ramp_filter, sample_bilinear};

/// Radial step (in voxel units) used to ray-march the forward projector.
const RAY_STEP: f64 = 0.5;

/// Cone-beam acquisition geometry.
///
/// The source/detector rig rotates about the volume's `z` axis. At angle
/// `theta` the source sits at distance [`Self::source_to_iso`] from the
/// origin along `-( cos theta, sin theta, 0 )`, and the detector plane sits
/// at distance [`Self::iso_to_detector`] beyond the origin along
/// `( cos theta, sin theta, 0 )`. The detector's own `u` axis (in-plane,
/// rotating with the gantry) is `( -sin theta, cos theta, 0 )` and its `v`
/// axis is the fixed `( 0, 0, 1 )`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConeBeamGeometry {
    /// Distance from the point source to the volume isocenter (SAD, "source
    /// to axis distance").
    pub source_to_iso: f64,
    /// Distance from the isocenter to the flat detector plane.
    pub iso_to_detector: f64,
    /// Detector pixel count along `u` (in-plane, rotates with the gantry).
    pub det_nu: usize,
    /// Detector pixel count along `v` (parallel to the rotation axis `z`).
    pub det_nv: usize,
    /// Detector pixel spacing along `u`.
    pub du: f64,
    /// Detector pixel spacing along `v`.
    pub dv: f64,
    /// Number of projection angles the acquisition is taken over (typically
    /// spanning a full turn `[0, 2*pi)`, unlike the parallel-beam path which
    /// only needs `[0, pi)`). This is metadata cross-checked against the
    /// `angles` slice length passed to [`cone_beam_forward_projection`] and
    /// [`fdk_reconstruction`]; those functions take the angles explicitly so
    /// callers may choose any angle set.
    pub n_angles: usize,
}

impl ConeBeamGeometry {
    /// Source-to-detector distance (SDD), `source_to_iso + iso_to_detector`.
    #[must_use]
    pub fn source_to_detector(&self) -> f64 {
        self.source_to_iso + self.iso_to_detector
    }

    /// Validate that every geometry parameter is usable.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidGeometry`] if any distance or pixel
    /// spacing is not positive, or if any pixel/angle count is zero.
    pub fn validate(&self) -> Result<(), ImageError> {
        if self.source_to_iso <= 0.0 {
            return Err(ImageError::InvalidGeometry {
                reason: format!("source_to_iso must be positive, got {}", self.source_to_iso),
            });
        }
        if self.iso_to_detector <= 0.0 {
            return Err(ImageError::InvalidGeometry {
                reason: format!(
                    "iso_to_detector must be positive, got {}",
                    self.iso_to_detector
                ),
            });
        }
        if self.du <= 0.0 || self.dv <= 0.0 {
            return Err(ImageError::InvalidGeometry {
                reason: format!(
                    "detector pixel spacing must be positive, got du={}, dv={}",
                    self.du, self.dv
                ),
            });
        }
        if self.det_nu == 0 || self.det_nv == 0 {
            return Err(ImageError::InvalidGeometry {
                reason: format!(
                    "detector pixel counts must be non-zero, got det_nu={}, det_nv={}",
                    self.det_nu, self.det_nv
                ),
            });
        }
        if self.n_angles == 0 {
            return Err(ImageError::InvalidGeometry {
                reason: "n_angles must be non-zero".to_string(),
            });
        }
        Ok(())
    }
}

/// The gantry-relative basis at projection angle `theta`: `(source, n, u_dir)`
/// where `n` is the source-to-detector unit direction and `u_dir` is the
/// detector's in-plane horizontal axis.
fn gantry_basis(geometry: &ConeBeamGeometry, theta: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let cos = theta.cos();
    let sin = theta.sin();
    let n = [cos, sin, 0.0];
    let u_dir = [-sin, cos, 0.0];
    let source = [
        -geometry.source_to_iso * cos,
        -geometry.source_to_iso * sin,
        0.0,
    ];
    (source, n, u_dir)
}

/// World-space position of detector pixel `(iu, iv)` at angle `theta`.
fn detector_pixel_position(
    geometry: &ConeBeamGeometry,
    theta: f64,
    iu: usize,
    iv: usize,
) -> [f64; 3] {
    let (_, n, u_dir) = gantry_basis(geometry, theta);
    let cu = (geometry.det_nu as f64 - 1.0) / 2.0;
    let cv = (geometry.det_nv as f64 - 1.0) / 2.0;
    let u = (iu as f64 - cu) * geometry.du;
    let v = (iv as f64 - cv) * geometry.dv;
    let center = [
        geometry.iso_to_detector * n[0],
        geometry.iso_to_detector * n[1],
        0.0,
    ];
    [
        center[0] + u * u_dir[0],
        center[1] + u * u_dir[1],
        center[2] + v,
    ]
}

/// Forward cone-beam projection of `volume` under `geometry` over `angles`.
///
/// For every angle and every detector pixel, a ray is marched (in
/// [`RAY_STEP`]-sized steps, trilinearly interpolated per [`crate::volume`])
/// from the point source through the pixel, and the samples are summed
/// (scaled by the step size) into an approximate line integral — the same
/// forward model used by [`crate::radon_transform`] and
/// [`crate::volume::radon_transform_3d`], generalized to divergent rays.
///
/// Returns one `[det_nv x det_nu]` [`DMatrix`] (`det_nv` rows/`v`, `det_nu`
/// columns/`u`) per angle.
///
/// # Errors
///
/// Returns [`ImageError::InvalidGeometry`] if `geometry` is invalid,
/// [`ImageError::EmptyImage`] if `volume` has no voxels, or
/// [`ImageError::AngleCountMismatch`] if `angles.len() != geometry.n_angles`.
///
/// # Examples
///
/// ```
/// use tpt_sci_image::{ConeBeamGeometry, cone_beam_forward_projection, linspace, volume::Volume};
///
/// let vol = Volume::from_fn(8, 8, 8, |ix, iy, iz| {
///     if ix == 4 && iy == 4 && iz == 4 { 1.0 } else { 0.0 }
/// });
/// let geometry = ConeBeamGeometry {
///     source_to_iso: 30.0,
///     iso_to_detector: 20.0,
///     det_nu: 12,
///     det_nv: 12,
///     du: 1.0,
///     dv: 1.0,
///     n_angles: 8,
/// };
/// let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 8);
/// let projections = cone_beam_forward_projection(&vol, &geometry, &angles).unwrap();
/// assert_eq!(projections.len(), 8);
/// assert_eq!(projections[0].nrows(), 12);
/// assert_eq!(projections[0].ncols(), 12);
/// ```
pub fn cone_beam_forward_projection(
    volume: &Volume,
    geometry: &ConeBeamGeometry,
    angles: &[f64],
) -> Result<Vec<DMatrix<f64>>, ImageError> {
    geometry.validate()?;
    if volume.is_empty() {
        return Err(ImageError::EmptyImage {
            nrows: volume.ny,
            ncols: volume.nx,
        });
    }
    if angles.len() != geometry.n_angles {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: angles.len(),
            n_angles: geometry.n_angles,
        });
    }

    let cx = (volume.nx as f64 - 1.0) / 2.0;
    let cy = (volume.ny as f64 - 1.0) / 2.0;
    let cz = (volume.nz as f64 - 1.0) / 2.0;

    let mut out = Vec::with_capacity(angles.len());
    for &theta in angles {
        let (source, _, _) = gantry_basis(geometry, theta);
        let mut proj = vec![0.0_f64; geometry.det_nv * geometry.det_nu];
        for iv in 0..geometry.det_nv {
            for iu in 0..geometry.det_nu {
                let pixel = detector_pixel_position(geometry, theta, iu, iv);
                let dir = [
                    pixel[0] - source[0],
                    pixel[1] - source[1],
                    pixel[2] - source[2],
                ];
                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                if len <= 0.0 {
                    continue;
                }
                let step = [
                    dir[0] / len * RAY_STEP,
                    dir[1] / len * RAY_STEP,
                    dir[2] / len * RAY_STEP,
                ];
                let n_steps = (len / RAY_STEP).floor() as usize;
                let mut acc = 0.0;
                let mut pos = source;
                for _ in 0..n_steps {
                    pos[0] += step[0];
                    pos[1] += step[1];
                    pos[2] += step[2];
                    acc += sample_trilinear(volume, pos[0] + cx, pos[1] + cy, pos[2] + cz);
                }
                proj[iv * geometry.det_nu + iu] = acc * RAY_STEP;
            }
        }
        out.push(DMatrix::from_fn(
            geometry.det_nv,
            geometry.det_nu,
            |iv, iu| proj[iv * geometry.det_nu + iu],
        ));
    }
    Ok(out)
}

/// Feldkamp-Davis-Kress (FDK) cone-beam reconstruction.
///
/// Implements the approximate but widely-used FDK filtered back-projection
/// algorithm (Feldkamp, Davis & Kress, *J. Opt. Soc. Am. A* 1(6), 612-619,
/// 1984):
///
/// 1. Each detector pixel `(u, v)` of every projection is cosine-weighted by
///    `SDD / sqrt(SDD^2 + u^2 + v^2)` (`SDD` = [`ConeBeamGeometry::source_to_detector`])
///    to account for ray divergence.
/// 2. Each detector row (fixed `v`, varying `u`) is ram-lak filtered via
///    [`crate::ramp_filter`] — the same filter used by
///    [`crate::filtered_back_projection`] and
///    [`crate::volume::filtered_back_projection_3d`].
/// 3. Each voxel is back-projected along the divergent ray through it at
///    every angle, weighted by the inverse-square distance factor
///    `(SAD / U)^2` where `U = SAD + x*cos(theta) + y*sin(theta)` is the
///    signed distance from the source to the voxel's projection onto the
///    central-ray axis (`SAD` = [`ConeBeamGeometry::source_to_iso`]).
///
/// The reconstruction has dimensions `nx x ny x nz`.
///
/// # Errors
///
/// Returns [`ImageError::InvalidGeometry`] if `geometry` is invalid,
/// [`ImageError::EmptyImage`] if `nx`/`ny`/`nz` is zero,
/// [`ImageError::AngleCountMismatch`] if `angles.len() != geometry.n_angles`
/// or `projections.len() != angles.len()`, or
/// [`ImageError::DimensionMismatch`] if a projection's dimensions are not
/// `det_nv x det_nu`.
///
/// # Examples
///
/// ```
/// use tpt_sci_image::{
///     ConeBeamGeometry, cone_beam_forward_projection, fdk_reconstruction, linspace,
///     volume::Volume,
/// };
///
/// let n = 10usize;
/// let vol = Volume::from_fn(n, n, n, |ix, iy, iz| {
///     if ix == n / 2 && iy == n / 2 && iz == n / 2 { 1.0 } else { 0.0 }
/// });
/// let geometry = ConeBeamGeometry {
///     source_to_iso: 30.0,
///     iso_to_detector: 20.0,
///     det_nu: 14,
///     det_nv: 14,
///     du: 1.0,
///     dv: 1.0,
///     n_angles: 16,
/// };
/// let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 16);
/// let projections = cone_beam_forward_projection(&vol, &geometry, &angles).unwrap();
/// let rec = fdk_reconstruction(&projections, &geometry, &angles, n, n, n).unwrap();
/// assert_eq!(rec.nx, n);
/// assert_eq!(rec.ny, n);
/// assert_eq!(rec.nz, n);
/// ```
pub fn fdk_reconstruction(
    projections: &[DMatrix<f64>],
    geometry: &ConeBeamGeometry,
    angles: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Volume, ImageError> {
    geometry.validate()?;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(ImageError::EmptyImage {
            nrows: ny,
            ncols: nx,
        });
    }
    if angles.len() != geometry.n_angles {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: angles.len(),
            n_angles: geometry.n_angles,
        });
    }
    if projections.len() != angles.len() {
        return Err(ImageError::AngleCountMismatch {
            sino_rows: projections.len(),
            n_angles: angles.len(),
        });
    }
    for p in projections {
        if p.nrows() != geometry.det_nv {
            return Err(ImageError::DimensionMismatch {
                expected: geometry.det_nv,
                got: p.nrows(),
            });
        }
        if p.ncols() != geometry.det_nu {
            return Err(ImageError::DimensionMismatch {
                expected: geometry.det_nu,
                got: p.ncols(),
            });
        }
    }

    let sdd = geometry.source_to_detector();
    let sad = geometry.source_to_iso;
    let cu = (geometry.det_nu as f64 - 1.0) / 2.0;
    let cv = (geometry.det_nv as f64 - 1.0) / 2.0;

    // Step 1 (cosine weight) and step 2 (ram-lak filter along u), producing
    // one weighted+filtered `[det_nv x det_nu]` matrix per angle.
    let mut filtered: Vec<DMatrix<f64>> = Vec::with_capacity(projections.len());
    for p in projections {
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(geometry.det_nv);
        for iv in 0..geometry.det_nv {
            let v = (iv as f64 - cv) * geometry.dv;
            let weighted: Vec<f64> = (0..geometry.det_nu)
                .map(|iu| {
                    let u = (iu as f64 - cu) * geometry.du;
                    let w = sdd / (sdd * sdd + u * u + v * v).sqrt();
                    p[(iv, iu)] * w
                })
                .collect();
            rows.push(ramp_filter(&weighted));
        }
        filtered.push(DMatrix::from_fn(
            geometry.det_nv,
            geometry.det_nu,
            |iv, iu| rows[iv][iu],
        ));
    }

    // Step 3: divergent back-projection with inverse-square weighting.
    let cx = (nx as f64 - 1.0) / 2.0;
    let cy = (ny as f64 - 1.0) / 2.0;
    let cz = (nz as f64 - 1.0) / 2.0;
    let mut rec = vec![0.0_f64; nx * ny * nz];
    for (a, &theta) in angles.iter().enumerate() {
        let cos = theta.cos();
        let sin = theta.sin();
        let proj = &filtered[a];
        for iz in 0..nz {
            let z = iz as f64 - cz;
            for iy in 0..ny {
                let y = iy as f64 - cy;
                for ix in 0..nx {
                    let x = ix as f64 - cx;
                    let u_axis = sad + x * cos + y * sin;
                    if u_axis <= 1e-9 {
                        // Voxel is behind (or at) the source along this ray;
                        // not physically illuminated at this angle.
                        continue;
                    }
                    let mag = sdd / u_axis;
                    let u = mag * (-x * sin + y * cos);
                    let v = mag * z;
                    let iu_f = u / geometry.du + cu;
                    let iv_f = v / geometry.dv + cv;
                    let val = sample_bilinear(proj, iu_f, iv_f);
                    let weight = (sad / u_axis) * (sad / u_axis);
                    rec[ix + iy * nx + iz * nx * ny] += weight * val;
                }
            }
        }
    }

    // Empirical amplitude normalisation, mirroring the `4.0 / (nb *
    // n_angles)` scale used by the parallel-beam paths (Kak & Slaney,
    // *Principles of Computerized Tomographic Imaging*, §3.3): the discrete
    // ram-lak filter's DC gain is matched to the continuous kernel and the
    // detector-bin/angle-spacing factors are absorbed into one constant.
    // FDK's own back-projection weight carries an explicit `1/2` prefactor
    // for the integral over the full `[0, 2*pi)` turn (Feldkamp, Davis &
    // Kress 1984, eq. 19); since a cone-beam acquisition doubles the angular
    // range of a parallel-beam `[0, pi)` scan at the same `n_angles`, that
    // `1/2` exactly cancels the doubled angle-spacing `dtheta`, leaving the
    // same `4.0 / (det_nu * n_angles)` constant.
    let scale = 4.0 / geometry.det_nu as f64 / angles.len() as f64;
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
    use crate::linspace;

    fn default_geometry(n_angles: usize) -> ConeBeamGeometry {
        ConeBeamGeometry {
            source_to_iso: 30.0,
            iso_to_detector: 20.0,
            det_nu: 16,
            det_nv: 16,
            du: 1.0,
            dv: 1.0,
            n_angles,
        }
    }

    #[test]
    fn geometry_validates_positive_parameters() {
        let g = default_geometry(8);
        assert!(g.validate().is_ok());

        let mut bad = g;
        bad.source_to_iso = 0.0;
        assert!(matches!(
            bad.validate(),
            Err(ImageError::InvalidGeometry { .. })
        ));

        let mut bad = g;
        bad.det_nu = 0;
        assert!(matches!(
            bad.validate(),
            Err(ImageError::InvalidGeometry { .. })
        ));

        let mut bad = g;
        bad.n_angles = 0;
        assert!(matches!(
            bad.validate(),
            Err(ImageError::InvalidGeometry { .. })
        ));
    }

    #[test]
    fn forward_projection_of_centred_voxel_peaks_at_detector_centre() {
        let n = 12usize;
        let vol = Volume::from_fn(n, n, n, |ix, iy, iz| {
            if ix == n / 2 && iy == n / 2 && iz == n / 2 {
                1.0
            } else {
                0.0
            }
        });
        let geometry = default_geometry(10);
        let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 10);
        let projections = cone_beam_forward_projection(&vol, &geometry, &angles).unwrap();

        assert_eq!(projections.len(), 10);
        let cu = (geometry.det_nu - 1) / 2;
        let cv = (geometry.det_nv - 1) / 2;
        for (a, proj) in projections.iter().enumerate() {
            assert_eq!(proj.nrows(), geometry.det_nv);
            assert_eq!(proj.ncols(), geometry.det_nu);

            let mut max_val = f64::NEG_INFINITY;
            let mut max_pos = (0usize, 0usize);
            for iv in 0..proj.nrows() {
                for iu in 0..proj.ncols() {
                    if proj[(iv, iu)] > max_val {
                        max_val = proj[(iv, iu)];
                        max_pos = (iv, iu);
                    }
                }
            }
            // A voxel exactly at the isocenter projects to the detector
            // centre at every angle (u = v = 0 regardless of theta).
            assert!(
                max_val > 0.0,
                "angle {a}: projection should be non-degenerate"
            );
            assert!(
                (max_pos.0 as isize - cv as isize).abs() <= 2,
                "angle {a}: v peak {} should be near centre {cv}",
                max_pos.0
            );
            assert!(
                (max_pos.1 as isize - cu as isize).abs() <= 2,
                "angle {a}: u peak {} should be near centre {cu}",
                max_pos.1
            );
        }
    }

    #[test]
    fn forward_projection_rejects_bad_inputs() {
        let vol = Volume::from_fn(4, 4, 4, |_, _, _| 1.0);
        let geometry = default_geometry(4);
        let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 3);
        assert!(matches!(
            cone_beam_forward_projection(&vol, &geometry, &angles),
            Err(ImageError::AngleCountMismatch { .. })
        ));

        let empty = Volume::new(0, 4, 4);
        let angles4 = linspace(0.0, 2.0 * std::f64::consts::PI, 4);
        assert!(matches!(
            cone_beam_forward_projection(&empty, &geometry, &angles4),
            Err(ImageError::EmptyImage { .. })
        ));

        let mut bad_geometry = geometry;
        bad_geometry.source_to_iso = -1.0;
        assert!(matches!(
            cone_beam_forward_projection(&vol, &bad_geometry, &angles4),
            Err(ImageError::InvalidGeometry { .. })
        ));
    }

    #[test]
    fn fdk_recovers_point_source() {
        let n = 12usize;
        let vol = Volume::from_fn(n, n, n, |ix, iy, iz| {
            if ix == n / 2 && iy == n / 2 && iz == n / 2 {
                1.0
            } else {
                0.0
            }
        });
        let geometry = default_geometry(20);
        let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 20);
        let projections = cone_beam_forward_projection(&vol, &geometry, &angles).unwrap();
        let rec = fdk_reconstruction(&projections, &geometry, &angles, n, n, n).unwrap();

        let c = rec.data[rec.index(n / 2, n / 2, n / 2)];
        let mean = rec.data.iter().sum::<f64>() / rec.data.len() as f64;
        assert!(c > 0.0, "centre should be positive, got {c}");
        assert!(
            c >= mean,
            "centre should be at least the mean (c={c}, mean={mean})"
        );
    }

    #[test]
    fn fdk_round_trip_recovers_uniform_cube_phantom() {
        let n = 14usize;
        let half = 2isize;
        let c = n as isize / 2;
        let phantom = Volume::from_fn(n, n, n, |ix, iy, iz| {
            let dx = ix as isize - c;
            let dy = iy as isize - c;
            let dz = iz as isize - c;
            if dx.abs() <= half && dy.abs() <= half && dz.abs() <= half {
                1.0
            } else {
                0.0
            }
        });
        let geometry = default_geometry(28);
        let angles = linspace(0.0, 2.0 * std::f64::consts::PI, 28);
        let projections = cone_beam_forward_projection(&phantom, &geometry, &angles).unwrap();
        let rec = fdk_reconstruction(&projections, &geometry, &angles, n, n, n).unwrap();

        // The reconstructed peak should land at (or immediately next to) the
        // phantom's true centre, and be substantially brighter than the
        // background far from the cube.
        let mut max_val = f64::NEG_INFINITY;
        let mut max_pos = (0usize, 0usize, 0usize);
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let v = rec.data[rec.index(ix, iy, iz)];
                    if v > max_val {
                        max_val = v;
                        max_pos = (ix, iy, iz);
                    }
                }
            }
        }
        let cu = n / 2;
        assert!(
            (max_pos.0 as isize - cu as isize).abs() <= 2
                && (max_pos.1 as isize - cu as isize).abs() <= 2
                && (max_pos.2 as isize - cu as isize).abs() <= 2,
            "reconstruction peak {max_pos:?} should be near the phantom centre ({cu}, {cu}, {cu})"
        );

        let corner = rec.data[rec.index(0, 0, 0)];
        assert!(
            max_val > 2.0 * corner.abs() + 1e-6,
            "peak {max_val} should stand out well above the background corner {corner}"
        );
    }
}
