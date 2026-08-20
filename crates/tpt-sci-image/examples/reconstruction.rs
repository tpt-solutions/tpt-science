//! # A tour of the `tpt-sci-image` surface
//!
//! Parallel-beam computed tomography (CT) recovers a cross-section from its
//! line integrals. We scan an object from many angles `theta` in `[0, π)`;
//! at each angle the detector records the sum of the object along parallel
//! rays — this is the **Radon transform** and its output is a **sinogram**.
//! Inverting that measurement reconstructs the original image.
//!
//! This example exercises a broad slice of the public API:
//!
//! * 2-D forward transform: [`radon_transform`].
//! * 2-D inverse transforms: [`filtered_back_projection`] (ram-lak) and
//!   [`naive_back_projection`] (unfiltered adjoint). The filtered result
//!   should beat the naive one: filtering removes the blur/halo of the raw
//!   back-projection, so we quantify both with RMSE against the ground truth.
//! * [`ImageError`] / `Result` handling: a deliberately malformed call is
//!   expected to fail, and we `assert!` on the specific error variant.
//! * 3-D volume CT: [`Volume`], [`radon_transform_3d`],
//!   [`filtered_back_projection_3d`].
//!
//! Watch the printed RMSE numbers: FBP is consistently tighter than naive,
//! and every reconstruction is finite.

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{
    ImageError, filtered_back_projection, linspace, naive_back_projection, radon_transform,
    volume::{Volume, filtered_back_projection_3d, radon_transform_3d},
};

/// Root-mean-square error between two equal-sized matrices.
fn rmse(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let mut s = 0.0;
    let mut n = 0usize;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let d = a[(i, j)] - b[(i, j)];
            s += d * d;
            n += 1;
        }
    }
    (s / n as f64).sqrt()
}

/// All entries finite?
fn all_finite(m: &DMatrix<f64>) -> bool {
    m.iter().all(|v| v.is_finite())
}

fn main() {
    // ---------------------------------------------------------------------
    // 2-D reconstruction of a two-disc phantom.
    // ---------------------------------------------------------------------
    let n = 64;
    let image = DMatrix::from_fn(n, n, |i, j| {
        let x = i as f64 - n as f64 / 2.0;
        let y = j as f64 - n as f64 / 2.0;
        // Two discs of radius ~6 px at two offsets.
        let d1 = (x).powi(2) + (y).powi(2);
        let d2 = (x - 12.0).powi(2) + (y - 10.0).powi(2);
        if d1 < 36.0 || d2 < 36.0 { 1.0 } else { 0.0 }
    });

    // Forward Radon transform over 90 evenly spaced angles in [0, π).
    let angles = linspace(0.0, std::f64::consts::PI, 90);
    let sinogram =
        radon_transform(&image, &angles).expect("radon_transform of a valid image must succeed");
    assert_eq!(sinogram.nrows(), angles.len());
    assert_eq!(sinogram.ncols(), n);

    // Reconstruct with both the filtered and the naive back-projection.
    let fbp =
        filtered_back_projection(&sinogram, &angles).expect("FBP of a valid sinogram must succeed");
    let naive = naive_back_projection(&sinogram, &angles)
        .expect("naive BP of a valid sinogram must succeed");

    let fbp_err = rmse(&fbp, &image);
    let naive_err = rmse(&naive, &image);

    println!("2-D ({n}x{n}) two-disc phantom reconstruction:");
    println!(
        "  sinogram            : {}x{}",
        sinogram.nrows(),
        sinogram.ncols()
    );
    println!("  FBP   reconstruction RMSE : {fbp_err:.4}");
    println!("  naive reconstruction RMSE : {naive_err:.4}");
    assert!(all_finite(&fbp), "FBP reconstruction must be finite");
    assert!(all_finite(&naive), "naive reconstruction must be finite");
    assert!(
        fbp_err < naive_err,
        "FBP ({fbp_err:.4}) must beat naive ({naive_err:.4})"
    );
    println!("  -> FBP beats naive (ram-lak removes the back-projection blur).");

    // ---------------------------------------------------------------------
    // ImageError handling: a deliberately malformed call must fail.
    // ---------------------------------------------------------------------
    let empty_image = DMatrix::<f64>::zeros(0, n);
    let err = radon_transform(&empty_image, &angles);
    assert!(
        matches!(err, Err(ImageError::EmptyImage { .. })),
        "empty image must yield ImageError::EmptyImage"
    );

    let bad_angles = linspace(0.0, std::f64::consts::PI, 4);
    let mismatch = filtered_back_projection(&sinogram, &bad_angles);
    assert!(
        matches!(mismatch, Err(ImageError::AngleCountMismatch { .. })),
        "angle/sinogram mismatch must yield AngleCountMismatch"
    );

    let no_angles = naive_back_projection(&sinogram, &[]);
    assert!(
        matches!(no_angles, Err(ImageError::EmptyAngles)),
        "empty angles must yield ImageError::EmptyAngles"
    );
    println!("ImageError handling: EmptyImage, AngleCountMismatch, EmptyAngles all matched.");

    // ---------------------------------------------------------------------
    // 3-D volume CT: a small spherical Gaussian volume rotated about z.
    // ---------------------------------------------------------------------
    let m = 16usize;
    let sigma = 2.5_f64;
    let c = m as f64 / 2.0;
    let volume = Volume::from_fn(m, m, m, |ix, iy, iz| {
        let dx = ix as f64 - c;
        let dy = iy as f64 - c;
        let dz = iz as f64 - c;
        (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
    });

    let angles_3d = linspace(0.0, std::f64::consts::PI, 24);
    let sinograms_3d = radon_transform_3d(&volume, &angles_3d)
        .expect("radon_transform_3d of a valid volume must succeed");
    assert_eq!(sinograms_3d.len(), angles_3d.len());

    let rec_3d = filtered_back_projection_3d(&sinograms_3d, &angles_3d, m, m, m)
        .expect("3-D FBP of a valid sinogram stack must succeed");
    assert_eq!(rec_3d.nx, m);
    assert_eq!(rec_3d.ny, m);
    assert_eq!(rec_3d.nz, m);
    assert!(
        rec_3d.data.iter().all(|v| v.is_finite()),
        "3-D rec must be finite"
    );

    // RMSE against the ground-truth volume over the interior (drop a 2-voxel border).
    let b = 2;
    let mut s = 0.0;
    let mut count = 0usize;
    for iz in b..m - b {
        for iy in b..m - b {
            for ix in b..m - b {
                let k = volume.index(ix, iy, iz);
                let d = rec_3d.data[k] - volume.data[k];
                s += d * d;
                count += 1;
            }
        }
    }
    let fbp_3d_err = (s / count as f64).sqrt();
    println!("3-D ({m}x{m}x{m}) Gaussian volume reconstruction:");
    println!(
        "  sinograms           : {} x {}x{}",
        sinograms_3d.len(),
        m,
        m
    );
    println!("  FBP reconstruction RMSE : {fbp_3d_err:.4}");

    println!("Done: 2-D FBP + naive, ImageError paths, and 3-D volume FBP all exercised.");
}
