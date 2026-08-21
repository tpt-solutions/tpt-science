//! # Sinogram anatomy and FBP vs naive on a Shepp–Logan phantom
//!
//! This example complements `reconstruction.rs`. Instead of mainly comparing
//! reconstructions, it focuses on the **forward** measurement:
//!
//! * We build a small [Shepp–Logan phantom](https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom)
//!   — a few overlapping ellipses — as the ground-truth cross-section.
//! * We compute its **Radon sinogram** with [`radon_transform`] and *visualize*
//!   it as a downsampled ASCII raster so the streak structure of parallel-beam
//!   projections is visible in the terminal.
//! * We then invert the sinogram with both [`filtered_back_projection`] (ram-lak)
//!   and [`naive_back_projection`] (unfiltered adjoint) and quantify how much
//!   the ramp filter helps by comparing the **RMS error** of each against the
//!   phantom over the interior (FBP should win).
//! * We exercise the [`ImageError`] / `Result` API on a deliberately malformed
//!   input and `assert!` on the exact error variant — no silent panics.
//!
//! The takeaway printed at the end: filtering removes the star/halo blur of the
//! raw back-projection, and the sinogram view shows why the angular sampling
//! matters.

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{
    ImageError, filtered_back_projection, linspace, naive_back_projection, radon_transform,
};

/// A Shepp–Logan-style phantom: list of `(centre_x, centre_y, semi_x, semi_y,
/// rotation_rad, intensity)` ellipses in a `[-1, 1]²` field.
const ELLIPSES: &[(f64, f64, f64, f64, f64, f64)] = &[
    (0.0, 0.0, 0.75, 0.90, 0.0, 1.00),      // outer skull
    (0.0, 0.0, 0.55, 0.70, 0.0, 0.60),      // inner cavity
    (-0.22, 0.12, 0.20, 0.32, -0.30, 0.85), // large lobe
    (0.30, -0.10, 0.18, 0.22, 0.20, 0.70),  // small lobe
    (0.0, 0.45, 0.12, 0.10, 0.0, 0.90),     // top feature
];

/// Sample the phantom at image grid coordinate `(i, j)` of an `n x n` field,
/// mapping pixels into `[-1, 1]²`.
fn phantom(n: usize, i: usize, j: usize) -> f64 {
    let cx = (n as f64 - 1.0) / 2.0;
    let x = (j as f64 - cx) / cx; // columns -> x in [-1, 1]
    let y = (i as f64 - cx) / cx; // rows    -> y in [-1, 1]
    let mut v = 0.0;
    for &(ex, ey, sx, sy, rot, amp) in ELLIPSES {
        let c = rot.cos();
        let s = rot.sin();
        let dx = x - ex;
        let dy = y - ey;
        // Rotate the sample point into the ellipse frame.
        let rx = c * dx + s * dy;
        let ry = -s * dx + c * dy;
        if (rx * rx) / (sx * sx) + (ry * ry) / (sy * sy) <= 1.0 {
            v = amp; // later (higher-amplitude) ellipses overpaint.
        }
    }
    v
}

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

/// Print a downsampled ASCII view of the sinogram (rows = angles, cols =
/// detector bins) so its streak structure is visible in the terminal.
fn print_sinogram_ascii(sino: &DMatrix<f64>, max_rows: usize, max_cols: usize) {
    // Normalise to [0, 1] over the whole sinogram for a stable ramp.
    let mut max = 0.0_f64;
    for &v in sino.iter() {
        if v > max {
            max = v;
        }
    }
    let norm = if max > 0.0 { 1.0 / max } else { 0.0 };
    let ramp = " .:-=+*#%@";

    let step_a = (sino.nrows() + max_rows - 1) / max_rows.max(1);
    let step_j = (sino.ncols() + max_cols - 1) / max_cols.max(1);
    for a in (0..sino.nrows()).step_by(step_a.max(1)) {
        let mut line = String::new();
        for j in (0..sino.ncols()).step_by(step_j.max(1)) {
            let t = (sino[(a, j)] * norm).clamp(0.0, 1.0);
            let idx = (t * (ramp.len() - 1) as f64).round() as usize;
            line.push(ramp.chars().nth(idx).unwrap());
        }
        println!("  |{line}|");
    }
}

fn main() {
    // ---------------------------------------------------------------------
    // Build the Shepp–Logan phantom ground truth.
    // ---------------------------------------------------------------------
    let n = 96usize;
    let image = DMatrix::from_fn(n, n, |i, j| phantom(n, i, j));
    let total_mass: f64 = image.iter().sum();
    assert!(total_mass > 0.0, "phantom must have non-zero support");

    // ---------------------------------------------------------------------
    // Forward transform: the sinogram. Use a dense angular sampling so FBP is
    // well-conditioned.
    // ---------------------------------------------------------------------
    let angles = linspace(0.0, std::f64::consts::PI, 120);
    let sinogram =
        radon_transform(&image, &angles).expect("radon_transform of a valid image must succeed");
    assert_eq!(sinogram.nrows(), angles.len());
    assert_eq!(sinogram.ncols(), n);

    println!("Shepp–Logan phantom ({n}x{n}) forward projection:");
    println!(
        "  sinogram            : {} angles x {} detector bins",
        sinogram.nrows(),
        sinogram.ncols()
    );
    println!("  total projected mass : {:.1} (rows of sinogram)", {
        let mut s = 0.0;
        for j in 0..sinogram.ncols() {
            s += sinogram[(0, j)];
        }
        s
    });

    // Visualize the sinogram streaks (downsampled) in the terminal.
    println!("  sinogram (ASCII, downsampled to ~40x40):");
    print_sinogram_ascii(&sinogram, 40, 40);

    // ---------------------------------------------------------------------
    // Invert with FBP and naive back-projection; quantify the improvement.
    // ---------------------------------------------------------------------
    let fbp =
        filtered_back_projection(&sinogram, &angles).expect("FBP of a valid sinogram must succeed");
    let naive = naive_back_projection(&sinogram, &angles)
        .expect("naive BP of a valid sinogram must succeed");
    assert!(fbp.iter().all(|v| v.is_finite()), "FBP must be finite");
    assert!(naive.iter().all(|v| v.is_finite()), "naive must be finite");

    // RMS over the interior, dropping an 8-px border (edge/reconstruction
    // artefacts dominate there for both methods).
    let lo = 8;
    let hi = n - 8;
    let truth = DMatrix::from_fn(hi - lo, hi - lo, |i, j| image[(i + lo, j + lo)]);
    let crop = |m: &DMatrix<f64>| DMatrix::from_fn(hi - lo, hi - lo, |i, j| m[(i + lo, j + lo)]);
    let fbp_err = rmse(&crop(&fbp), &truth);
    let naive_err = rmse(&crop(&naive), &truth);

    println!("  reconstruction RMSE (interior {}+{}) :", lo, hi);
    println!("    FBP   : {fbp_err:.4}");
    println!("    naive : {naive_err:.4}");
    let improvement = naive_err / fbp_err.max(f64::MIN_POSITIVE);
    println!("    -> FBP reduces error by {improvement:.1}x vs naive (ramp filter removes blur).");
    assert!(
        fbp_err < naive_err,
        "FBP ({fbp_err:.4}) must beat naive ({naive_err:.4})"
    );

    // ---------------------------------------------------------------------
    // ImageError handling: a deliberately malformed call must fail cleanly.
    // ---------------------------------------------------------------------
    let empty_image = DMatrix::<f64>::zeros(0, n);
    let err = radon_transform(&empty_image, &angles);
    assert!(
        matches!(err, Err(ImageError::EmptyImage { .. })),
        "empty image must yield ImageError::EmptyImage"
    );

    let too_few = linspace(0.0, std::f64::consts::PI, 4);
    let mismatch = filtered_back_projection(&sinogram, &too_few);
    assert!(
        matches!(mismatch, Err(ImageError::AngleCountMismatch { .. })),
        "angle/sinogram mismatch must yield AngleCountMismatch"
    );

    println!("ImageError handling: EmptyImage and AngleCountMismatch matched; no silent panics.");
    println!("Done: sinogram visualised, FBP vs naive quantified, error API exercised.");
}
