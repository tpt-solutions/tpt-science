//! Tutorial: reconstruct a CT phantom from its projections.
//!
//! A CT scanner records line integrals (a sinogram) of an object at many
//! angles, and reconstruction inverts that. Naive back-projection smears each
//! projection back across the image, which blurs it. Filtered back-projection
//! (FBP) first applies a ramp filter, which removes the blur.
//!
//! Run with: `cargo run --release --example tutorial_ct_phantom -p tpt-sci --features full`

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci::image::{filtered_back_projection, linspace, naive_back_projection, radon_transform};

const N: usize = 64;
const ANGLES: usize = 90;

/// Two overlapping discs, a stand-in for a simple anatomical phantom.
fn phantom() -> DMatrix<f64> {
    DMatrix::from_fn(N, N, |i, j| {
        let (x, y) = (i as f64 - N as f64 / 2.0, j as f64 - N as f64 / 2.0);
        let inside_a = x * x + y * y < 36.0;
        let inside_b = (x - 12.0).powi(2) + (y - 10.0).powi(2) < 36.0;
        if inside_a || inside_b { 1.0 } else { 0.0 }
    })
}

fn rmse(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let n = (a.nrows() * a.ncols()) as f64;
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum / n).sqrt()
}

fn main() {
    // Step 1 — forward model: the scanner's measurements.
    let truth = phantom();
    let angles = linspace(0.0, std::f64::consts::PI, ANGLES);
    let sinogram = radon_transform(&truth, &angles).expect("valid phantom");
    println!(
        "step 1: sinogram has {} angles x {} detector bins",
        sinogram.nrows(),
        sinogram.ncols()
    );

    // Step 2 — inversion, two ways.
    let naive = naive_back_projection(&sinogram, &angles).expect("valid sinogram");
    let fbp = filtered_back_projection(&sinogram, &angles).expect("valid sinogram");

    // Step 3 — compare. The ramp filter should win clearly.
    let (e_naive, e_fbp) = (rmse(&naive, &truth), rmse(&fbp, &truth));
    println!("step 2: naive back-projection RMSE = {e_naive:.4}");
    println!("step 3: filtered back-projection RMSE = {e_fbp:.4}");
    assert!(e_fbp < e_naive, "FBP should beat naive back-projection");
}
