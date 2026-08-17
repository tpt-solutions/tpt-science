//! Parallel-beam CT reconstruction of a simple phantom (a disc plus an
//! off-centre point source) via the Radon transform and filtered back-projection.
//!
//! Run with: `cargo run --example ct_reconstruction -p tpt-sci-image`

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{filtered_back_projection, linspace, radon_transform};

fn main() {
    let n = 64usize;
    let (cx, cy) = (n as f64 / 2.0, n as f64 / 2.0);

    // Phantom: a filled disc plus an off-centre point source.
    let image = DMatrix::from_fn(n, n, |i, j| {
        let di = i as f64 - cx;
        let dj = j as f64 - cy;
        let disc = if (di * di + dj * dj).sqrt() < 12.0 {
            1.0
        } else {
            0.0
        };
        let point = if i == 20 && j == 20 { 1.0 } else { 0.0 };
        disc + point
    });

    let angles = linspace(0.0, std::f64::consts::PI, 90);
    let sinogram = radon_transform(&image, &angles).unwrap();
    let recon = filtered_back_projection(&sinogram, &angles).unwrap();

    let mut total_orig = 0.0;
    let mut total_rec = 0.0;
    for i in 0..n {
        for j in 0..n {
            total_orig += image[(i, j)];
            total_rec += recon[(i, j)];
        }
    }

    println!("phantom total mass      = {total_orig:.2}");
    println!("reconstruction total    = {total_rec:.2}");
    println!(
        "centre pixel: original = {:.3}, reconstruction = {:.3}",
        image[(n / 2, n / 2)],
        recon[(n / 2, n / 2)]
    );
}
