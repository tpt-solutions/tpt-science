//! Parallel-beam CT reconstruction of a phantom with `tpt-sci-image`.
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{filtered_back_projection, linspace, radon_transform};

fn main() {
    let n = 64;
    let image = DMatrix::from_fn(n, n, |i, j| {
        let cx = (i as f64 - n as f64 / 2.0).powi(2) + (j as f64 - n as f64 / 2.0).powi(2);
        if cx < 100.0 { 1.0 } else { 0.0 }
    });
    let angles = linspace(0.0, std::f64::consts::PI, 90);
    let sino = radon_transform(&image, &angles).unwrap();
    let rec = filtered_back_projection(&sino, &angles).unwrap();

    let c = n / 2;
    let centre = rec[(c, c)];
    let mean = rec.iter().sum::<f64>() / (n * n) as f64;
    println!("Reconstruction: centre = {centre:.3}, mean = {mean:.3} (centre should dominate)");
}
