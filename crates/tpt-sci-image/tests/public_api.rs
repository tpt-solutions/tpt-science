//! Public-API invariants for parallel-beam CT.

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_sci_image::{linspace, radon_transform};

#[test]
fn every_projection_angle_sees_the_same_total_mass() {
    // A centred square: each parallel projection integrates the same mass.
    let n = 32;
    let image = DMatrix::from_fn(n, n, |i, j| {
        if (10..22).contains(&i) && (10..22).contains(&j) {
            1.0
        } else {
            0.0
        }
    });
    let angles = linspace(0.0, std::f64::consts::PI, 16);
    let sino = radon_transform(&image, &angles).unwrap();
    let row_sums: Vec<f64> = (0..sino.nrows())
        .map(|a| (0..sino.ncols()).map(|j| sino[(a, j)]).sum())
        .collect();
    let mean = row_sums.iter().sum::<f64>() / row_sums.len() as f64;
    for (a, s) in row_sums.iter().enumerate() {
        assert!(
            (s - mean).abs() < 0.02 * mean,
            "angle {a}: mass {s} vs mean {mean}"
        );
    }
}
