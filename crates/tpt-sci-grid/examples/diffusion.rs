//! 1-D diffusion of a Gaussian bump on a uniform grid using the dense Laplacian.
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_grid::{Boundary, UniformGrid1D, laplacian_1d};

fn main() {
    let g = UniformGrid1D::new(101, 0.0, 1.0).unwrap();
    let l = laplacian_1d(&g, Boundary::Dirichlet);
    let xs = g.coordinates();
    let sigma = 0.05;
    let u0: Vec<f64> = xs
        .iter()
        .map(|&x| (-((x - 0.5).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();

    let coeff = 0.01;
    let dt = 0.0005;
    let mut u = u0.clone();
    for _ in 0..2000 {
        let lu: Vec<f64> = (l.clone() * DVector::from_vec(u.clone())).iter().cloned().collect();
        for k in 0..u.len() {
            u[k] += dt * coeff * lu[k];
        }
    }
    let peak0 = u0.iter().cloned().fold(0.0_f64, f64::max);
    let peak1 = u.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "Diffusion: bump peak {peak0:.4} -> {peak1:.4} over 2000 steps (should decrease)"
    );
}
