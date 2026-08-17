//! Discretised 1-D Laplacian on a uniform grid, checked against the analytic
//! second derivative of sin(2πx).
//!
//! Run with: `cargo run --example diffusion_operator -p tpt-sci-grid`

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_grid::{laplacian_1d, Boundary, UniformGrid1D};

fn main() {
    let g = UniformGrid1D::new(101, 0.0, 1.0).unwrap();
    let lap = laplacian_1d(&g, Boundary::Dirichlet);

    let coords = g.coordinates();
    let f: DVector<f64> = DVector::from_fn(g.n(), |i| {
        (2.0 * std::f64::consts::PI * coords[i]).sin()
    });

    // (L f)_i ≈ d²/dx² sin(2πx) = -(2π)² sin(2πx)
    let lf = lap.clone() * f;

    let mid = g.n() / 2;
    let x_mid = coords[mid];
    let analytic = -(2.0 * std::f64::consts::PI).powi(2) * (2.0 * std::f64::consts::PI * x_mid).sin();

    println!("At x = {x_mid:.3}:");
    println!("  discrete  L·f = {:.4}", lf[mid]);
    println!("  analytic  f''  = {:.4}", analytic);
    assert!((lf[mid] - analytic).abs() < 0.2, "finite-difference error too large");

    // A uniform field should have (almost) zero Laplacian under Neumann
    // (zero-flux) BC; with Dirichlet the boundary rows are identity, so L·1 = 1
    // at the ends. Use a Neumann Laplacian here to demonstrate the ~0 result.
    let lap_neu = laplacian_1d(&g, Boundary::Neumann);
    let ones = DVector::from_fn(g.n(), |_| 1.0);
    let l_ones = lap_neu * ones;
    let max_lap: f64 = l_ones.iter().copied().map(f64::abs).fold(0.0, f64::max);
    println!("max |L·1| (should be ~0) = {max_lap:.2e}");
    assert!(max_lap < 1e-9);
}
