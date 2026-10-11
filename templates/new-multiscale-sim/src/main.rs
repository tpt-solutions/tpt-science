//! A reaction-diffusion model on a 1-D grid: logistic growth in every cell,
//! coupled to its neighbours by diffusion. The grid is discretised with
//! `tpt-sci-grid` and the whole system is advanced by `tpt-sci-ode` (the method
//! of lines). Replace the reaction term with your own.

use tpt_sci::grid::sparse::laplacian_1d_sparse;
use tpt_sci::grid::{Boundary, UniformGrid1D};
use tpt_sci::ode::{Method, OdeProblemBuilder};

const CELLS: usize = 101;
const DIFFUSION: f64 = 0.01;
const GROWTH: f64 = 1.0;
const CARRYING_CAPACITY: f64 = 1.0;
const T_END: f64 = 5.0;

fn main() {
    let grid = UniformGrid1D::new(CELLS, 0.0, 1.0).expect("valid grid");
    let laplacian = laplacian_1d_sparse(&grid, Boundary::Neumann);

    // Initial condition: a bump of population in the middle of the domain.
    let coords = grid.coordinates();
    let y0: Vec<f64> = coords
        .iter()
        .map(|x| (-((x - 0.5) / 0.05).powi(2)).exp() * 0.5)
        .collect();

    // Method of lines: dy/dt = D·∇²y + r·y·(1 − y/K) for every cell at once.
    let rhs = move |_t: f64, y: &[f64], dy: &mut [f64]| {
        let diffusion = laplacian.mul_vec(y);
        for i in 0..y.len() {
            dy[i] = DIFFUSION * diffusion[i]
                + GROWTH * y[i] * (1.0 - y[i] / CARRYING_CAPACITY);
        }
    };
    let problem = OdeProblemBuilder::new(rhs, y0.clone(), 0.0)
        .rtol(1e-6)
        .atol(1e-9)
        .build()
        .expect("valid problem");

    let y_end = problem
        .solve(Method::Esdirk34, T_END)
        .expect("integration succeeds");

    let total_before: f64 = y0.iter().sum::<f64>() / (CELLS - 1) as f64;
    let total_after: f64 = y_end.iter().sum::<f64>() / (CELLS - 1) as f64;
    let peak = y_end.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("population before: {total_before:.4}");
    println!("population after t = {T_END}: {total_after:.4}");
    println!("peak density: {peak:.4} (carrying capacity {CARRYING_CAPACITY})");
    assert!(peak <= CARRYING_CAPACITY + 1e-3, "growth must saturate at the capacity");
}
