//! Showcase of the three v1 capabilities added alongside the explicit
//! structured solver: an implicit SIMPLE pressure-correction, a `k`-`ω` SST
//! turbulence closure, and an unstructured triangular finite-volume Poisson
//! solve.

use tpt_sci_cfd_core::{CollocatedGrid, KOmegaSst, SimpleSolver, UnstructuredMesh};

fn main() {
    // 1. SIMPLE pressure correction on a structured grid.
    let grid = CollocatedGrid::new(24, 24, 1.0, 1.0).unwrap();
    let mut simple = SimpleSolver::new(grid, 1e-2, 1.0, 1e-3);
    simple.u.fill(0.5);
    for _ in 0..5 {
        assert!(simple.advance());
    }
    println!(
        "SIMPLE: max velocity divergence after 5 steps = {:.3e}",
        simple.max_divergence()
    );

    // 2. k-omega SST: decaying turbulence stays finite and positive.
    let tgrid = CollocatedGrid::new(12, 12, 1.0, 1.0).unwrap();
    let mut sst = KOmegaSst::new(tgrid, 1e-3);
    for _ in 0..30 {
        sst.step(1e-3);
    }
    let k_finite = sst.k.iter().all(|&x| x.is_finite() && x >= 1e-12);
    println!("SST: k finite and positive after decay = {k_finite}");

    // 3. Unstructured triangular Poisson solve on the unit square.
    let mesh = UnstructuredMesh::from_unit_square(16, 16).unwrap();
    let ncell = mesh.n_cells();
    let mut dirichlet = vec![None; ncell];
    let mut source = vec![0.0; ncell];
    for (c, ((src, &is_b), d)) in source
        .iter_mut()
        .zip(mesh.is_boundary_cell.iter())
        .zip(dirichlet.iter_mut())
        .enumerate()
    {
        let [x, y] = mesh.cell_center(c);
        let analytic = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
        *src = 2.0 * std::f64::consts::PI * std::f64::consts::PI * analytic;
        if is_b {
            *d = Some(analytic);
        }
    }
    let phi = mesh.solve_poisson(1.0, &source, &dirichlet);
    let mut max_err = 0.0_f64;
    for (c, (&is_b, &phi_val)) in mesh
        .is_boundary_cell
        .iter()
        .zip(phi.iter())
        .enumerate()
    {
        if is_b {
            continue;
        }
        let [x, y] = mesh.cell_center(c);
        let analytic = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
        max_err = max_err.max((phi_val - analytic).abs());
    }
    println!("Unstructured Poisson: max error vs analytic = {max_err:.3e}");
}
