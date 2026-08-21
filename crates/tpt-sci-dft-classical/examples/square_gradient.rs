//! From-scratch square-gradient classical DFT demo.
//!
//! Solves the 1-D planar wall profile (van der Waals square gradient) and a 3-D
//! density field around an impenetrable spherical core, using the same bulk
//! functional, and prints a few sanity quantities.

use tpt_sci_dft_classical::{PlanarSolve, SquareGradientDft, VdWParams, VolumetricSolve};
use tpt_sci_grid::{Boundary, UniformGrid1D, UniformGrid3D};

fn main() {
    let params = VdWParams::new(0.5, 1.0, 0.5);
    let dft = SquareGradientDft::new(params);
    let n_b = 0.1;
    let mu = dft.params.chemical_potential(n_b);

    let bulk = dft.bulk_density(mu).unwrap();
    println!("target bulk density n_b = {n_b}, analytic bulk = {bulk:.6}");

    let grid1d = UniformGrid1D::new(81, 0.0, 4.0).unwrap();
    let mut initial = vec![n_b; grid1d.n()];
    initial[0] = 0.0;
    let cfg1d = PlanarSolve {
        mu,
        initial,
        boundary: Boundary::Dirichlet,
        external_potential: None,
        tol: 1e-4,
        max_iter: 20000,
    };
    let sol1d = dft.solve_1d(&grid1d, &cfg1d).unwrap();
    let gamma = dft.surface_tension_1d(&grid1d, &sol1d.profile);
    println!(
        "1-D hard wall: wall density = {:.4}, far-field density = {:.4}, surface tension gamma = {:.5}, iters = {}",
        sol1d.profile[1],
        sol1d.profile[sol1d.profile.len() - 2],
        gamma,
        sol1d.stats.iterations,
    );

    let grid3d = UniformGrid3D::new(20, 0.0, 2.0, 20, 0.0, 2.0, 20, 0.0, 2.0).unwrap();
    let cx = grid3d.nx() / 2;
    let cy = grid3d.ny() / 2;
    let cz = grid3d.nz() / 2;
    let xs = grid3d.x_coordinates();
    let ys = grid3d.y_coordinates();
    let zs = grid3d.z_coordinates();
    let idx =
        |ix: usize, iy: usize, iz: usize| ix + iy * grid3d.nx() + iz * grid3d.nx() * grid3d.ny();
    let r_core = 0.4;
    let mut initial3d = vec![n_b; grid3d.len()];
    let mut fixed3d = vec![None; grid3d.len()];
    for iz in 0..grid3d.nz() {
        for iy in 0..grid3d.ny() {
            for ix in 0..grid3d.nx() {
                let dx = xs[ix] - xs[cx];
                let dy = ys[iy] - ys[cy];
                let dz = zs[iz] - zs[cz];
                if (dx * dx + dy * dy + dz * dz).sqrt() < r_core {
                    fixed3d[idx(ix, iy, iz)] = Some(0.0);
                    initial3d[idx(ix, iy, iz)] = 0.0;
                }
            }
        }
    }
    let cfg3d = VolumetricSolve {
        mu,
        initial: initial3d,
        boundary: Boundary::Dirichlet,
        external_potential: None,
        fixed: fixed3d,
        tol: 2e-3,
        max_iter: 15000,
    };
    let sol3d = dft.solve_3d(&grid3d, &cfg3d).unwrap();
    let far = sol3d.field[idx(cx, cy, grid3d.nz() - 2)];
    let excess = dft.excess_free_energy_3d(&grid3d, &sol3d.field);
    println!(
        "3-D core: far-field density = {far:.4} (1-D bulk = {bulk:.4}), excess free energy = {excess:.5}, iters = {}",
        sol3d.stats.iterations,
    );
}
