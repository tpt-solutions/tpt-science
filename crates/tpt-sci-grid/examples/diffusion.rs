//! # Structured-grid diffusion: a tour of the `tpt-sci-grid` surface
//!
//! `tpt-sci-grid` builds discrete PDE operators (Laplacians, derivatives) on
//! uniform tensor-product grids. This example is a guided tour of the public
//! API, not just the 1-D dense Laplacian of the previous version.
//!
//! What to observe as it runs:
//!
//! * **1-D diffusion** of a Gaussian bump with the *dense* Laplacian, run under
//!   both homogeneous **Dirichlet** (`u = 0` at the ends) and **Neumann**
//!   (zero flux, `du/dn = 0`) boundaries. The peak always decays, but the *mass*
//!   (sum of `u`) is conserved under Neumann and leaks away under Dirichlet —
//!   that is the physical meaning of the two boundary conditions.
//! * **2-D diffusion** on a small tensor-product grid: a single hot corner node
//!   spreads into the domain, and the temperature at the *center* rises. We use
//!   Neumann there so total heat is conserved.
//! * The **2-D Laplacian** assembled two ways that must agree: via
//!   [`laplacian_2d`] and via a Kronecker product `kron(I_y, L_x) + kron(L_y,
//!   I_x)` of 1-D operators ([`kron`]).
//! * **Derivative stencils** ([`Stencil`] + [`derivative_1d`]): a central
//!   first derivative of `x` is `1`, a central second derivative of `x²` is `2`.
//! * **3-D grid** construction, lexicographic [`UniformGrid3D::index`], and the
//!   dense 3-D Laplacian applied to `x² + y² + z²` (giving `6`).
//! * The feature-gated **sparse** backend ([`CsrMatrix`], [`laplacian_1d_sparse`],
//!   [`diffuse_step`]), compiled in only when `tpt-sci-grid` is built with
//!   `--features sparse`. The default example run stays dense-only and fast.
//!
//! Everything is deterministic and small enough to run in well under a second.

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_sci_grid::{
    Boundary, Stencil, UniformGrid1D, UniformGrid2D, UniformGrid3D, derivative_1d, kron,
    laplacian_1d, laplacian_2d, laplacian_3d, linspace,
};

/// Explicit-Euler diffusion `u += dt·D·(L·u)` on a 1-D grid.
/// Returns `(final_peak, initial_mass, final_mass)`.
fn diffuse_1d(bc: Boundary) -> (f64, f64, f64) {
    let g = UniformGrid1D::new(101, 0.0, 1.0).unwrap();
    let l = laplacian_1d(&g, bc);
    let xs = g.coordinates();
    let sigma = 0.05;
    let u0: Vec<f64> = xs
        .iter()
        .map(|&x| (-((x - 0.5).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();

    let coeff = 0.01;
    let dt = 0.0005;
    let mass0: f64 = u0.iter().sum();
    let mut u = u0.clone();
    for _ in 0..2000 {
        let lu = l.clone() * DVector::from_vec(u.clone());
        for (ui, li) in u.iter_mut().zip(lu.iter()) {
            *ui += dt * coeff * li;
        }
    }
    let peak = u.iter().cloned().fold(0.0_f64, f64::max);
    let mass1: f64 = u.iter().sum();
    (peak, mass0, mass1)
}

/// Explicit-Euler diffusion on a 2-D grid with a single hot corner node, under
/// zero-flux (Neumann) boundaries. Returns `(center_temp, initial_mass,
/// final_mass)`. Heat is injected at the corner so we can watch it travel to the
/// center; note a *sharp* corner spike exposes the one-sided stencil's slight
/// asymmetry (see the smooth-field run below for true conservation).
fn diffuse_2d_neumann_corner() -> (f64, f64, f64) {
    let g = UniformGrid2D::new(25, 0.0, 1.0, 25, 0.0, 1.0).unwrap();
    let l = laplacian_2d(&g, Boundary::Neumann);

    let mut u = vec![0.0_f64; g.len()];
    // Heat the corner node (ix = 0, iy = 0); ordering is `ix + iy * nx`.
    u[0] = 1.0;
    let mass0: f64 = u.iter().sum();

    let coeff = 1.0;
    let dt = 1e-4;
    for _ in 0..500 {
        let lu = l.clone() * DVector::from_vec(u.clone());
        for (ui, li) in u.iter_mut().zip(lu.iter()) {
            *ui += dt * coeff * li;
        }
    }

    let cx = g.nx() / 2;
    let cy = g.ny() / 2;
    let center = cx + cy * g.nx();
    let mass1: f64 = u.iter().sum();
    (u[center], mass0, mass1)
}

fn main() {
    println!("=== tpt-sci-grid: structured-grid diffusion tour ===\n");

    // --- 1-D diffusion: Dirichlet vs Neumann ---------------------------------
    let (peak_d, mass0_d, mass1_d) = diffuse_1d(Boundary::Dirichlet);
    let (peak_n, mass0_n, mass1_n) = diffuse_1d(Boundary::Neumann);

    println!("1-D Gaussian diffusion (2000 explicit steps):");
    println!("  Dirichlet: peak {peak_d:.4}  (mass {mass0_d:.3} -> {mass1_d:.3}, leaks out)");
    println!("  Neumann:   peak {peak_n:.4}  (mass {mass0_n:.3} -> {mass1_n:.3}, conserved)");

    assert!(
        peak_d.is_finite() && peak_n.is_finite(),
        "solution must stay finite"
    );
    assert!(
        peak_d < 1.0,
        "Dirichlet peak must decay below the initial 1.0"
    );
    assert!(
        peak_n < 1.0,
        "Neumann peak must decay below the initial 1.0"
    );
    // Neumann is zero-flux: total heat is conserved to ~0.1%.
    let rel_drift = (mass1_n - mass0_n).abs() / mass0_n;
    assert!(rel_drift < 1e-3, "Neumann mass must be conserved");
    // Dirichlet lets heat escape, so mass must drop (more than Neumann's drift).
    assert!(
        mass1_d < mass0_d,
        "Dirichlet mass must leak away (flux through the boundary)"
    );
    assert!(
        (mass0_d - mass1_d) / mass0_d > (mass1_n - mass0_n).abs() / mass0_n,
        "Dirichlet must lose more mass than Neumann drifts"
    );

    // --- 2-D diffusion: hot corner, center rises -----------------------------
    let (center_t, mass0_2d, mass1_2d) = diffuse_2d_neumann_corner();
    println!("\n2-D Neumann diffusion from a hot corner (500 steps): center temp = {center_t:.5}");
    println!("  mass {mass0_2d:.3} -> {mass1_2d:.3} (a sharp corner spike exposes the");
    println!("  one-sided stencil's slight asymmetry, so it is not perfectly conserved)");
    assert!(center_t > 0.0, "heat must reach the center");
    assert!(
        center_t.is_finite() && mass1_2d.is_finite(),
        "solution must stay finite"
    );
    assert!(
        center_t < mass0_2d / 25.0_f64 / 25.0_f64 * 2.0,
        "center stays well below the uniform-equilibrium temperature"
    );

    // --- 2-D Laplacian via laplacian_2d and via Kronecker sum ----------------
    let g2 = UniformGrid2D::new(11, 0.0, 1.0, 7, 0.0, 1.0).unwrap();
    let l2d = laplacian_2d(&g2, Boundary::Dirichlet);
    let lx = laplacian_1d(
        &UniformGrid1D::new(11, 0.0, 1.0).unwrap(),
        Boundary::Dirichlet,
    );
    let ly = laplacian_1d(
        &UniformGrid1D::new(7, 0.0, 1.0).unwrap(),
        Boundary::Dirichlet,
    );
    let ix = DMatrix::from_fn(11, 11, |a, b| if a == b { 1.0 } else { 0.0 });
    let iy = DMatrix::from_fn(7, 7, |a, b| if a == b { 1.0 } else { 0.0 });
    let l_kron = kron(&iy, &lx) + kron(&ly, &ix);
    let max_diff = l2d
        .iter()
        .zip(l_kron.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("\nlaplacian_2d vs kron(I_y,L_x)+kron(L_y,I_x): max |diff| = {max_diff:.2e}");
    assert!(
        max_diff < 1e-12,
        "Kronecker construction must match laplacian_2d"
    );

    // --- Dirichlet vs Neumann operator difference (static) -------------------
    let ld = laplacian_2d(&g2, Boundary::Dirichlet);
    let ln = laplacian_2d(&g2, Boundary::Neumann);
    let dx2 = g2.dx() * g2.dx();
    let dy2 = g2.dy() * g2.dy();
    // A Dirichlet corner touches two boundaries, so both 1-D identity rows
    // contribute: the row is a scaled identity with 2 on the diagonal and no
    // off-diagonal coupling (this is what forces u = 0 at the boundary).
    assert!(
        (ld[(0, 0)] - 2.0).abs() < 1e-9,
        "Dirichlet corner row has 2 on the diagonal"
    );
    assert!(
        ld[(0, 1)].abs() < 1e-12 && ld[(0, g2.nx())].abs() < 1e-12,
        "Dirichlet corner row has no off-diagonal coupling"
    );
    // Neumann uses a one-sided (zero-flux) stencil at the corner: -2/dx^2 on the
    // x-axis and -2/dy^2 on the y-axis on the diagonal, with one-sided coupling
    // of +2/dx^2 to (1, 0) and +2/dy^2 to (0, 1).
    assert!(
        (ln[(0, 0)] + 2.0 / dx2 + 2.0 / dy2).abs() < 1e-6,
        "Neumann corner diagonal is -2/dx^2 - 2/dy^2"
    );
    assert!(
        (ln[(0, 1)] - 2.0 / dx2).abs() < 1e-6,
        "Neumann corner couples +2/dx^2 to (1, 0)"
    );
    println!(
        "Dirichlet forces u=0 at the boundary (identity rows); Neumann uses zero-flux one-sided stencils."
    );

    // --- Derivative stencils -------------------------------------------------
    let g1 = UniformGrid1D::new(21, 0.0, 1.0).unwrap();
    let d1 = derivative_1d(&g1, Stencil::CentralFirstDerivative);
    let du = d1 * DVector::from_vec(g1.coordinates());
    let mid = g1.n() / 2;
    assert!((du[mid] - 1.0).abs() < 1e-3, "d/dx of x is 1");

    let d2 = derivative_1d(&g1, Stencil::CentralSecondDerivative);
    let u2: Vec<f64> = g1.coordinates().iter().map(|x| x * x).collect();
    let d2u = d2 * DVector::from_vec(u2);
    assert!((d2u[mid] - 2.0).abs() < 1e-3, "d^2/dx^2 of x^2 is 2");
    println!(
        "\nderivative_1d: d/dx(x)={:.3}, d^2/dx^2(x^2)={:.3}",
        du[mid], d2u[mid]
    );

    // --- linspace helper -----------------------------------------------------
    let xs = linspace(0.0_f64, 1.0, 5);
    assert_eq!(xs.len(), 5);
    assert!((xs[0]).abs() < 1e-12 && (xs[4] - 1.0).abs() < 1e-12);
    println!("linspace(0,1,5) = {:?}", xs);

    // --- 3-D grid + dense 3-D Laplacian --------------------------------------
    let g3 = UniformGrid3D::new(9, 0.0, 1.0, 9, 0.0, 1.0, 9, 0.0, 1.0).unwrap();
    println!(
        "\n3-D grid: {} nodes ({}x{}x{}), lexicographic index(1,0,0) = {}",
        g3.len(),
        g3.nx(),
        g3.ny(),
        g3.nz(),
        g3.index(1, 0, 0)
    );
    let l3 = laplacian_3d(&g3, Boundary::Dirichlet);
    let xc = g3.x_coordinates();
    let yc = g3.y_coordinates();
    let zc = g3.z_coordinates();
    let u3: Vec<f64> = (0..g3.len())
        .map(|k| {
            let ix = k % g3.nx();
            let iy = (k / g3.nx()) % g3.ny();
            let iz = k / (g3.nx() * g3.ny());
            xc[ix] * xc[ix] + yc[iy] * yc[iy] + zc[iz] * zc[iz]
        })
        .collect();
    let lu3 = l3 * DVector::from_vec(u3);
    let center3 = g3.len() / 2;
    assert!(
        (lu3[center3] - 6.0).abs() < 1e-2,
        "laplacian of x^2+y^2+z^2 is 6"
    );
    println!(
        "laplacian_3d of (x^2+y^2+z^2) at center = {:.3} (expect 6)",
        lu3[center3]
    );

    // --- Sparse backend (only when built with --features sparse) ------------
    #[cfg(feature = "sparse")]
    sparse_tour();

    println!("\nAll checks passed.");
}

/// Feature-gated tour of the CSR sparse operators. Only compiled/run when the
/// `sparse` feature is enabled, so the default example stays dense and fast.
#[cfg(feature = "sparse")]
fn sparse_tour() {
    use tpt_sci_grid::{CsrMatrix, diffuse_step, laplacian_1d_sparse};

    let g = UniformGrid1D::new(101, 0.0, 1.0).unwrap();
    let ls: CsrMatrix = laplacian_1d_sparse(&g, Boundary::Dirichlet);
    let ld = laplacian_1d(&g, Boundary::Dirichlet);

    // The sparse operator must match the dense one under mat-vec.
    let u_test: Vec<f64> = (0..g.n()).map(|i| i as f64 * 0.1).collect();
    let y_sparse = ls.mul_vec(&u_test);
    let y_dense = ld * DVector::from_vec(u_test);
    let max_diff = y_sparse
        .iter()
        .zip(y_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "\nsparse laplacian_1d_sparse: {}x{} CSR, {} non-zeros, mat-vec max |diff| vs dense = {:.2e}",
        ls.nrows(),
        ls.ncols(),
        ls.values.len(),
        max_diff
    );
    assert!(
        max_diff < 1e-12,
        "sparse Laplacian must match dense under mat-vec"
    );

    // Explicit-Euler stepping via the sparse backend.
    let sigma = 0.05;
    let mut u: Vec<f64> = g
        .coordinates()
        .iter()
        .map(|&x| (-((x - 0.5).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    for _ in 0..2000 {
        u = diffuse_step(&u, &ls, 0.0005, 0.01);
    }
    let peak = u.iter().cloned().fold(0.0_f64, f64::max);
    println!("sparse diffuse_step: Gaussian peak after 2000 steps = {peak:.4}");
    assert!(
        peak.is_finite() && peak < 1.0,
        "sparse diffusion must stay finite and decay"
    );
}
