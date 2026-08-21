//! # 2-D heat equation on a `UniformGrid2D` — a focused `tpt-sci-grid` demo.
//!
//! `tpt-sci-grid` assembles discrete PDE operators on structured tensor-product
//! grids. The existing `diffusion` example is a *tour* of the whole surface
//! (1-D/2-D/3-D, Dirichlet vs Neumann, Kronecker products, stencils, sparse).
//! This example instead solves one concrete, fully analytic 2-D problem so the
//! operator can be checked against a closed-form solution:
//!
//! ```text
//!     ∂u/∂t = D · (∂²u/∂x² + ∂²u/∂y²),   x, y ∈ [0, 1]
//! ```
//!
//! with the separable initial condition `u(x, y, 0) = sin(πx) · sin(πy)` and
//! homogeneous **Dirichlet** boundaries (`u = 0` on the box edge). Because the
//! initial field is the eigenfunction of the Laplacian for this domain, the
//! exact solution is simply
//!
//! ```text
//!     u(x, y, t) = sin(πx) · sin(πy) · exp(-2·D·π²·t)
//! ```
//!
//! so the dense [`laplacian_2d`] operator plus an explicit-Euler time step can
//! be validated to a few parts in a thousand against the formula.
//!
//! # What this example exercises
//!
//! * [`UniformGrid2D`] construction and the `index = ix + iy·nx` node ordering,
//! * the **dense** 2-D Laplacian ([`laplacian_2d`]) — no `sparse` feature needed,
//! * an explicit-Euler time stepping loop `u += dt·D·(L·u)` using the
//!   `tpt-math` dense [`DMatrix`]/[`DVector`] mat-vec,
//! * a quantitative self-check against the analytic solution at the box centre,
//! * a second run with **Neumann** (zero-flux) boundaries, where total heat is
//!   conserved, demonstrating the physical meaning of that boundary condition.
//!
//! # What to observe in the output
//!
//! * the Dirichlet run decays exactly as the analytic exponential (centre value
//!   matches to < 1%), confirming the assembled `L` is the true Laplacian,
//! * the Dirichlet field stays zero on the boundary at all times,
//! * the Neumann run conserves total heat far better than the Dirichlet run
//!   (to within a few percent — limited by the dense one-sided boundary
//!   stencil, which is not bit-exact at the walls) while the peak smooths out —
//!   the signature of a closed (insulated) domain.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_grid::{Boundary, UniformGrid2D, laplacian_2d};

/// Diffusion coefficient (thermal diffusivity).
const D: f64 = 0.01;
/// Explicit time step. Stability for 2-D explicit-Euler heat requires
/// `dt <= 1 / (2·D·(1/dx² + 1/dy²))`; with dx = dy = 1/40 this bound is
/// ~1.6e-2, so 1e-3 leaves a comfortable margin.
const DT: f64 = 1e-3;
/// Number of explicit steps (so `t_final = NSTEPS · DT`).
const NSTEPS: usize = 1000; // t_final = 1.0

/// Exact solution of the Dirichlet heat problem at `(x, y, t)`.
fn analytic(x: f64, y: f64, t: f64) -> f64 {
    (std::f64::consts::PI * x).sin()
        * (std::f64::consts::PI * y).sin()
        * (-2.0 * D * std::f64::consts::PI * std::f64::consts::PI * t).exp()
}

/// Solve the 2-D heat equation with homogeneous `bc`, initial field
/// `u0(x, y) = sin(πx)·sin(πy)`, for `NSTEPS` explicit-Euler steps.
/// Returns the final field (flat, `ix + iy·nx` ordering) plus its total mass.
fn heat_solve(g: &UniformGrid2D, bc: Boundary) -> (Vec<f64>, f64) {
    let l = laplacian_2d(g, bc);
    let xs = g.x_coordinates();
    let ys = g.y_coordinates();

    let mut u: Vec<f64> = (0..g.len())
        .map(|k| {
            let ix = k % g.nx();
            let iy = k / g.nx();
            (std::f64::consts::PI * xs[ix]).sin() * (std::f64::consts::PI * ys[iy]).sin()
        })
        .collect();
    let mass0: f64 = u.iter().sum();

    for _ in 0..NSTEPS {
        let lu = l.clone() * DVector::from_vec(u.clone());
        for (ui, li) in u.iter_mut().zip(lu.iter()) {
            *ui += DT * D * li;
        }
    }

    let mass1: f64 = u.iter().sum();
    (u, mass0 - mass1) // (final field, boundary mass flux / drift)
}

fn main() {
    println!("=== tpt-sci-grid: 2-D heat equation (u_t = D·∇²u) ===\n");

    let g = UniformGrid2D::new(41, 0.0, 1.0, 41, 0.0, 1.0).unwrap();
    let t_final = NSTEPS as f64 * DT;
    println!(
        "grid: {}x{} nodes over [0,1]x[0,1], dx = dy = {:.4}, D = {D}, dt = {DT}, steps = {NSTEPS} (t_final = {t_final})",
        g.nx(),
        g.ny(),
        g.dx()
    );

    // --- Dirichlet: compare against the closed-form solution -----------------
    let (u_dir, dir_flux) = heat_solve(&g, Boundary::Dirichlet);

    let cx = g.nx() / 2;
    let cy = g.ny() / 2;
    let center = cx + cy * g.nx();
    let xc = g.x_coordinates()[cx];
    let yc = g.y_coordinates()[cy];
    let u_num = u_dir[center];
    let u_exact = analytic(xc, yc, t_final);
    let rel = (u_num - u_exact).abs() / u_exact;

    println!("\n[Dirichlet]  u(x,y,0) = sin(πx)·sin(πy),  u = 0 on the boundary");
    println!(
        "  centre field u({:.3},{:.3},{:.1}) = {:.6}",
        xc, yc, t_final, u_num
    );
    println!("  analytic                = {u_exact:.6}");
    println!("  relative error          = {rel:.3e}");

    // Boundary must remain at zero (Dirichlet identity rows enforce it).
    let mut boundary_max = 0.0_f64;
    for iy in 0..g.ny() {
        for ix in [0, g.nx() - 1] {
            boundary_max = boundary_max.max(u_dir[ix + iy * g.nx()].abs());
        }
    }
    for ix in 0..g.nx() {
        for iy in [0, g.ny() - 1] {
            boundary_max = boundary_max.max(u_dir[ix + iy * g.nx()].abs());
        }
    }
    println!("  max |u| on the boundary = {boundary_max:.3e} (should be ~0)");

    assert!(u_num.is_finite(), "solution must stay finite");
    assert!(
        u_num > 0.0,
        "centre must remain positive (decaying, not sign-flipping)"
    );
    assert!(
        rel < 1e-2,
        "Dirichlet centre must match the analytic solution"
    );
    assert!(boundary_max < 1e-9, "Dirichlet boundary must stay at u = 0");
    // Heat leaks through the Dirichlet boundary (mass must drop).
    assert!(
        dir_flux > 0.0,
        "Dirichlet domain must lose heat through its boundary"
    );

    // --- Neumann: total heat is conserved (closed/insulated box) -------------
    let (u_neu, neu_flux) = heat_solve(&g, Boundary::Neumann);
    let peak0 = analytic(0.5, 0.5, 0.0); // = 1.0
    let peak1 = u_neu[center];

    println!("\n[Neumann]    u = 0 flux on the boundary (insulated box)");
    println!("  centre field u = {peak1:.6}  (started at {peak0:.3})");
    println!(
        "  total heat drift over {NSTEPS} steps = {:.3e} (should be ~0)",
        neu_flux
    );

    assert!(peak1.is_finite(), "Neumann solution must stay finite");
    // The peak must smooth out (never grow, since there is no interior source).
    assert!(peak1 <= peak0 + 1e-9, "Neumann peak must decay, not grow");
    // Closed domain: total heat is conserved far better than through the
    // Dirichlet boundary. The dense one-sided Neumann stencil is not bit-exact
    // at the walls, so we check the drift is small (a few percent) rather than
    // machine-zero; the key contrast is that Dirichlet bleeds heat (dir_flux > 0)
    // while Neumann keeps almost all of it.
    let mass0: f64 = (0..g.len())
        .map(|k| {
            let ix = k % g.nx();
            let iy = k / g.nx();
            (std::f64::consts::PI * g.x_coordinates()[ix]).sin()
                * (std::f64::consts::PI * g.y_coordinates()[iy]).sin()
        })
        .sum();
    let mass1: f64 = u_neu.iter().sum();
    let rel_drift = (mass1 - mass0).abs() / mass0;
    println!(
        "  Neumann relative heat drift = {:.3e}  (Dirichlet bleeds {:.3e})",
        rel_drift,
        dir_flux / mass0
    );
    assert!(
        rel_drift < 0.05,
        "Neumann total heat must be conserved to within a few percent (drift {rel_drift:.3e})"
    );

    println!(
        "\nAll checks passed: dense 2-D Laplacian + explicit step reproduce the\n\
         analytic Dirichlet heat kernel and conserve Neumann heat."
    );
}
