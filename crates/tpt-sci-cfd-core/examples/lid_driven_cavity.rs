//! # tpt-sci-cfd-core — lid-driven cavity Reynolds study
//!
//! A second, independent example for `tpt-sci-cfd-core` that exercises the same
//! public incompressible Navier–Stokes stack as `examples/cavity.rs` but on a
//! *different* setup and extracts a *different* diagnostic:
//!
//! * A **finer, square 48×48 grid** (vs the 32×32 tour) with a **higher
//!   Reynolds number** drive (`Re = 2000`, achieved with a smaller kinematic
//!   viscosity `ν`).
//! * A **Reynolds-number sweep** (`Re ∈ {100, 500, 1000, 2000}`) on a coarse
//!   grid, reporting the pseudo-steady kinetic energy and the post-projection
//!   interior divergence for each, so the example characterises how the
//!   fractional-step solver loads up as `Re` rises.
//! * A **horizontal centerline `u`-profile** through the cavity mid-height — the
//!   companion to the vertical profile in `cavity.rs` — which should show the
//!   classic two-roll structure (fast lid-driven flow near the top, recirculation
//!   below) once the lid has been driven for many steps.
//!
//! Run with: `cargo run --example lid_driven_cavity -p tpt-sci-cfd-core`
//!
//! The whole flow is driven only through the public API:
//! [`CollocatedGrid::new`], [`Step::new`], [`Step::set_boundary`] and
//! [`Step::advance`], with [`Step::max_divergence`] and the public `u`/`p`
//! fields used for the diagnostics. No private surface is touched.

use tpt_sci_cfd_core::{Boundary, CollocatedGrid, Step};

/// Total kinetic energy `½ Σ (u² + v²)` of a velocity field.
fn kinetic_energy(step: &Step) -> f64 {
    let mut ke = 0.0;
    for k in 0..step.u.len() {
        ke += 0.5 * (step.u[k] * step.u[k] + step.v[k] * step.v[k]);
    }
    ke
}

/// Maximum absolute speed `max |(u, v)|` over the field.
fn max_speed(step: &Step) -> f64 {
    let mut m = 0.0_f64;
    for k in 0..step.u.len() {
        m = m.max(step.u[k].abs().max(step.v[k].abs()));
    }
    m
}

/// Run the lid-driven cavity to a pseudo-steady state and report diagnostics.
fn run_cavity(nx: usize, ny: usize, nu: f64, dt: f64, u_lid: f64, nsteps: usize) -> Step {
    let grid = CollocatedGrid::new(nx, ny, 1.0, 1.0).unwrap();
    let mut step = Step::new(grid, nu, dt, 1.0);
    step.set_boundary(Boundary::Top, u_lid);
    for k in 0..nsteps {
        assert!(
            step.advance(),
            "lid-driven cavity velocity blew up at step {k} (nu={nu}, dt={dt})"
        );
    }
    step
}

fn main() {
    println!("=== tpt-sci-cfd-core lid-driven cavity Reynolds study ===\n");

    // ---------------------------------------------------------------------
    // 1. High-Reynolds cavity on a finer grid (Re = 2000).
    // ---------------------------------------------------------------------
    let u_lid = 1.0;
    let nu = 5e-4; // Re = u_lid * L / nu = 2000 on the unit square
    let dt = 5e-3; // explicit timestep (CFL- and diffusion-stable at nu=5e-4)
    let reynolds = u_lid * 1.0 / nu;
    println!("--- fine cavity (48×48, Re = {reynolds:.0}) ---");

    let nsteps = 400;
    let cavity = run_cavity(48, 48, nu, dt, u_lid, nsteps);

    let div = cavity.max_divergence();
    let ke = kinetic_energy(&cavity);
    let speed = max_speed(&cavity);
    println!(
        "  after {nsteps} steps: max|div(u)| = {:.3e}   KE = {:.4e}   max|u| = {:.4}",
        div, ke, speed
    );
    assert!(div.is_finite() && div < 0.2, "interior divergence too large: {div}");
    assert!(speed.is_finite() && speed > 0.0, "lid should drive a non-zero field");
    assert!(
        ke > 1e-6,
        "a driven lid must pump kinetic energy into the field (KE={ke})"
    );

    // ---------------------------------------------------------------------
    // 2. Horizontal centerline u-profile through the cavity mid-height.
    //    Distinct from cavity.rs, which printed the *vertical* centerline.
    // ---------------------------------------------------------------------
    let g = cavity.grid();
    let j = g.ny / 2; // mid-height row
    println!("\n  u along horizontal centerline (x, u, p) at y = {:.3}:", (j as f64 + 0.5) * g.dy);
    let mut profile = Vec::with_capacity(g.nx);
    for i in 0..g.nx {
        let c = g.idx(i, j);
        profile.push(cavity.u[c]);
        println!(
            "    x={:>5.3}  u={:>7.4}  p={:>7.4}",
            (i as f64 + 0.5) * g.dx,
            cavity.u[c],
            cavity.p[c]
        );
    }
    assert!(
        profile.iter().all(|&u| u.is_finite()),
        "centerline u must be finite everywhere"
    );
    // The lid drags fluid rightward overall: the bulk of the centerline should
    // have positive (rightward) u, with recirculation dipping negative.
    let mean_u: f64 = profile.iter().sum::<f64>() / profile.len() as f64;
    println!("    mean centerline u = {mean_u:.4}");
    assert!(
        mean_u > 0.0,
        "lid-driven mean centerline velocity should be rightward (mean={mean_u})"
    );

    // ---------------------------------------------------------------------
    // 3. Reynolds-number sweep on a coarse grid: how does the solver load up
    //    as Re increases? Report final KE and post-projection divergence.
    // ---------------------------------------------------------------------
    println!("\n--- Reynolds sweep (24×24, 200 steps each) ---");
    println!("  {:>6} {:>10} {:>12} {:>12}", "Re", "nu", "KE", "max|div|");
    let coarse_dt = 5e-3;
    let mut last_div = 0.0_f64;
    let mut last_ke = 0.0_f64;
    for re in [100.0, 500.0, 1000.0, 2000.0] {
        let nuv = u_lid * 1.0 / re;
        let step = run_cavity(24, 24, nuv, coarse_dt, u_lid, 200);
        let d = step.max_divergence();
        let e = kinetic_energy(&step);
        println!("  {:>6.0} {:>10.2e} {:>12.4e} {:>12.3e}", re, nuv, e, d);
        assert!(d.is_finite() && e.is_finite());
        // Higher Re (lower viscosity) stores more kinetic energy for the same
        // drive, since viscous dissipation is weaker.
        if last_ke > 0.0 {
            assert!(
                e >= last_ke * 0.5,
                "KE should not collapse as Re rises (prev={last_ke}, this={e})"
            );
        }
        last_div = d;
        last_ke = e;
    }
    assert!(last_div < 0.2, "sweep divergence should stay small: {last_div}");

    println!("\nAll checks passed: the fine Re=2000 cavity reached a finite,");
    println!("near-divergence-free state with a rightward mid-height profile, and");
    println!("the Reynolds sweep loaded up kinetic energy without blowing up.");
}
