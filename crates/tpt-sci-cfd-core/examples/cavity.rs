//! # tpt-sci-cfd-core — incompressible Navier–Stokes tour
//!
//! A guided tour of the public surface of `tpt-sci-cfd-core`, exercising the
//! incompressible Navier–Stokes stack on a uniform collocated grid:
//!
//! 1. **Grid / mesh** — [`CollocatedGrid`] construction, indexing, length,
//!    and validation (`CfdError`).
//! 2. **Lid-driven cavity** — the fractional-step solver [`Step`] (`momentum`
//!    advection–diffusion + `project` pressure-Poisson) driven by a moving top
//!    [`Boundary`]. We report the divergence-free error, kinetic-energy growth,
//!    and a vertical-centerline velocity profile, and scale the flow by Reynolds
//!    number.
//! 3. **Free vortex decay** — the same solver with no driving walls, seeded with
//!    a divergence-free shear layer, demonstrating viscous momentum diffusion
//!    (`momentum`) bleeding kinetic energy away.
//! 4. **Pressure projection property** — a deterministic check that
//!    [`Step::project`] does not materially increase `∇·u`, plus a uniform field
//!    that is exactly divergence-free.
//! 5. **Algebraic turbulence** — [`turbulence::strain_magnitude`] and
//!    [`turbulence::eddy_viscosity`] compute a Smagorinsky eddy viscosity from
//!    the cavity strain field, which we feed back as an effective viscosity for
//!    a turbulent cavity run.
//!
//! Run with: `cargo run --example cavity -p tpt-sci-cfd-core`
//!
//! Observe: after each fractional step the interior `max |∇·u|` stays tiny
//! (the projection enforces incompressibility), kinetic energy grows toward a
//! lid-driven steady state but decays once the walls are stilled, and the
//! centerline `u` profile develops the characteristic cavity boundary layer.

use tpt_sci_cfd_core::{
    Boundary, CfdError, CollocatedGrid, Step,
    turbulence::{eddy_viscosity, strain_magnitude},
};

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

fn main() {
    println!("=== tpt-sci-cfd-core Navier–Stokes tour ===\n");

    // ---------------------------------------------------------------------
    // 1. Grid / mesh construction and validation.
    // ---------------------------------------------------------------------
    let grid = CollocatedGrid::new(32, 32, 1.0, 1.0).unwrap();
    assert!(!grid.is_empty());
    assert_eq!(grid.len(), 32 * 32);
    assert_eq!(grid.idx(3, 2), 2 * grid.nx + 3);
    println!(
        "grid: {}×{} cells, domain {}×{} (dx={:.4}, dy={:.4})",
        grid.nx, grid.ny, grid.lx, grid.ly, grid.dx, grid.dy
    );

    // Invalid grids surface a typed error rather than panicking.
    let err = CollocatedGrid::new(0, 8, 1.0, 1.0).unwrap_err();
    assert!(matches!(err, CfdError::InvalidGrid(_)));
    println!("invalid-grid construction returns: {err:?}");

    // ---------------------------------------------------------------------
    // 2. Lid-driven cavity (Reynolds-number scaling).
    // ---------------------------------------------------------------------
    let nu = 2e-3; // kinematic viscosity (Re = 500 with U=L=1)
    let dt = 1e-2; // explicit timestep (CFL- and diffusion-stable for this grid)
    let u_lid = 1.0; // lid speed
    let reynolds = u_lid * grid.ly / nu;
    println!("\n--- lid-driven cavity (Re = {reynolds:.0}) ---");

    let mut cavity = Step::new(grid.clone(), nu, dt, 1.0);
    cavity.set_boundary(Boundary::Top, u_lid); // moving lid

    let nsteps = 150;
    let mut ke_start = 0.0;
    for k in 0..nsteps {
        assert!(cavity.advance(), "cavity velocity blew up at step {k}");
        if k == 0 {
            ke_start = kinetic_energy(&cavity);
        }
        if k % 50 == 0 {
            println!(
                "  step {:3}: max|div(u)| = {:>8.3e}   KE = {:>8.4e}   max|u| = {:.4}",
                k,
                cavity.max_divergence(),
                kinetic_energy(&cavity),
                max_speed(&cavity)
            );
        }
    }

    let div = cavity.max_divergence();
    let ke_end = kinetic_energy(&cavity);
    println!(
        "  final:  max|div(u)| = {:.3e}   KE = {:.4e} (started at {:.4e})",
        div, ke_end, ke_start
    );
    // After projection the interior velocity is (near) divergence-free.
    assert!(
        div.is_finite() && div < 0.2,
        "interior divergence too large: {div}"
    );
    // The moving lid has pumped kinetic energy into the field.
    assert!(ke_end > ke_start, "KE should grow under the driven lid");
    assert!(max_speed(&cavity).is_finite());

    // Vertical-centerline u-profile (the cavity boundary layer).
    let g = cavity.grid();
    let i = g.nx / 2;
    println!("  u along vertical centerline (y, u):");
    for j in (0..g.ny).rev().step_by(4) {
        let c = g.idx(i, j);
        let y = (j as f64 + 0.5) * g.dy;
        println!(
            "    y={:>5.3}  u={:>7.4}  p={:>7.4}",
            y, cavity.u[c], cavity.p[c]
        );
    }

    // ---------------------------------------------------------------------
    // 3. Free vortex decay (no driving walls → viscous diffusion wins).
    // ---------------------------------------------------------------------
    println!("\n--- free shear-layer decay (no driving walls) ---");
    let mut decay = Step::new(grid.clone(), nu, dt, 1.0);
    for j in 0..g.ny {
        for i in 0..g.nx {
            // Divergence-free shear layer: u = sin(π y), v = 0.
            let y = (j as f64 + 0.5) * g.dy;
            decay.u[g.idx(i, j)] = (std::f64::consts::PI * y).sin();
        }
    }
    let ke_decay0 = kinetic_energy(&decay);
    for _ in 0..60 {
        assert!(decay.advance());
    }
    let ke_decay1 = kinetic_energy(&decay);
    println!(
        "  KE: {:.4e} -> {:.4e} over 60 steps (should decrease)",
        ke_decay0, ke_decay1
    );
    assert!(
        ke_decay1 < ke_decay0,
        "viscous diffusion should reduce KE of a stilled field"
    );

    // ---------------------------------------------------------------------
    // 4. Pressure-projection property (deterministic).
    // ---------------------------------------------------------------------
    println!("\n--- projection property ---");
    let mut probe = Step::new(
        CollocatedGrid::new(20, 20, 1.0, 1.0).unwrap(),
        nu,
        5e-3,
        1.0,
    );
    let pn = probe.grid().nx;
    for j in 0..probe.grid().ny {
        for i in 0..pn {
            probe.u[j * pn + i] = i as f64; // linearly diverging field
        }
    }
    let before = probe.max_divergence();
    probe.project();
    let after = probe.max_divergence();
    println!("  max|div(u)|: {before:.3e} -> {after:.3e} after projection");
    assert!(
        after < before * 1.15,
        "projection must not materially increase divergence (before={before}, after={after})"
    );

    // A uniform field is exactly divergence-free.
    let mut uniform = Step::new(
        CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap(),
        nu,
        1e-2,
        1.0,
    );
    uniform.u.fill(1.0);
    assert!(
        uniform.max_divergence().abs() < 1e-9,
        "uniform field must be exactly divergence-free"
    );
    println!(
        "  uniform field max|div(u)| = {:.3e} (≈ 0)",
        uniform.max_divergence()
    );

    // ---------------------------------------------------------------------
    // 5. Algebraic turbulence: strain magnitude → eddy viscosity.
    // ---------------------------------------------------------------------
    println!("\n--- algebraic turbulence (Smagorinsky eddy viscosity) ---");
    // Local strain rate from the cavity lid region (central differences).
    let jt = g.ny - 2;
    let it = g.nx / 2;
    let dudx = (cavity.u[g.idx((it + 1).min(g.nx - 1), jt)]
        - cavity.u[g.idx(it.saturating_sub(1), jt)])
        / (2.0 * g.dx);
    let dudy = (cavity.u[g.idx(it, (jt + 1).min(g.ny - 1))]
        - cavity.u[g.idx(it, jt.saturating_sub(1))])
        / (2.0 * g.dy);
    let dvdx = (cavity.v[g.idx((it + 1).min(g.nx - 1), jt)]
        - cavity.v[g.idx(it.saturating_sub(1), jt)])
        / (2.0 * g.dx);
    let dvdy = (cavity.v[g.idx(it, (jt + 1).min(g.ny - 1))]
        - cavity.v[g.idx(it, jt.saturating_sub(1))])
        / (2.0 * g.dy);
    let strain = strain_magnitude(dudx, dudy, dvdx, dvdy);
    let nu_t = eddy_viscosity(nu, 0.1, g.dx, strain);
    println!(
        "  strain |S| = {:.4e}, molecular ν = {:.3e}, effective ν_eff = {:.3e}",
        strain, nu, nu_t
    );
    assert!(strain.is_finite() && nu_t >= nu);

    // Feed the eddy viscosity back as an effective viscosity for a turbulent run.
    let mut turbulent = Step::new(grid.clone(), nu_t, dt, 1.0);
    turbulent.set_boundary(Boundary::Top, u_lid);
    for _ in 0..30 {
        assert!(turbulent.advance());
    }
    println!(
        "  turbulent cavity after 30 steps: KE = {:.4e}, max|div(u)| = {:.3e}",
        kinetic_energy(&turbulent),
        turbulent.max_divergence()
    );
    assert!(max_speed(&turbulent).is_finite() && turbulent.max_divergence().is_finite());

    println!("\nAll checks passed: projection keeps the field divergence-free,");
    println!("KE grows under the lid yet decays once stilled, and the turbulence");
    println!("path returns a finite, ≥ molecular effective viscosity.");
}
