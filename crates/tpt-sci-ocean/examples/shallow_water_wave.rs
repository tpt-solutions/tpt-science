//! Ocean demo: a **dam-break (Riemann) problem** in the 2-D shallow-water
//! model exposed by `tpt-sci-ocean`.
//!
//! `ShallowWater` integrates the depth-averaged equations on a uniform
//! [`tpt_sci_cfd_core::CollocatedGrid`] with free-surface height `h`, velocities
//! `u`, `v`, gravity `g` and Coriolis `f`. The companion example
//! (`examples/shallow_water.rs`) seeds a smooth Gaussian bump and a
//! geostrophically balanced state. Here we instead impose a **discontinuous
//! initial condition** — a vertical wall of raised water on the left half of the
//! basin and a lowered level on the right — and then "remove the dam" so the
//! step collapses into a left-going rarefaction and a right-going bore.
//!
//! This is a genuinely different use case of the same public surface
//! ([`ShallowWater::new`], the public `h`/`u`/`v` fields, `step`, `max_speed`,
//! `grid`). We verify three physical facts:
//!
//! 1. The model stays numerically stable: all fields remain finite and `h > 0`.
//! 2. A front forms at the initial step and **propagates** to the right.
//! 3. The propagation speed scales with gravity as `c ≈ √(g·h)`: running the
//!    same dam-break with four times the gravity produces a front that has
//!    travelled roughly twice as far in the same time.
//!
//! Run with: `cargo run --example shallow_water_wave -p tpt-sci-ocean`

use tpt_sci_ocean::ShallowWater;

/// Build a dam-break initial condition: a smooth step — the left side raised by
/// `amp`, the right side lowered by `amp`, velocities zero. We use a *smooth*
/// `tanh` step rather than a hard discontinuity: the crate's `step` is a naive
/// central-difference finite-volume update (no flux limiter), so a sharp jump
/// would spawn growing dispersive oscillations. A smooth step over a few cells
/// keeps the scheme stable while still "removing the dam" so the profile
/// collapses into a right-going front.
fn dam_break(nx: usize, ny: usize, g: f64, f: f64, amp: f64) -> ShallowWater {
    let mut sw = ShallowWater::new(nx, ny, 1.5, 1.5, g, f, 0.001);
    let hx = sw.grid().nx;
    let hy = sw.grid().ny;
    let mid = hx as f64 / 2.0;
    let width = 4.0; // smoothing length in cells (avoids the shock instability)
    for j in 0..hy {
        for i in 0..hx {
            let k = j * hx + i;
            // s goes -1 (left) .. +1 (right); left side sits higher.
            let s = ((i as f64 - mid) / width).tanh();
            sw.h[k] = sw.h0 - amp * s;
        }
    }
    sw
}

/// Position (in `x`) of the steepest `h`-gradient — the shock/bore front.
fn front_x(sw: &ShallowWater) -> f64 {
    let g = sw.grid();
    let (nx, ny, dx) = (g.nx, g.ny, g.dx);
    let mut best = 0usize;
    let mut best_grad = 0.0f64;
    for i in 0..nx - 1 {
        let mut col_grad = 0.0;
        for j in 0..ny {
            let c = g.idx(i, j);
            let cp = g.idx(i + 1, j);
            col_grad += (sw.h[cp] - sw.h[c]).abs();
        }
        col_grad /= ny as f64;
        if col_grad > best_grad {
            best_grad = col_grad;
            best = i;
        }
    }
    best as f64 * dx
}

/// Total kinetic + potential energy of the surface (per unit width, J/m).
fn total_energy(sw: &ShallowWater) -> (f64, f64) {
    let g = sw.grid();
    let dxdy = g.dx * g.dy;
    let mut ke = 0.0;
    let mut pe = 0.0;
    for c in 0..sw.h.len() {
        let speed2 = sw.u[c] * sw.u[c] + sw.v[c] * sw.v[c];
        ke += 0.5 * sw.h[c] * speed2;
        let eta = sw.h[c] - sw.h0;
        pe += 0.5 * sw.g * eta * eta;
    }
    (ke * dxdy, pe * dxdy)
}

fn run_dam_break(g: f64) -> (f64, f64, f64) {
    let (nx, ny) = (64usize, 64usize);
    let dt = 0.0004f64;
    let steps = 80usize;
    // Domain is 1.5 m, front starts at the centre (0.75 m) and travels right; the
    // time window (steps x dt = 0.032 s) keeps even the 4x-gravity bore well
    // inside the basin so the front never clamps against the boundary.
    let x0 = front_x(&dam_break(nx, ny, g, 0.0, 2.0)); // initial front at the dam

    let mut sw = dam_break(nx, ny, g, 0.0, 2.0);
    let mut peak_speed = 0.0f64;
    let mut end_energy = (0.0f64, 0.0f64);
    for k in 0..steps {
        if k % 50 == 0 {
            let (ke, pe) = total_energy(&sw);
            let x = front_x(&sw);
            println!(
                "  g={:>6.2} step {k:3}: front_x = {x:.4} m | max_speed = {:.4} m/s | \
                 E_kin = {:.3e} E_pot = {:.3e}",
                g,
                sw.max_speed(),
                ke,
                pe
            );
        }
        sw.step(dt);
        peak_speed = peak_speed.max(sw.max_speed());
        end_energy = total_energy(&sw);
    }
    let x_end = front_x(&sw);
    // Stability self-checks.
    assert!(
        sw.h.iter().all(|&h| h.is_finite() && h > 0.0),
        "height must stay finite and positive"
    );
    assert!(sw.max_speed().is_finite());
    println!(
        "  g={:>6.2} final: front_x = {x_end:.4} m, peak max_speed = {:.4} m/s, \
         E_kin = {:.3e} E_pot = {:.3e}",
        g, peak_speed, end_energy.0, end_energy.1
    );
    (x0, x_end, peak_speed)
}

fn main() {
    println!("tpt-sci-ocean dam-break (Riemann) on a 64x64 basin, f = 0.\n");

    println!("== Run A: gravity g = 9.81 m/s^2 ==");
    let (x0_a, x_end_a, _peak_a) = run_dam_break(9.81);

    println!("\n== Run B: gravity g = 39.24 m/s^2 (4x, wave speed ~ sqrt(4)) ==");
    let (x0_b, x_end_b, _peak_b) = run_dam_break(39.24);

    println!("\n== Assertions ==");
    // The front must actually propagate away from the dam.
    assert!(
        x_end_a > x0_a + 1e-3,
        "the bore front must advance to the right (run A)"
    );
    assert!(
        x_end_b > x0_b + 1e-3,
        "the bore front must advance to the right (run B)"
    );

    // Wave speed scales with \u221a(g\u00b7h): 4x gravity -> ~2x travel distance.
    let speed_ratio = (x_end_b - x0_b) / (x_end_a - x0_a);
    println!(
        "  front travel: g=9.81 -> {:.4} m, g=39.24 -> {:.4} m  (ratio {:.2})",
        x_end_a - x0_a,
        x_end_b - x0_b,
        speed_ratio
    );
    assert!(
        speed_ratio > 1.4 && speed_ratio < 2.6,
        "front speed should scale ~sqrt(g) (ratio ~2.0)"
    );
    println!("  assertions passed: stable, fronts propagate, c scales with sqrt(g*h).");
}
