//! Ocean demo: a tour of the 2-D shallow-water surface exposed by
//! `tpt-sci-ocean`.
//!
//! `ShallowWater` integrates the 2-D shallow-water (primitive-equation
//! prototype) equations on a uniform [`tpt_sci_cfd_core::CollocatedGrid`]:
//! free-surface height `h` plus depth-averaged velocities `u`, `v`, with
//! gravity `g` and Coriolis `f`.
//!
//! This example exercises the public surface in three slices:
//!
//! 1. **Gravity waves** — a Gaussian bump in `h` (zero velocity) is unbalanced
//!    and immediately radiates fast gravity waves; we track the free-surface
//!    statistics, total (kinetic + potential) energy, and max speed.
//! 2. **Geostrophic balance** — we seed a gentle height anomaly and set the
//!    velocities from the geostrophic relation (`f·v = g·∂η/∂x`,
//!    `f·u = -g·∂η/∂y`). A balanced state is nearly steady, so its max speed
//!    and geostrophic residual stay small.
//! 3. **Comparison** — the balanced state should stay slower than the
//!    unbalanced bump, demonstrating that geostrophic balance suppresses fast
//!    gravity-wave radiation.
//!
//! Run with: `cargo run --example shallow_water -p tpt-sci-ocean`

use tpt_sci_ocean::ShallowWater;

/// Min / max / mean of the free-surface height `h` over the whole grid.
fn surface_stats(sw: &ShallowWater) -> (f64, f64, f64) {
    let (mut mn, mut mx, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for &h in &sw.h {
        mn = mn.min(h);
        mx = mx.max(h);
        sum += h;
    }
    (mn, mx, sum / sw.h.len() as f64)
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

/// Mean absolute geostrophic residual: `|f·v - g·∂η/∂x|` and
/// `|f·u + g·∂η/∂y|` should both be ~0 for a balanced state.
fn geostrophic_residual(sw: &ShallowWater) -> (f64, f64) {
    let g = sw.grid();
    let (nx, ny, dx, dy) = (g.nx, g.ny, g.dx, g.dy);
    let idx = |i: usize, j: usize| j * nx + i;
    let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
    let mut rx = 0.0;
    let mut ry = 0.0;
    let mut n = 0usize;
    for j in 0..ny {
        for i in 0..nx {
            let c = idx(i, j);
            let im = idx(clamp(i as isize - 1, nx), j);
            let ip = idx(clamp(i as isize + 1, nx), j);
            let jm = idx(i, clamp(j as isize - 1, ny));
            let jp = idx(i, clamp(j as isize + 1, ny));
            let dndx = (sw.h[ip] - sw.h[im]) / (2.0 * dx);
            let dndy = (sw.h[jp] - sw.h[jm]) / (2.0 * dy);
            rx += (sw.f * sw.v[c] - sw.g * dndx).abs();
            ry += (sw.f * sw.u[c] + sw.g * dndy).abs();
            n += 1;
        }
    }
    (rx / n as f64, ry / n as f64)
}

/// Build a geostrophically balanced state: gentle height anomaly `h`, with `u`
/// and `v` set from the geostrophic relations.
fn balanced_state() -> ShallowWater {
    let mut sw = ShallowWater::new(48, 48, 1.0, 1.0, 9.81, 1e-4, 0.001);
    // Gentle bump -> geostrophic velocities of order 1 m/s (g/f is large).
    sw.perturb_center(1e-7);
    let g = sw.grid();
    let (nx, ny, dx, dy) = (g.nx, g.ny, g.dx, g.dy);
    let idx = |i: usize, j: usize| j * nx + i;
    let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
    for j in 0..ny {
        for i in 0..nx {
            let c = idx(i, j);
            let im = idx(clamp(i as isize - 1, nx), j);
            let ip = idx(clamp(i as isize + 1, nx), j);
            let jm = idx(i, clamp(j as isize - 1, ny));
            let jp = idx(i, clamp(j as isize + 1, ny));
            let dndx = (sw.h[ip] - sw.h[im]) / (2.0 * dx);
            let dndy = (sw.h[jp] - sw.h[jm]) / (2.0 * dy);
            sw.v[c] = (sw.g / sw.f) * dndx;
            sw.u[c] = -(sw.g / sw.f) * dndy;
        }
    }
    sw
}

fn main() {
    let dt = 0.0005;
    let steps = 80;
    let (nx, ny) = (48usize, 48usize);

    println!("tpt-sci-ocean ShallowWater tour on a {nx}x{ny} basin (g=9.81, f=1e-4, dt={dt}).\n");

    // --- Slice 1: unbalanced Gaussian bump -> gravity waves -----------------
    let mut waves = ShallowWater::new(nx, ny, 1.0, 1.0, 9.81, 1e-4, dt);
    waves.perturb_center(0.4);
    println!("== 1. Unbalanced bump (gravity waves) ==");
    let mut wave_max_speed = 0.0f64;
    for k in 0..steps {
        if k % 20 == 0 {
            let (mn, mx, mean) = surface_stats(&waves);
            let (ke, pe) = total_energy(&waves);
            println!(
                "  step {k:3}: max_speed={:.4} m/s | h∈[{:.4},{:.4}] mean={:.4} | E_kin={:.3e} E_pot={:.3e}",
                waves.max_speed(),
                mn,
                mx,
                mean,
                ke,
                pe
            );
        }
        waves.step(dt);
        wave_max_speed = wave_max_speed.max(waves.max_speed());
    }
    println!("  peak max speed over run: {wave_max_speed:.4} m/s\n");

    // --- Slice 2: geostrophically balanced state ----------------------------
    let mut geo = balanced_state();
    println!("== 2. Geostrophically balanced state ==");
    let (g_rx0, g_ry0) = geostrophic_residual(&geo);
    let geo_max0 = geo.max_speed();
    let mut geo_max_speed = geo_max0;
    for k in 0..steps {
        if k % 20 == 0 {
            let (rx, ry) = geostrophic_residual(&geo);
            println!(
                "  step {k:3}: max_speed={:.4} m/s | geo_residual x={:.3e} y={:.3e}",
                geo.max_speed(),
                rx,
                ry
            );
        }
        geo.step(dt);
        geo_max_speed = geo_max_speed.max(geo.max_speed());
    }
    let (g_rx1, g_ry1) = geostrophic_residual(&geo);
    println!("  initial max speed={geo_max0:.4} m/s, peak over run={geo_max_speed:.4} m/s");
    println!("  geo residual: start ({g_rx0:.3e},{g_ry0:.3e}) -> end ({g_rx1:.3e},{g_ry1:.3e})\n");

    // --- Slice 3: comparison / assertions -----------------------------------
    println!("== 3. Comparison ==");
    println!(
        "  unbalanced peak max speed = {wave_max_speed:.4} m/s, balanced peak = {geo_max_speed:.4} m/s"
    );

    // The constructed state is geostrophically balanced: residual ~0 at t=0.
    assert!(
        g_rx0 < 1e-3 && g_ry0 < 1e-3,
        "initial state must be balanced"
    );
    let all_finite = waves.h.iter().chain(geo.h.iter()).all(|x| x.is_finite())
        && waves.max_speed().is_finite()
        && geo.max_speed().is_finite();
    assert!(all_finite, "solution fields must stay finite");
    assert!(
        geo_max_speed < wave_max_speed,
        "balanced state should stay slower than the unbalanced gravity-wave bump"
    );
    // Near-steady: the balanced flow does not radiate fast gravity waves, so its
    // peak speed stays close to (a few times) its initial value.
    assert!(
        geo_max_speed < geo_max0 * 2.0 + 1e-3,
        "balanced state should stay near-steady (low speed)"
    );
    println!(
        "  assertions passed: balanced at t=0 (residual ~0), finite fields, \
         balanced < unbalanced, near-steady."
    );
}
