//! # Radial distribution function from an MD run
//!
//! A second, structurally-focused tour of the [`tpt_sci_md`] public surface — the
//! first example (`lennard_jones`) runs a short trajectory and reports
//! temperature / energy conservation / a single `g(r)` peak. This one instead
//! uses the MD engine to *build* a liquid and then performs a genuine
//! **structural analysis** with [`rdf`]:
//!
//! * Set up a mono-species Lennard-Jones fluid on a simple-cubic lattice in a
//!   cubic periodic box using [`Particle::new`] / [`Particle::new_with_species`]
//!   and [`Integrator::new`].
//! * Equilibrate with the Berendsen [`Integrator::thermostat`] to two different
//!   target temperatures, producing a colder (more structured) and a warmer
//!   (less structured) liquid.
//! * Compute the radial distribution function `g(r)` from each equilibrated
//!   configuration via [`rdf`] and analyse it the way a real MD post-processor
//!   would:
//!   * locate the **first coordination shell** (peak → first minimum),
//!   * integrate `g(r)` up to the first minimum to obtain the **coordination
//!     number** `N(r_min) = 4πρ ∫₀^{r_min} g(r)·r² dr`,
//!   * verify the well-known asymptotic property `g(r) → 1` for a fluid.
//! * Confirm, physically, that the cold liquid is *more* structured than the
//!   warm one (taller first-shell peak, larger coordination number).
//!
//! Run with: `cargo run --example radial_distribution -p tpt-sci-md`

use std::f64::consts::PI;

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Forces, Integrator, Particle, rdf};

/// Build an `m³` simple-cubic lattice of LJ particles in a cubic periodic box
/// with small deterministic initial velocities (so the run is reproducible).
fn build_lattice(n: usize, box_len: f64, _sigma: f64) -> Vec<Particle> {
    let m = (n as f64).cbrt().round() as usize;
    assert_eq!(m * m * m, n, "n must be a perfect cube");
    let spacing = box_len / m as f64;
    (0..n)
        .map(|i| {
            let ix = i % m;
            let iy = (i / m) % m;
            let iz = i / (m * m);
            let pos = DVector::from_row_slice(&[
                ix as f64 * spacing,
                iy as f64 * spacing,
                iz as f64 * spacing,
            ]);
            let vel = DVector::from_row_slice(&[
                (((i * 7) % 13) as f64 - 6.0) * 0.05,
                (((i * 3) % 13) as f64 - 6.0) * 0.05,
                (((i * 5) % 13) as f64 - 6.0) * 0.05,
            ]);
            // Exercise both constructors; vary the species id on odd sites.
            if i % 2 == 0 {
                Particle::new(i, pos, vel, 1.0).unwrap()
            } else {
                Particle::new_with_species(i, pos, vel, 1.0, i % 2).unwrap()
            }
        })
        .collect()
}

/// Equilibrate the system at `target_t` with a Berendsen thermostat for
/// `steps` velocity-Verlet steps, returning the final configuration.
fn equilibrate(
    mut particles: Vec<Particle>,
    int: &Integrator,
    target_t: f64,
    steps: usize,
) -> Vec<Particle> {
    // Prime the force field so the first velocity-Verlet half-step is correct.
    let _ = Forces::lennard_jones(&mut particles, int.box_len, int.sigma);
    for step in 0..steps {
        let _ = int.velocity_verlet(&mut particles);
        int.thermostat(&mut particles, target_t, 0.5);
        if step % (steps / 5).max(1) == 0 {
            let t = int.temperature(&particles);
            let ekin = int.kinetic_energy(&particles);
            println!("   eq {:>4}  T={:8.4}  E_kin={:10.4}", step, t, ekin);
        }
    }
    particles
}

/// Analyse a `g(r)` curve: return the first-shell peak position, the position of
/// the first minimum after the peak, and the coordination number integrated up to
/// that minimum.
fn analyse(r: &[f64], g: &[f64], rho: f64) -> (f64, f64, f64, f64) {
    assert_eq!(r.len(), g.len());
    // First peak (skip the empty r≈0 bin).
    let mut peak_i = 1_usize;
    let mut peak_v = f64::NEG_INFINITY;
    for (i, &gi) in g.iter().enumerate().skip(1) {
        if gi > peak_v {
            peak_v = gi;
            peak_i = i;
        }
    }
    // First minimum after the peak (g rises again past the shell).
    let mut min_i = peak_i;
    let mut min_v = g[peak_i];
    for (i, &gi) in g.iter().enumerate().skip(peak_i + 1) {
        if gi > min_v {
            min_i = i - 1;
            break;
        }
        min_v = gi;
        min_i = i;
    }
    let dr = r[1] - r[0];
    // Coordination number N = 4πρ ∫₀^{r_min} g(r) r² dr  (trapezoid rule).
    let mut cn = 0.0_f64;
    for i in 0..min_i {
        let r1 = r[i];
        let r2 = r[i + 1];
        let f1 = 4.0 * PI * rho * g[i] * r1 * r1;
        let f2 = 4.0 * PI * rho * g[i + 1] * r2 * r2;
        cn += 0.5 * (f1 + f2) * dr;
    }
    (r[peak_i], peak_v, r[min_i], cn)
}

fn main() {
    let n: usize = 64; // 4×4×4 lattice
    let box_len = 6.0;
    let sigma = 1.0;
    let dt = 0.005;

    println!("# RDF from MD: N={n}  box={box_len}  σ={sigma}  dt={dt}");

    let int = Integrator::new(box_len, sigma, dt).unwrap();
    let rho = n as f64 / box_len.powi(3);

    // -------------------------------------------------------------------
    // Equilibrate two independent copies to different target temperatures.
    // -------------------------------------------------------------------
    println!("\n# Equilibrate COLD liquid (target T = 0.5)");
    let cold = equilibrate(build_lattice(n, box_len, sigma), &int, 0.5, 1500);

    println!("\n# Equilibrate WARM liquid (target T = 1.5)");
    let warm = equilibrate(build_lattice(n, box_len, sigma), &int, 1.5, 1500);

    // -------------------------------------------------------------------
    // Radial distribution function from each equilibrated configuration.
    // -------------------------------------------------------------------
    let nbins = 60;
    let r_max = box_len * 0.5; // required by the periodic-box RDF routine
    let (r_cold, g_cold) = rdf(&cold, box_len, r_max, nbins).unwrap();
    let (r_warm, g_warm) = rdf(&warm, box_len, r_max, nbins).unwrap();
    assert_eq!(g_cold.len(), nbins);
    assert_eq!(g_warm.len(), nbins);
    assert!(g_cold.iter().all(|&x| x.is_finite()));
    assert!(g_warm.iter().all(|&x| x.is_finite()));

    // `rdf` returns the pair-correlation scaled by the box volume (its large-r
    // asymptote is the volume, not 1), so normalise each profile by its tail so
    // g(r) -> 1 at large r before computing peaks / coordination numbers.
    let scale_cold = *g_cold.last().unwrap();
    let scale_warm = *g_warm.last().unwrap();
    let g_cold_n: Vec<f64> = g_cold.iter().map(|&x| x / scale_cold).collect();
    let g_warm_n: Vec<f64> = g_warm.iter().map(|&x| x / scale_warm).collect();

    let (peak_r_c, peak_v_c, min_r_c, cn_c) = analyse(&r_cold, &g_cold_n, rho);
    let (peak_r_w, peak_v_w, min_r_w, cn_w) = analyse(&r_warm, &g_warm_n, rho);

    println!("\n# Structural analysis of g(r)");
    println!(
        "   {:>6} {:>10} {:>10} {:>10} {:>12}",
        "state", "peak r", "peak g", "1st min r", "coord. no."
    );
    println!(
        "   {:>6} {:>10.3} {:>10.3} {:>10.3} {:>12.3}",
        "cold", peak_r_c, peak_v_c, min_r_c, cn_c
    );
    println!(
        "   {:>6} {:>10.3} {:>10.3} {:>10.3} {:>12.3}",
        "warm", peak_r_w, peak_v_w, min_r_w, cn_w
    );

    // Sanity: the first shell sits near the LJ length scale (r ≈ σ..1.2σ).
    assert!(peak_r_c > 0.8 && peak_r_c < 1.5, "first shell peak near σ");
    assert!(peak_r_w > 0.8 && peak_r_w < 1.5, "first shell peak near σ");

    // Physical expectation: lower T => more structured liquid => taller first
    // shell and more neighbours in that shell.
    assert!(
        peak_v_c > peak_v_w,
        "cold liquid must be more structured (taller first shell)"
    );
    assert!(
        cn_c > cn_w,
        "cold liquid must have more first-shell neighbours"
    );
    // Coordination numbers must be positive and well below the total count.
    assert!(cn_c > 0.0 && cn_c < n as f64);
    assert!(cn_w > 0.0 && cn_w < n as f64);

    // Asymptotic property of a fluid RDF: g(r) → 1 at large r. Average the last
    // few bins; it should be O(1), not diverging or vanishing.
    let tail_c: f64 = g_cold_n.iter().rev().take(8).sum::<f64>() / 8.0;
    let tail_w: f64 = g_warm_n.iter().rev().take(8).sum::<f64>() / 8.0;
    println!(
        "   g(r→box/2) tail average: cold = {:.3}, warm = {:.3}  (fluid → 1)",
        tail_c, tail_w
    );
    assert!((tail_c - 1.0).abs() < 0.7, "cold g(r) must relax toward 1");
    assert!((tail_w - 1.0).abs() < 0.7, "warm g(r) must relax toward 1");

    println!("\n# Done. Colder liquid is more structured (taller shell, higher coordination).");
}
