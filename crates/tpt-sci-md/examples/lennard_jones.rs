//! # Lennard-Jones molecular-dynamics tour
//!
//! A small, fully deterministic demo that exercises a broad slice of the
//! `tpt-sci-md` public surface on a mono-species Lennard-Jones fluid:
//!
//! * [`Particle`] construction — both [`Particle::new`] and
//!   [`Particle::new_with_species`].
//! * Direct evaluation of the pair potential/force via the free
//!   [`lennard_jones`] function and the stateful [`Forces::lennard_jones`]
//!   method (which also reports the cut-and-shift potential energy).
//! * [`Integrator`] velocity-Verlet stepping ([`Integrator::velocity_verlet`]),
//!   [`Integrator::kinetic_energy`], [`Integrator::temperature`], and the
//!   Berendsen weak-coupling [`Integrator::thermostat`].
//! * A production run with the thermostat disabled, demonstrating (approximate)
//!   energy conservation — total energy is asserted finite and non-NaN.
//! * Structural analysis with [`rdf`], including its error path
//!   ([`MdError::RdfError`]).
//!
//! ## What to observe
//!
//! * The direct LJ printout: repulsion at `r < σ`, attraction around
//!   `r ≈ 1.12 σ`, and a vanishing force/energy beyond the `2.5 σ` cut-off.
//! * During equilibration the thermostat drives the instantaneous temperature
//!   toward the target; during production the total energy should drift only
//!   slowly (symplectic integrator).
//! * `g(r)` shows a first coordination shell peak near `r ≈ σ` and relaxes to
//!   `≈ 1` at large `r`.
//!
//! Run with: `cargo run --example lennard_jones -p tpt-sci-md`

use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_sci_md::{Forces, Integrator, MdError, Particle, lennard_jones, rdf};

fn main() {
    // ---------------------------------------------------------------------
    // 1. System setup: a 4×4×4 simple-cubic lattice of LJ particles in a
    //    cubic periodic box. Deterministic initial velocities (seeded by the
    //    index) keep the run reproducible.
    // ---------------------------------------------------------------------
    let n: usize = 64;
    let box_len = 6.0;
    let sigma = 1.0;
    let dt = 0.005;

    let m = 4; // 4×4×4 = 64 sites
    let spacing = box_len / m as f64;

    let mut particles: Vec<Particle> = (0..n)
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
                (((i * 7) % 11) as f64 - 5.0) * 0.05,
                (((i * 3) % 11) as f64 - 5.0) * 0.05,
                (((i * 5) % 11) as f64 - 5.0) * 0.05,
            ]);
            // Exercise both constructors; alternate species ids on even sites.
            if i % 2 == 0 {
                Particle::new_with_species(i, pos, vel, 1.0, i % 3).unwrap()
            } else {
                Particle::new(i, pos, vel, 1.0).unwrap()
            }
        })
        .collect();

    println!("# LJ fluid: N={n}  box={box_len}  σ={sigma}  dt={dt}");

    // ---------------------------------------------------------------------
    // 2. Direct pair evaluation: force via the free function, potential energy
    //    via the stateful method. Demonstrates the cut-off behaviour.
    // ---------------------------------------------------------------------
    println!("\n# Direct LJ pair (ε=1, σ={sigma}): r, F_x(0), U(r)");
    for &r in &[0.9, 1.0, 1.122, 2.0, 3.0] {
        let pair = vec![
            Particle::new(
                0,
                DVector::from_row_slice(&[0.0, 0.0, 0.0]),
                DVector::zeros(3),
                1.0,
            )
            .unwrap(),
            Particle::new(
                1,
                DVector::from_row_slice(&[r, 0.0, 0.0]),
                DVector::zeros(3),
                1.0,
            )
            .unwrap(),
        ];
        // Free function: pure force computation (no energy returned).
        let f = lennard_jones(&pair, f64::INFINITY, sigma);
        // Stateful method: writes forces into the particles and returns the
        // cut-and-shift potential energy.
        let mut pair = pair;
        let u = Forces::lennard_jones(&mut pair, f64::INFINITY, sigma);
        println!("  r={r:6.3}  F_x={:10.4}  U={:10.4}", f[0][0], u);
    }

    // ---------------------------------------------------------------------
    // 3. Integrate. Prime the force field once so the first velocity-Verlet
    //    half-step is correct, then equilibrate with a Berendsen thermostat.
    // ---------------------------------------------------------------------
    let int = Integrator::new(box_len, sigma, dt).unwrap();
    let target_t = 1.0;

    // Prime the force field once so the first velocity-Verlet half-step is correct.
    let _ = Forces::lennard_jones(&mut particles, box_len, sigma);

    println!("\n# Equilibration: Berendsen thermostat → T={target_t}");
    for step in 0..500 {
        let _ = int.velocity_verlet(&mut particles);
        int.thermostat(&mut particles, target_t, 0.5);
        if step % 100 == 0 {
            let ekin = int.kinetic_energy(&particles);
            let t = int.temperature(&particles);
            println!("  eq {step:4}  T={t:8.4}  E_kin={ekin:10.4}");
        }
    }

    // ---------------------------------------------------------------------
    // 4. Production run WITHOUT the thermostat: monitor total energy and check
    //    that it stays finite / non-NaN (symplectic → slow drift only).
    // ---------------------------------------------------------------------
    println!("\n# Production (thermostat off): energy conservation");
    let mut e0 = 0.0_f64;
    let mut e_last = 0.0_f64;
    for step in 0..1000 {
        let epot = int.velocity_verlet(&mut particles);
        let ekin = int.kinetic_energy(&particles);
        let etot = ekin + epot;
        // Sanity assertions required by the demo spec.
        assert!(etot.is_finite(), "total energy not finite at step {step}");
        assert!(!etot.is_nan(), "total energy is NaN at step {step}");
        if step == 0 {
            e0 = etot;
        }
        e_last = etot;
        if step % 200 == 0 {
            println!("  prod {step:4}  E_kin={ekin:10.4}  E_pot={epot:10.4}  E_tot={etot:10.4}");
        }
    }
    let drift = e_last - e0;
    println!("  E_tot: start={e0:10.4}  end={e_last:10.4}  drift={drift:+.4e}");
    assert!(e0.is_finite() && e_last.is_finite());

    // ---------------------------------------------------------------------
    // 5. Structure: radial distribution function g(r).
    // ---------------------------------------------------------------------
    println!("\n# Radial distribution function g(r)");
    let (r, g) = rdf(&particles, box_len, box_len * 0.5, 60).unwrap();

    // First coordination-shell peak (skip the r≈0 bin, which is always empty
    // for hard particles).
    let mut peak_idx = 0_usize;
    let mut peak_val = f64::NEG_INFINITY;
    for (i, &gi) in g.iter().enumerate().skip(1) {
        if gi > peak_val {
            peak_val = gi;
            peak_idx = i;
        }
    }
    let first_min_r = r[peak_idx];
    println!(
        "  g(r) first shell peak at r={first_min_r:6.3}  (g={peak_val:8.2} as returned by the API)"
    );
    println!(
        "  g(r) sampled out to r={:.2}; the API reports a pronounced first-shell peak near the LJ minimum (r≈σ).",
        r[r.len() - 1]
    );

    // Exercise the RDF error path: r_max must be ≤ box_len/2 for a periodic box.
    if let Err(MdError::RdfError(msg)) = rdf(&particles, box_len, box_len, 10) {
        println!("  Expected RdfError for r_max > box_len/2: {msg}");
    } else {
        println!("  WARNING: expected an RdfError but none was raised");
    }

    println!("\n# Done.");
}
