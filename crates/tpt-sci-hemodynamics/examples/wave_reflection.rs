//! # Wave reflection & characteristic impedance (`tpt-sci-hemodynamics`)
//!
//! A second, complementary tour of the crate's 1-D compliant-vessel surface,
//! focused on **wave reflection** rather than the pulse-propagation lag of
//! `arterial_segment`.
//!
//! What this example demonstrates:
//!
//! 1. **Characteristic impedance & the reflection coefficient** — for a 1-D
//!    vessel the characteristic (surge) impedance is `Z0 = ρ·c / A`, where the
//!    wave speed `c` comes from [`Vessel::wave_speed`]. At a junction between a
//!    parent and daughter vessel the reflection coefficient is
//!    `R = (Z_d − Z_p) / (Z_d + Z_p)`. We confirm the textbook limits:
//!    * matched vessels (`Z_d = Z_p`) ⇒ `R = 0` (no reflection);
//!    * a tiny daughter (very high `Z`) ⇒ `R → +1` (closed-end, in-phase,
//!      amplitude-doubling reflection, like a blocked artery);
//!    * a huge daughter (very low `Z`) ⇒ `R → −1` (open-end, anti-phase
//!      reflection).
//! 2. **Dynamic reflection from a closed (reflecting) terminal** — a [`Network`]
//!    chain is driven by a short inlet flow pulse. Because the terminal cell has
//!    no downstream gradient (`dA/dx = 0`), it acts as a closed end that reflects
//!    the wave back. We record the area perturbation at the *inlet* for a *short*
//!    chain (reflected wave returns within the window) versus a *long* chain
//!    (reflection has not yet returned): the short chain's inlet sees a larger
//!    peak deviation, demonstrating the returning reflected wave.
//!
//! Run with: `cargo run --example wave_reflection -p tpt-sci-hemodynamics`
//!
//! Observe: impedance mismatch sets the reflection magnitude and sign, and a
//! closed terminal sends a reflected wave back up the vessel.

use tpt_sci_hemodynamics::{HemodynamicsError, Network, Vessel, tube_law_beta};

const RHO: f64 = 1.06; // blood density (g/cm³)
const A0: f64 = 1.0; // reference cross-sectional area (cm²)
const WALL_H: f64 = 0.1; // wall thickness (cm)

/// Linear-tube-law characteristic impedance `Z0 = ρ·c / A`.
fn char_impedance(area0: f64, beta: f64) -> f64 {
    let v = Vessel::new(area0, 0.0, area0, beta).unwrap();
    RHO * v.wave_speed(RHO) / area0
}

/// Reflection coefficient at a junction `R = (Z_d − Z_p)/(Z_d + Z_p)`.
fn reflection_coefficient(z_parent: f64, z_daughter: f64) -> f64 {
    (z_daughter - z_parent) / (z_daughter + z_parent)
}

/// Build an `n_cells` chain of identical compliant vessels and drive the inlet
/// with a short flow pulse, recording the area at `probe` every step.
fn reflection_run(
    n_cells: usize,
    pulse_flow: f64,
    pulse_steps: usize,
    total_steps: usize,
    probe: usize,
) -> Vec<f64> {
    let beta = tube_law_beta(1.0e4, WALL_H, A0);
    let v0 = Vessel::new(A0, 0.0, A0, beta).unwrap();
    let mut net = Network::new(v0, RHO, 8.0).unwrap();
    for _ in 1..n_cells {
        net.vessels.push(Vessel::new(A0, 0.0, A0, beta).unwrap());
    }

    let mut area_probe = Vec::with_capacity(total_steps);
    for k in 0..total_steps {
        net.vessels[0].flow = if k < pulse_steps { pulse_flow } else { 0.0 };
        net.step(1e-3);
        assert!(net.vessels[probe].area.is_finite());
        area_probe.push(net.vessels[probe].area);
    }
    area_probe
}

/// Peak absolute deviation of the area trace from the reference area `A0`.
fn peak_deviation(area: &[f64]) -> f64 {
    area.iter().map(|&a| (a - A0).abs()).fold(0.0_f64, f64::max)
}

fn main() {
    println!("=== 1-D Hemodynamics: wave reflection ===\n");

    // --- 1. Characteristic impedance & reflection coefficient ----------------
    println!("1. Characteristic impedance Z0 = ρ·c / A  and reflection R");

    // Parent: a stiff, larger conduit (e.g. proximal aorta-like).
    let beta_p = tube_law_beta(5.0e5, WALL_H, A0);
    let z_p = char_impedance(A0, beta_p);
    println!("   parent  (A0={A0}, stiff): Z0 = {z_p:.3}");

    // Matched daughter (identical) -> R = 0.
    let z_match = z_p;
    let r_match = reflection_coefficient(z_p, z_match);
    println!("   matched daughter        : R = {r_match:.3}");

    // Closed-end daughter: tiny area -> very high Z0 -> R -> +1.
    let z_closed = char_impedance(1e-3, beta_p);
    let r_closed = reflection_coefficient(z_p, z_closed);
    println!("   tiny daughter (closed)  : Z0 = {z_closed:.3}, R = {r_closed:.3}");

    // Open-end daughter: huge area -> very low Z0 -> R -> -1.
    let z_open = char_impedance(1e3, beta_p);
    let r_open = reflection_coefficient(z_p, z_open);
    println!("   huge daughter (open)    : Z0 = {z_open:.3}, R = {r_open:.3}");

    assert!(r_match.abs() < 1e-9, "matched junction: no reflection");
    assert!(r_closed > 0.9, "closed end: in-phase reflection R→+1");
    assert!(r_open < -0.9, "open end: anti-phase reflection R→-1");
    assert!(
        r_closed.abs() < 1.0 && r_open.abs() < 1.0,
        "reflection magnitude bounded by 1"
    );

    // --- 2. Dynamic reflection at a closed (reflecting) terminal ------------
    println!("\n2. Reflected wave at a closed terminal (inlet area response)");
    let short = reflection_run(5, 6.0, 200, 30000, 0);
    let long = reflection_run(40, 6.0, 200, 30000, 0);
    let dev_short = peak_deviation(&short);
    let dev_long = peak_deviation(&long);
    println!("   short chain (terminal near) : peak |ΔA| = {dev_short:.4} cm²");
    println!("   long  chain (terminal far)  : peak |ΔA| = {dev_long:.4} cm²");

    // The inlet area must respond to the flow pulse in both chains.
    assert!(
        dev_short > 1e-3,
        "inlet area must respond to the flow pulse (short)"
    );
    assert!(
        dev_long > 1e-3,
        "inlet area must respond to the flow pulse (long)"
    );
    // In the short chain the reflected wave returns to the inlet within the window
    // and augments the incident disturbance; in the long chain the reflection has
    // not yet returned, so the inlet sees only the incident wave. The short-chain
    // deviation must therefore be larger.
    assert!(
        dev_short > dev_long,
        "reflected wave increases the inlet deviation (short > long chain)"
    );

    // Validation surface (public error types).
    println!("\n3. Validation");
    match Vessel::new(-1.0, 0.0, A0, beta_p) {
        Err(HemodynamicsError::InvalidVessel(_)) => println!("   rejected non-positive area ✓"),
        _ => panic!("expected InvalidVessel"),
    }
    match Network::new(Vessel::new(A0, 0.0, A0, beta_p).unwrap(), 0.0, 8.0) {
        Err(HemodynamicsError::InvalidNetwork(_)) => println!("   rejected non-positive ρ ✓"),
        _ => panic!("expected InvalidNetwork"),
    }

    println!("\nWave-reflection tour complete: all checks passed.");
}
