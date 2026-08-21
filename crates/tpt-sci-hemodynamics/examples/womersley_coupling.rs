//! # Exact Womersley profile + 0-D/1-D Windkessel coupling
//!
//! Demonstrates the two v1 features closed out for `tpt-sci-hemodynamics`:
//!
//! 1. The exact complex-Bessel Womersley axial-velocity profile
//!    (`womersley_velocity_profile`) and the analytic flow-rate formula.
//! 2. A 1-D [`Network`] outlet coupled to a lumped 3-element [`Windkessel`]
//!    (RCR) terminal load via [`couple`].
//!
//! Run with:
//! ```text
//! cargo run --example womersley_coupling -p tpt-sci-hemodynamics
//! ```

use tpt_sci_hemodynamics::{
    CfdCoupling, Network, Vessel, Windkessel, couple, tube_law_beta, womersley_complex_velocity,
    womersley_flow_rate_analytic,
};

fn main() {
    println!("=== Exact Womersley profile + Windkessel coupling ===\n");

    // ------------------------------------------------------------------
    // 1. Exact Womersley velocity profile vs. the analytic flow rate.
    // ------------------------------------------------------------------
    let r0 = 1.0;
    let omega = 2.0 * std::f64::consts::PI; // 1 Hz
    let nu = 0.04;
    let rho = 1.06;
    let alpha = r0 * (omega / nu).sqrt();

    println!("1. Womersley velocity amplitude |u(r)| (r in [0, R])");
    for frac in [0.0_f64, 0.5, 0.9] {
        let r = frac * r0;
        let amp = womersley_complex_velocity(r, r0, alpha, omega, rho).norm();
        println!("   r/R={frac:.1}: |u|={amp:.4}");
    }

    let q = womersley_flow_rate_analytic(alpha, omega, rho, r0);
    println!(
        "   flow-rate amplitude |Q̃| = {:.4} (analytic)\n",
        (q.re * q.re + q.im * q.im).sqrt()
    );

    // ------------------------------------------------------------------
    // 2. 1-D network outlet coupled to a Windkessel (0-D) terminal load.
    // ------------------------------------------------------------------
    let beta = tube_law_beta(1.0e5, 0.1, 1.0);
    let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
    let mut net = Network::new(v, rho, 8.0).unwrap();
    let mut wk = Windkessel::new(1.0, 10.0, 0.1, 0.0, 80.0).unwrap();

    let dt = 1e-3;
    let mut max_p = 0.0_f64;
    for k in 0..2000 {
        let t = k as f64 * dt;
        net.vessels[0].flow = 1.0 + 0.8 * (omega * t).sin();
        let p = couple(&mut net, &mut wk, dt);
        max_p = max_p.max(p.abs());
        assert!(p.is_finite(), "coupled outlet pressure must stay finite");
    }
    println!("2. Coupled outlet pressure");
    println!(
        "   bounded max |p_outlet| = {max_p:.3} (Windkessel τ = {:.3} s)\n",
        wk.time_constant()
    );

    // The Windkessel also satisfies the thin CfdCoupling interface a 3-D
    // tpt-sci-cfd-core domain would implement.
    let q_back = wk.couple_step(dt, max_p);
    println!("3. CfdCoupling boundary step returned flow q = {q_back:.4}\n");

    println!("Womersley + coupling demo complete: all checks passed.");
}
