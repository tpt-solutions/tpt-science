//! Hemodynamics demo: a compliant arterial segment advanced under a Womersley
//! pulsatile inlet, reporting area / flow / pressure over a cardiac cycle.
//!
//! Run with: `cargo run --example arterial_segment -p tpt-sci-hemodynamics`

use tpt_sci_hemodynamics::{Network, Vessel, tube_law_beta, womersley_velocity};

fn main() {
    let beta = tube_law_beta(1.0e5, 0.1, 1.0); // aortic-scale stiffness
    let v = Vessel::new(1.0, 0.0, 1.0, beta).unwrap();
    let mut net = Network::new(v, 1.06, 8.0).unwrap();

    let dt = 1e-3;
    let cardiac = 2.0 * std::f64::consts::PI / 1.0; // 1 Hz heart
    let n = 1000;
    for k in 0..n {
        let t = k as f64 * dt;
        // Pulsatile inlet flow via Womersley profile (peak amplitude 1).
        let u = womersley_velocity(0.0, 1.0, cardiac, 0.04);
        net.vessels[0].flow = u * (0.5 + 0.5 * (cardiac * t).sin());
        net.step(dt);
        if k % 200 == 0 {
            let v = &net.vessels[0];
            println!(
                "t={t:.2}: A={:.4} Q={:.4} p={:.4}",
                v.area,
                v.flow,
                v.pressure()
            );
        }
    }
    println!("Arterial segment completed one cardiac cycle.");
}
