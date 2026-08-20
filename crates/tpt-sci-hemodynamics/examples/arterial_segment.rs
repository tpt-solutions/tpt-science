//! # 1-D compliant-vessel hemodynamics tour
//!
//! A guided demonstration of the `tpt-sci-hemodynamics` public surface: the
//! compliant [`Vessel`], the linear tube-law stiffness [`tube_law_beta`], the
//! Womersley pulsatile profile [`womersley_velocity`], the Casson
//! shear-thinning [`casson_viscosity`], and the method-of-lines [`Network`]
//! (`rhs` / `step`).
//!
//! Run with:
//! ```text
//! cargo run --example arterial_segment -p tpt-sci-hemodynamics
//! ```
//!
//! What to observe:
//! * Stiffer walls (larger `β`) raise the pulse-wave velocity `c` and shrink
//!   the area swing for the same inlet drive.
//! * The Womersley profile is parabolic near the wall at low Womersley number
//!   and flattens toward plug flow as `α` grows.
//! * The Casson correction raises viscosity at low shear rate (yield stress).
//! * A wave driven into the inlet takes time to reach the outlet — the outlet
//!   flow waveform **lags** the inlet flow waveform.

use tpt_sci_hemodynamics::{
    HemodynamicsError, Network, Vessel, casson_viscosity, tube_law_beta, womersley_velocity,
};

const RHO: f64 = 1.06; // blood density (g/cm³)
const A0: f64 = 1.0; // reference cross-sectional area (cm²)
const WALL_H: f64 = 0.1; // wall thickness (cm)

#[derive(Clone)]
struct Waveform {
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Waveform {
    /// Time (s) of the maximum value (peak of the waveform).
    fn peak_time(&self) -> f64 {
        let idx = self
            .values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        self.times[idx]
    }

    /// Peak-to-peak swing of the waveform.
    fn swing(&self) -> f64 {
        let max = self.values.iter().cloned().fold(f64::MIN, f64::max);
        let min = self.values.iter().cloned().fold(f64::MAX, f64::min);
        max - min
    }
}

/// Build an `n`-cell compliant arterial chain (a network of identical vessels).
fn build_chain(beta: f64, n: usize, friction: f64) -> Network {
    let v0 = Vessel::new(A0, 0.0, A0, beta).unwrap();
    let mut net = Network::new(v0, RHO, friction).unwrap();
    for _ in 1..n {
        net.vessels.push(Vessel::new(A0, 0.0, A0, beta).unwrap());
    }
    net
}

/// Drive the inlet with a Womersley-modulated sinusoidal flow over one cardiac
/// cycle and record the inlet/outlet area and flow responses.
fn simulate(beta: f64, friction: f64, cells: usize) -> (Waveform, Waveform, Waveform, Waveform) {
    let mut net = build_chain(beta, cells, friction);
    let dt = 1e-3;
    let cardiac = 2.0 * std::f64::consts::PI; // 1 Hz heart
    let steps = 1000; // one cardiac cycle
    let u0 = womersley_velocity(0.0, 1.0, cardiac, 0.04);

    let mut inlet_flow = Vec::with_capacity(steps);
    let mut outlet_flow = Vec::with_capacity(steps);
    let mut inlet_area = Vec::with_capacity(steps);
    let mut outlet_area = Vec::with_capacity(steps);
    let mut times = Vec::with_capacity(steps);

    for k in 0..steps {
        let t = k as f64 * dt;
        // Pulsatile inlet drive via the Womersley centreline amplitude.
        net.vessels[0].flow = u0 * (0.5 + 0.5 * (cardiac * t).sin());
        net.step(dt);

        times.push(t);
        inlet_flow.push(net.vessels[0].flow);
        outlet_flow.push(net.vessels[cells - 1].flow);
        inlet_area.push(net.vessels[0].area);
        outlet_area.push(net.vessels[cells - 1].area);
    }

    (
        Waveform {
            times: times.clone(),
            values: inlet_flow,
        },
        Waveform {
            times: times.clone(),
            values: outlet_flow,
        },
        Waveform {
            times: times.clone(),
            values: inlet_area,
        },
        Waveform {
            times,
            values: outlet_area,
        },
    )
}

fn main() {
    println!("=== 1-D Hemodynamics Tour ===\n");

    // ------------------------------------------------------------------
    // 1. Tube-law stiffness β from wall mechanics, and Vessel diagnostics.
    // ------------------------------------------------------------------
    let beta_soft = tube_law_beta(1.0e4, WALL_H, A0);
    let beta_stiff = tube_law_beta(1.0e6, WALL_H, A0);
    println!("1. Tube-law stiffness");
    println!("   β (soft E=1e4)  = {beta_soft:.3}");
    println!("   β (stiff E=1e6) = {beta_stiff:.3}");

    let v = Vessel::new(A0, 0.0, A0, beta_soft).unwrap();
    // At the reference area the transmural pressure is zero.
    assert!(v.pressure().abs() < 1e-9);
    let c_soft = v.wave_speed(RHO);
    let c_val = Vessel::new(A0, 0.0, A0, beta_stiff)
        .unwrap()
        .wave_speed(RHO);
    println!("   wave speed c (soft)  = {c_soft:.1} cm/s, c (stiff) = {c_val:.1} cm/s\n");
    assert!(c_val > c_soft, "stiffer wall must raise wave speed");

    // ------------------------------------------------------------------
    // 2. Womersley pulsatile profile: parabolic at low α, plug at high α.
    // ------------------------------------------------------------------
    let r0 = 1.0;
    let omega = 2.0 * std::f64::consts::PI; // 1 Hz
    println!("2. Womersley profile (centreline r=0 vs wall r=r0)");
    for alpha_disp in ["low α (ν large)", "high α (ν small)"] {
        let (nu_a, label) = match alpha_disp {
            "low α (ν large)" => (0.4, "low"),
            _ => (0.004, "high"),
        };
        let centre = womersley_velocity(0.0, r0, omega, nu_a);
        let wall = womersley_velocity(r0, r0, omega, nu_a);
        println!("   {label}: u(centre)={centre:.3}, u(wall)={wall:.3}");
        assert!(centre.is_finite() && wall.is_finite());
        if label == "low" {
            assert!(centre > 1.0, "low α profile should exceed mean (>1)");
        } else {
            assert!(centre <= 1.5, "high α should flatten toward plug");
        }
    }
    println!();

    // ------------------------------------------------------------------
    // 3. Casson non-Newtonian (shear-thinning) viscosity.
    // ------------------------------------------------------------------
    let mu_inf = 0.04;
    println!("3. Casson viscosity at γ̇ = 100 s⁻¹");
    for tau_y in [0.0, 0.1, 0.5] {
        let mu = casson_viscosity(mu_inf, tau_y, 100.0);
        println!("   τy={tau_y}: μ = {mu:.4} (μ∞={mu_inf})");
        assert!(mu >= mu_inf);
    }
    // Zero shear rate returns the asymptotic viscosity (fully yielded limit).
    assert!((casson_viscosity(mu_inf, 0.5, 0.0) - mu_inf).abs() < 1e-12);
    println!();

    // ------------------------------------------------------------------
    // 4. Error handling: invalid vessel construction.
    // ------------------------------------------------------------------
    println!("4. Validation");
    match Vessel::new(-1.0, 0.0, A0, beta_soft) {
        Err(HemodynamicsError::InvalidVessel(_)) => println!("   rejected non-positive area ✓"),
        _ => panic!("expected InvalidVessel"),
    }

    // ------------------------------------------------------------------
    // 5. Network: inspect the RHS, then step a full cardiac cycle.
    // ------------------------------------------------------------------
    let peek = build_chain(beta_soft, 2, 8.0);
    let mut out = [0.0_f64; 2];
    peek.rhs(0, &mut out);
    println!(
        "\n5. Network RHS at cell 0: dA/dt={:.4}, dQ/dt={:.4} (cell count = {})\n",
        out[0],
        out[1],
        peek.vessels.len()
    );

    // Soft vs stiff segment: compare area swing and wave-propagation lag.
    let cells = 12usize;
    let (in_s, out_s, ina_s, outa_s) = simulate(beta_soft, 8.0, cells);
    let (in_f, out_f, ina_f, outa_f) = simulate(beta_stiff, 8.0, cells);

    println!("6. Waveforms over one cardiac cycle");
    println!(
        "   soft  : inlet Q peak t={:.3}s, outlet Q peak t={:.3}s, area swing={:.4} cm²",
        in_s.peak_time(),
        out_s.peak_time(),
        ina_s.swing()
    );
    println!(
        "   stiff : inlet Q peak t={:.3}s, outlet Q peak t={:.3}s, area swing={:.4} cm²",
        in_f.peak_time(),
        out_f.peak_time(),
        ina_f.swing()
    );

    // Pressure swing at the inlet scales with β via the linear tube law
    // p = β·(√A − √A0); the same area waveform therefore yields a larger
    // pressure swing for the stiffer wall.
    let p_swing = |beta: f64, area: &Waveform| -> f64 {
        let vals: Vec<f64> = area
            .values
            .iter()
            .map(|&a| beta * (a.sqrt() - A0.sqrt()))
            .collect();
        let max = vals.iter().cloned().fold(f64::MIN, f64::max);
        let min = vals.iter().cloned().fold(f64::MAX, f64::min);
        max - min
    };
    let ps_swing = p_swing(beta_soft, &ina_s);
    let pf_swing = p_swing(beta_stiff, &ina_f);
    println!("   pressure swing: soft={ps_swing:.4} dyn/cm², stiff={pf_swing:.4} dyn/cm²\n");

    // Robust invariants across both stiffnesses.
    let lag_s = out_s.peak_time() - in_s.peak_time();
    let lag_f = out_f.peak_time() - in_f.peak_time();
    println!("   propagation lag: soft={lag_s:.3}s, stiff={lag_f:.3}s (outlet lags inlet)\n");
    for w in [&ina_s, &outa_s, &ina_f, &outa_f] {
        for &x in &w.values {
            assert!(
                x.is_finite() && x > 0.0,
                "area must stay finite and positive"
            );
        }
    }
    assert!(lag_s > 0.0, "outlet must lag inlet (soft)");
    assert!(lag_f > 0.0, "outlet must lag inlet (stiff)");
    assert!(
        pf_swing > ps_swing,
        "stiffer wall (larger β) must give larger pressure swing"
    );

    println!("Arterial segment tour complete: all checks passed.");
}
