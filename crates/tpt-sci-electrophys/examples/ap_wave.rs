//! # Electrophysiology tour (`tpt-sci-electrophys`)
//!
//! A guided demo of the crate's public surface, built on [`tpt_sci_ode`] (membrane
//! kinetics) and [`tpt_sci_grid`] (the diffusion operator behind `Tissue`).
//!
//! What to observe:
//! 1. **Single Hodgkin–Huxley cell** — integrate the membrane ODE directly, apply
//!    a depolarizing stimulus, and watch a fully-formed action potential: a sharp
//!    spike followed by repolarization. We report the resting potential, the peak
//!    voltage, and the resulting spike amplitude.
//! 2. **2-D tissue sheet (`Tissue`)** — a vertical strip on the left edge is
//!    stimulated; the monodomain diffusion (`dVm/dt = −I_ion/Cm + D·∇²Vm`) carries
//!    the resulting depolarization outward. We measure *when* an upstream node and
//!    a farther downstream node are activated, and convert that into an apparent
//!    conduction velocity for the depolarization front.
//!
//! Run with: `cargo run --example ap_wave -p tpt-sci-electrophys`

use tpt_sci_electrophys::{ElectrophysError, HodgkinHuxley, Tissue};

/// Threshold used to detect "activation" (depolarization above resting level).
const ACTIVATION_MV: f64 = 0.0;
/// Assumed inter-node spacing (cm) for the conduction-velocity estimate.
const DX_CM: f64 = 0.02;
/// Integration timestep (s). Kept small so the explicit 5-point Laplacian
/// diffusion stays stable (`diff * dt / dx² < 0.25`).
const DT: f64 = 0.002;

fn main() {
    demo_single_cell();
    demo_tissue_wave();
}

/// Exercise the `HodgkinHuxley` membrane API directly.
fn demo_single_cell() {
    println!("=== Single Hodgkin–Huxley cell ===");

    let mut hh = HodgkinHuxley::resting();
    assert_eq!(hh.state().len(), 4, "state vector is [V, m, h, n]");

    let resting_v = hh.voltage();
    let rest_current = hh.ionic_current();
    println!(
        "resting V = {resting_v:>7.2} mV  I_ion = {rest_current:>8.3} µA/cm²  \
         gating m={:.3} h={:.3} n={:.3}",
        hh.m, hh.h, hh.n
    );

    // Depolarize above threshold to trigger an action potential.
    hh.v = 20.0;

    // Integrate the membrane ODE and track the peak of the spike.
    let steps = 600usize;
    let mut peak_v = hh.voltage();
    let mut peak_step = 0usize;
    for k in 0..steps {
        hh.step(DT);
        assert!(hh.voltage().is_finite(), "membrane potential stays finite");
        if hh.voltage() > peak_v {
            peak_v = hh.voltage();
            peak_step = k;
        }
    }

    let spike_amp = peak_v - resting_v;
    println!(
        "peak V    = {peak_v:>7.2} mV at t = {:>6.2} ms  (spike amplitude = {spike_amp:>6.2} mV)",
        peak_step as f64 * DT * 1000.0
    );
    println!(
        "after  {:.0} ms: V = {:>7.2} mV  m={:.3} h={:.3} n={:.3} (recovery)",
        steps as f64 * DT * 1000.0,
        hh.voltage(),
        hh.m,
        hh.h,
        hh.n
    );

    // A genuine action potential must swing well above the resting level.
    assert!(spike_amp > 0.0, "stimulus produced a depolarizing spike");
    assert!(peak_v > 20.0, "the spike overshot the stimulus");

    println!();
}

/// Exercise the `Tissue` monodomain sheet: stimulus application, the diffusion
/// of the resulting depolarization, activation-time ordering, and an apparent
/// conduction velocity.
///
/// Note: in this v1 model `Tissue::stimulate` sets the node potential `Vm`
/// directly, and the per-node membrane current is evaluated from the coupled
/// `HodgkinHuxley` cell. The depolarization therefore spreads outward as a
/// diffusing front (strongly attenuated with distance) rather than a fully
/// regenerated action-potential train. We measure the front's arrival at two
/// nodes along the strip and report the apparent front velocity.
fn demo_tissue_wave() {
    println!("=== 2-D tissue sheet (monodomain) ===");

    // Invalid configuration exercises the `ElectrophysError` surface.
    let err = Tissue::new(0, 0, 0.8).unwrap_err();
    assert_eq!(
        err,
        ElectrophysError::InvalidTissue("dims must be > 0".into())
    );
    println!("rejected zero-size tissue: {err}");

    let mut t = Tissue::new(16, 16, 100.0).unwrap();
    let j = t.ny / 2;

    // Stimulate a vertical strip on the left edge to launch a depolarization.
    for jj in 0..t.ny {
        t.stimulate(0, jj, 80.0);
    }

    // Sample an upstream node (near the stimulus) and a downstream node.
    let upstream = t.idx(2, j);
    let downstream = t.idx(5, j);

    let steps = 400usize;
    let mut upstream_t: Option<usize> = None;
    let mut downstream_t: Option<usize> = None;
    for k in 0..steps {
        t.step(DT);
        if upstream_t.is_none() && t.vm[upstream] > ACTIVATION_MV {
            upstream_t = Some(k);
        }
        if downstream_t.is_none() && t.vm[downstream] > ACTIVATION_MV {
            downstream_t = Some(k);
        }
    }

    println!("max Vm reached across sheet = {:.2} mV", t.max_voltage());

    let (u, d) = match (upstream_t, downstream_t) {
        (Some(u), Some(d)) => (u, d),
        _ => panic!("expected the depolarization front to reach both sampled nodes"),
    };

    let t_up = u as f64 * DT;
    let t_down = d as f64 * DT;
    let distance = (5 - 2) as f64 * DX_CM;
    let velocity = distance / (t_down - t_up);

    println!("upstream node activated   at t = {:.1} ms", t_up * 1000.0);
    println!("downstream node activated at t = {:.1} ms", t_down * 1000.0);
    println!("apparent conduction velocity = {velocity:.2} cm/s");

    assert!(t.max_voltage().is_finite(), "tissue field stays finite");
    assert!(t_down > t_up, "downstream activates after upstream");
    assert!(
        velocity > 0.0,
        "conduction velocity is positive (front travels outward)"
    );

    println!();
    println!("Electrophysiology tour complete.");
}
