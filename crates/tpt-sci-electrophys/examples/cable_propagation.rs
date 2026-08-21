//! # Cable propagation (`tpt-sci-electrophys`)
//!
//! A second, complementary tour of the crate's public surface, focused on
//! **1-D cable behaviour** rather than the 2-D sheet wave of `ap_wave`.
//!
//! What this example demonstrates:
//!
//! 1. **Single-cell excitability under current injection** — the `HodgkinHuxley`
//!    membrane API integrated with a constant intracellular current clamp
//!    (`dV/dt = −I_ion/Cm + I_inj/Cm`, using the public `v`/`cm` fields). We count
//!    action potentials over a window and recover the classic threshold behaviour:
//!    a sub-threshold bias produces no spikes, a suprathreshold bias drives
//!    repetitive firing (an f–I style curve).
//! 2. **1-D cable propagation** — a `Tissue` with `ny = 1` reduces the 5-point
//!    Laplacian to a genuine 1-D discrete Laplacian, forming a cable. A stimulating
//!    electrode clamps the left node every step (a continuous source), and we
//!    measure the depolarization front's arrival at two downstream probes to obtain
//!    an apparent **conduction velocity**. We then lower the coupling
//!    (`diff` → "ischemic / uncoupled" tissue) and confirm the front travels more
//!    slowly (later arrival), and that the front always moves outward from the
//!    source.
//!
//! Run with: `cargo run --example cable_propagation -p tpt-sci-electrophys`
//!
//! Observe: a current clamp turns a silent HH cell into a repetitive oscillator
//! once it exceeds threshold, and a 1-D cable conducts a depolarization front at a
//! velocity that drops when the tissue is poorly coupled.

use tpt_sci_electrophys::{ElectrophysError, HodgkinHuxley, Tissue};

/// Membrane potential (mV) above which we count a node as "activated".
const ACTIVATION_MV: f64 = 0.0;
/// Assumed inter-node spacing (cm) for the conduction-velocity estimate.
const DX_CM: f64 = 0.02;
/// Integration timestep (s). The Hodgkin–Huxley membrane ODE has a fastest
/// time constant of order 0.1 ms, so an explicit-Euler step must stay well
/// below ~0.025 ms to remain stable (a 0.002 s step spuriously fires).
const DT: f64 = 2.5e-5;

/// Count rising-edge crossings of `thresh` in a voltage trace (one per spike).
fn count_spikes(voltage: &[f64], thresh: f64) -> usize {
    let mut n = 0;
    let mut above = false;
    for &v in voltage {
        if v > thresh && !above {
            n += 1;
            above = true;
        } else if v < thresh {
            above = false;
        }
    }
    n
}

/// Integrate a single HH cell under a constant injected current `i_inj`
/// (µA/cm²) and return the voltage trace + spike count.
fn current_clamp(i_inj: f64, steps: usize) -> (Vec<f64>, usize) {
    let mut hh = HodgkinHuxley::resting();
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        // Public-field current clamp: dV/dt += I_inj / Cm.
        hh.v += i_inj * DT / hh.cm;
        hh.step(DT);
        assert!(hh.voltage().is_finite(), "membrane potential stays finite");
        trace.push(hh.voltage());
    }
    let spikes = count_spikes(&trace, ACTIVATION_MV);
    (trace, spikes)
}

/// Run a 1-D cable (`ny = 1`) driven by a clamped source node at `x = 0` and
/// report the first arrival time (s) of the depolarization front at each probe
/// index. Returns one `Option<f64>` per probe (the first time `vm[probe] > 0`).
fn cable_arrivals(
    nx: usize,
    diff: f64,
    source_v: f64,
    probes: &[usize],
    steps: usize,
) -> Vec<Option<f64>> {
    let mut t = Tissue::new(nx, 1, diff).unwrap();
    let mut arrivals = vec![None; probes.len()];
    for k in 0..steps {
        t.stimulate(0, 0, source_v); // sustained electrode each step
        t.step(DT);
        for (pi, &p) in probes.iter().enumerate() {
            if arrivals[pi].is_none() && t.vm[p] > ACTIVATION_MV {
                arrivals[pi] = Some(k as f64 * DT);
            }
        }
    }
    arrivals
}

fn main() {
    demo_current_clamp();
    demo_cable();
}

/// Section 1: single-cell excitability vs injected current (threshold / f–I).
fn demo_current_clamp() {
    println!("=== Single Hodgkin–Huxley cell: current clamp ===");

    let steps = 4000usize;
    let (silent_v, silent_n) = current_clamp(0.0, steps);
    let (firing_v, firing_n) = current_clamp(10.0, steps);

    let v_rest = silent_v[0];
    println!(
        "sub-threshold bias I=0     : spikes = {silent_n}, V end = {:>7.2} mV",
        silent_v[steps - 1]
    );
    println!(
        "supra-threshold bias I=10  : spikes = {firing_n}, V end = {:>7.2} mV",
        firing_v[steps - 1]
    );
    assert!(silent_n == 0, "no current -> no action potentials");
    assert!(firing_n >= 1, "supra-threshold current -> repetitive firing");
    assert!(firing_v[steps - 1].is_finite());
    assert!(v_rest.is_finite());

    println!(
        "threshold confirmed: 0 µA/cm² is silent, 10 µA/cm² drives {} spikes\n",
        firing_n
    );
}

/// Section 2: 1-D cable depolarization front + conduction velocity, with and
/// without reduced coupling ("ischemia").
fn demo_cable() {
    println!("=== 1-D cable (Tissue ny = 1): conduction ===");

    // Invalid configuration exercises the ElectrophysError surface.
    let err = Tissue::new(0, 1, 0.5).unwrap_err();
    assert_eq!(err, ElectrophysError::InvalidTissue("dims must be > 0".into()));
    println!("rejected zero-size tissue: {err}");

    let nx = 60usize;
    let source_v = 80.0;
    let steps = 20000usize;

    // Normal coupling: measure arrival at two probes -> conduction velocity.
    let normal = cable_arrivals(nx, 30.0, source_v, &[4, 10], steps);
    let (t_up, t_down) = match (normal[0], normal[1]) {
        (Some(u), Some(d)) => (u, d),
        _ => panic!("front must reach both probes in the normal cable"),
    };
    let velocity = ((10 - 4) as f64 * DX_CM) / (t_down - t_up);
    println!(
        "normal coupling  : upstream t = {:.1} ms, downstream t = {:.1} ms, \
         velocity = {:.2} cm/s",
        t_up * 1000.0,
        t_down * 1000.0,
        velocity
    );
    assert!(t_down > t_up, "front travels outward (downstream after upstream)");
    assert!(velocity > 0.0, "conduction velocity positive");
    assert!(velocity.is_finite());

    // Reduced coupling ("ischemic"): the front arrives later at the same probe.
    let ischemic = cable_arrivals(nx, 12.0, source_v, &[4], steps);
    let t_isc = ischemic[0].expect("ischemic front must still reach probe 4");
    println!(
        "ischemic coupling: probe-4 arrival t = {:.1} ms (slower than normal)",
        t_isc * 1000.0
    );
    assert!(t_isc > t_up, "reduced coupling slows the front");

    println!("\n1-D cable propagation tour complete.");
}
