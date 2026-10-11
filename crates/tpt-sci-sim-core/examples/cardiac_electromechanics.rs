//! Coupled scenario: a human ventricular myocyte (Ten Tusscher–Panfilov 2004,
//! epicardial) driving the Land et al. active-tension model through its calcium
//! transient, with length and velocity dependence.
//!
//! Two `SubModel`s run under the orchestrator:
//!
//! * `electrical` is the ionic cell. It is stimulated once at t = 0 and advances
//!   its gates, concentrations and voltage with a fixed explicit step.
//! * `mechanics` is the Land et al. (2017) active-tension model. It is driven by
//!   the cytosolic Ca²⁺ from the cell, and by a prescribed length protocol.
//!
//! The length protocol is the authors' constant-velocity shortening, from the
//! release's `generate_figures.m` (`flen(1)·(1 − (0.1/te)·t)`, with te in
//! 100–300 ms): the sarcomere holds its resting length until `SHORTEN_START` ms,
//! then shortens by 10 % at a constant speed over `SHORTEN_DURATION` ms (te = 200),
//! and holds. The shortening starts during the active phase, so it acts on the
//! twitch.
//!
//! Two limits, both from the release. First, it has no mechanics model that sets
//! length from a load, and its recorded `Length` channel is not a usable sarcomere
//! length (it flips between about +1350 and −1310 µm), so the nominal protocol is
//! the length input. Second, its constant-shortening fits use the skinned parameter
//! set, while this example runs the intact model. Applying the protocol to intact
//! tissue goes beyond the authors' validation.
//!
//! The example runs the same cell twice, isometric and shortening, and checks the
//! force–velocity relation on the coupled system.
//!
//! The coupling is one-way (`Ca²⁺ → tension`), which is the excitation–contraction
//! link. Both models are published, and their parameters come from the sources
//! they cite: see `tpt_sci_electrophys::LandActiveTension` and
//! `tpt_sci_electrophys::TenTusscher`.
//!
//! Run with: `cargo run --release --example cardiac_electromechanics -p tpt-sci-sim-core`

use tpt_sci_electrophys::{IonicModel, LandActiveTension, TenTusscher};
use tpt_sci_sim_core::{Coupling, SimError, Simulation, SubModel};

/// Index of the cytosolic Ca²⁺ in the Ten Tusscher state vector (mM).
const CA_INDEX: usize = 15;
/// Index of the transmembrane potential in the Ten Tusscher state vector.
const V_INDEX: usize = 0;
/// Cell-model step (ms). The cell advances in sub-steps of at most this size.
const CELL_STEP: f64 = 0.01;
/// Stimulus: a 1 ms pulse of −52 pA/pF at t = 0 (CellML `stim_amplitude`).
const STIM_AMPLITUDE: f64 = -52.0;
const STIM_DURATION: f64 = 1.0;
/// Simulated time (ms) and orchestrator step (ms).
const DURATION: f64 = 600.0;
const DT: f64 = 0.5;
/// Contraction sub-step (ms) for the Runge–Kutta integrator.
const CONTRACTION_STEP: f64 = 0.01;
/// Shortening protocol (ms): start time, duration, and the fractional shortening
/// (10 % of resting length, as in `generate_figures.m`).
const SHORTEN_START: f64 = 30.0;
const SHORTEN_DURATION: f64 = 200.0;
const SHORTEN_FRACTION: f64 = 0.1;

/// The Ten Tusscher cell as an orchestrated sub-model.
struct CellModel {
    id: String,
    cell: TenTusscher,
    time: f64,
    /// The full state, exposed to couplings through `state()`.
    state: Vec<f64>,
}

impl CellModel {
    fn new(id: &str) -> Self {
        let cell = TenTusscher::resting();
        let state = cell.state();
        Self {
            id: id.to_string(),
            cell,
            time: 0.0,
            state,
        }
    }

    /// One explicit step of the cell model, with the stimulus applied during the
    /// first millisecond.
    fn step(&mut self, h: f64) {
        let stim = if self.time < STIM_DURATION {
            STIM_AMPLITUDE
        } else {
            0.0
        };
        let v = self.cell.voltage();
        self.cell.advance_reaction(v, h);
        let dv = -(self.cell.ionic_current() + stim) / self.cell.capacitance();
        self.cell.v = v + h * dv;
        self.time += h;
    }
}

impl SubModel for CellModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn max_step(&self) -> f64 {
        CELL_STEP
    }

    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        // Split `dt` into equal sub-steps no larger than `CELL_STEP`.
        let n = (dt / CELL_STEP).ceil().max(1.0) as usize;
        let h = dt / n as f64;
        for _ in 0..n {
            self.step(h);
        }
        if !self.cell.v.is_finite() {
            return Err(SimError::Advance(
                self.id.clone(),
                "membrane potential diverged".into(),
            ));
        }
        self.state = self.cell.state();
        Ok(())
    }

    fn state(&self) -> &[f64] {
        &self.state
    }

    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        if state.len() != self.state.len() {
            return Err(SimError::Advance(
                self.id.clone(),
                format!(
                    "expected {} state values, got {}",
                    self.state.len(),
                    state.len()
                ),
            ));
        }
        // Rebuild the cell from its state vector, keeping the external concentrations.
        let (nao, ko, cao) = (self.cell.nao, self.cell.ko, self.cell.cao);
        self.cell = TenTusscher {
            v: state[0],
            m: state[1],
            h: state[2],
            j: state[3],
            xr1: state[4],
            xr2: state[5],
            xs: state[6],
            r: state[7],
            s: state[8],
            d: state[9],
            f: state[10],
            fca: state[11],
            g: state[12],
            nai: state[13],
            ki: state[14],
            cai: state[15],
            ca_sr: state[16],
            nao,
            ko,
            cao,
        };
        self.time = time;
        self.state = state.to_vec();
        Ok(())
    }
}

/// The sarcomere length ratio at time `t` (ms) under the shortening protocol. A
/// `velocity_fraction` of 0 gives the isometric case.
fn length_ratio(t: f64, velocity_fraction: f64) -> f64 {
    let elapsed = (t - SHORTEN_START).clamp(0.0, SHORTEN_DURATION);
    1.0 - velocity_fraction * elapsed
}

/// The Land et al. contractile element, with the length protocol. Its output is
/// `[XS, XW, TRPN, TmBlocked, ζS, ζW, Cd, λ, total tension]`.
struct ContractileModel {
    id: String,
    time: f64,
    model: LandActiveTension,
    /// Shortening speed as a fraction of resting length per ms (0 = isometric).
    speed: f64,
    /// Output buffer, see the type docs.
    state: Vec<f64>,
    /// Coupled input: the cytosolic Ca²⁺ (µM), written by the orchestrator.
    input: Vec<f64>,
}

impl ContractileModel {
    fn new(id: &str, speed: f64) -> Self {
        let model = LandActiveTension::new();
        let state = Self::pack(&model, 1.0);
        Self {
            id: id.to_string(),
            time: 0.0,
            model,
            speed,
            state,
            input: vec![0.0],
        }
    }

    fn pack(model: &LandActiveTension, lambda: f64) -> Vec<f64> {
        vec![
            model.xs,
            model.xw,
            model.trpn,
            model.tm_blocked,
            model.zeta_s,
            model.zeta_w,
            model.cd,
            lambda,
            model.total_tension(lambda),
        ]
    }

    /// Length ratio and its rate at the model's current time.
    fn kinematics(&self, t: f64) -> (f64, f64) {
        let dl = if (SHORTEN_START..SHORTEN_START + SHORTEN_DURATION).contains(&t) {
            -self.speed
        } else {
            0.0
        };
        (length_ratio(t, self.speed), dl)
    }
}

impl SubModel for ContractileModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn max_step(&self) -> f64 {
        DT
    }

    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        // The Ca²⁺ input is held over the step, which is how the orchestrator
        // applies couplings. The model integrates with fixed Runge–Kutta steps.
        let cai = self.input[0];
        let n = (dt / CONTRACTION_STEP).ceil().max(1.0) as usize;
        let h = dt / n as f64;
        for _ in 0..n {
            let (lambda, dlambda) = self.kinematics(self.time);
            self.model.advance(cai, lambda, dlambda, h);
            self.time += h;
        }
        let (lambda, _) = self.kinematics(self.time);
        self.state = Self::pack(&self.model, lambda);
        if !self.state.iter().all(|v| v.is_finite()) {
            return Err(SimError::Advance(
                self.id.clone(),
                "contraction state diverged".into(),
            ));
        }
        Ok(())
    }

    fn state(&self) -> &[f64] {
        &self.state
    }

    fn input_mut(&mut self) -> Option<&mut [f64]> {
        Some(&mut self.input)
    }

    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        if state.len() != self.state.len() {
            return Err(SimError::Advance(
                self.id.clone(),
                format!(
                    "expected {} state values, got {}",
                    self.state.len(),
                    state.len()
                ),
            ));
        }
        self.model = LandActiveTension {
            xs: state[0],
            xw: state[1],
            trpn: state[2],
            tm_blocked: state[3],
            zeta_s: state[4],
            zeta_w: state[5],
            cd: state[6],
        };
        self.time = time;
        self.state = state.to_vec();
        Ok(())
    }
}

/// Time-series output of one coupled run.
struct Run {
    voltage: Vec<(f64, f64)>,
    calcium: Vec<(f64, f64)>,
    tension: Vec<(f64, f64)>,
}

/// Time (ms) at which `series` peaks, with the peak value.
fn peak(series: &[(f64, f64)]) -> (f64, f64) {
    series.iter().copied().fold(
        (0.0, f64::MIN),
        |best, p| if p.1 > best.1 { p } else { best },
    )
}

/// Run the coupled cell and contraction model with the given shortening speed.
fn run(speed: f64) -> Run {
    let mut sim = Simulation::new();
    sim.add_model(CellModel::new("electrical"))
        .expect("unique id");
    sim.add_model(ContractileModel::new("mechanics", speed))
        .expect("unique id");
    // The cell reports Ca²⁺ in mM; the contraction model takes µM.
    sim.add_coupling(Coupling::new("electrical", "mechanics", |src, input| {
        input[0] = src[CA_INDEX] * 1000.0;
    }));

    let mut out = Run {
        voltage: Vec::new(),
        calcium: Vec::new(),
        tension: Vec::new(),
    };
    let mut t = 0.0;
    while t < DURATION {
        t += DT;
        sim.step_until(t).expect("coupled step");
        let cell = sim.model("electrical").expect("registered").state();
        let contraction = sim.model("mechanics").expect("registered").state();
        out.voltage.push((t, cell[V_INDEX]));
        out.calcium.push((t, cell[CA_INDEX]));
        out.tension.push((t, contraction[8]));
    }
    out
}

/// Peak tension (kPa) of a trace within `[from, to]` ms.
fn window_peak(series: &[(f64, f64)], from: f64, to: f64) -> f64 {
    series
        .iter()
        .filter(|p| (from..=to).contains(&p.0))
        .map(|p| p.1)
        .fold(f64::MIN, f64::max)
}

fn main() {
    // Shortening speed: 10 % of resting length over the protocol duration.
    let speed = SHORTEN_FRACTION / SHORTEN_DURATION;
    let isometric = run(0.0);
    let shortening = run(speed);

    let (v_peak_t, v_peak) = peak(&isometric.voltage);
    let (ca_peak_t, ca_peak) = peak(&isometric.calcium);
    let (ta_peak_t, ta_peak) = peak(&isometric.tension);

    // Action-potential duration: time above −40 mV, from the first upstroke.
    let voltage = &isometric.voltage;
    let up = voltage.iter().position(|p| p.1 > -40.0);
    let down = up.and_then(|i| {
        voltage[i..]
            .iter()
            .position(|p| p.1 < -40.0)
            .map(|j| voltage[i + j].0)
    });
    let apd = match (up, down) {
        (Some(i), Some(d)) => d - voltage[i].0,
        _ => f64::NAN,
    };

    // Force–velocity: compare the peak tension within the shortening window.
    let win_start = SHORTEN_START;
    let win_end = SHORTEN_START + SHORTEN_DURATION;
    let f_iso = window_peak(&isometric.tension, win_start, win_end);
    let f_short = window_peak(&shortening.tension, win_start, win_end);

    println!("action potential");
    println!("  peak V              {v_peak:7.2} mV at {v_peak_t:6.1} ms");
    println!("  APD (-40 mV)        {apd:7.1} ms");
    println!("calcium transient");
    println!(
        "  peak Ca_i           {:7.3} µM at {ca_peak_t:6.1} ms",
        ca_peak * 1000.0
    );
    println!("contraction (Land et al. 2017, length and velocity dependent)");
    println!("  isometric peak      {ta_peak:7.2} kPa at {ta_peak_t:6.1} ms");
    println!("  lag after Ca peak   {:7.1} ms", ta_peak_t - ca_peak_t);
    println!(
        "  peak in shortening window ({win_start:.0}–{win_end:.0} ms): isometric {f_iso:6.2} kPa, shortening {f_short:6.2} kPa"
    );

    // The checks. The cell must depolarize and plateau, the calcium transient must
    // follow the upstroke, the tension must develop and lag the calcium peak, and
    // shortening must lower the tension (force–velocity).
    assert!(v_peak > 0.0, "no action potential: peak {v_peak} mV");
    assert!(
        (200.0..350.0).contains(&apd),
        "APD {apd} ms outside the epicardial range"
    );
    assert!(
        ca_peak_t > v_peak_t - 5.0 && ca_peak_t < v_peak_t + 100.0,
        "Ca transient should follow the upstroke"
    );
    assert!(ta_peak > 0.0, "no active tension: peak {ta_peak} kPa");
    assert!(
        ta_peak_t > ca_peak_t,
        "tension should lag the calcium peak (excitation–contraction delay)"
    );
    assert!(
        f_short < f_iso,
        "shortening should lower the tension: shortening {f_short} vs isometric {f_iso} kPa"
    );
    println!("all checks passed");
}
