//! Coupled scenario: a human ventricular myocyte (Ten Tusscher–Panfilov 2004,
//! epicardial) driving the Land et al. active-tension model through its calcium
//! transient.
//!
//! Two `SubModel`s run under the orchestrator:
//!
//! * `electrical` is the ionic cell. It is stimulated once at t = 0 and advances
//!   its gates, concentrations and voltage with a fixed explicit step.
//! * `mechanics` is the Land et al. (2017) active-tension model at fixed resting
//!   length (isometric). It is driven by the cytosolic Ca²⁺ it receives from the
//!   cell, and it returns the active tension in kPa.
//!
//! The coupling is one-way (`Ca²⁺ → tension`), which is the excitation–contraction
//! link. Both models are published, and their parameters are taken from the
//! sources they cite: see `tpt_sci_electrophys::LandActiveTension` and
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

/// The Land et al. contractile element at fixed length. Its state is the four
/// cross-bridge and regulatory variables, followed by the active tension (kPa).
struct ContractileModel {
    id: String,
    time: f64,
    model: LandActiveTension,
    /// Output buffer: `[XS, XW, TRPN, TmBlocked, Ta]`.
    state: Vec<f64>,
    /// Coupled input: the cytosolic Ca²⁺ (µM), written by the orchestrator.
    input: Vec<f64>,
}

impl ContractileModel {
    fn new(id: &str) -> Self {
        let model = LandActiveTension::new();
        let state = Self::pack(&model);
        Self {
            id: id.to_string(),
            time: 0.0,
            model,
            state,
            input: vec![0.0],
        }
    }

    fn pack(model: &LandActiveTension) -> Vec<f64> {
        vec![
            model.xs,
            model.xw,
            model.trpn,
            model.tm_blocked,
            model.tension(),
        ]
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
            self.model.advance(cai, h);
        }
        self.time += dt;
        self.state = Self::pack(&self.model);
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
        if state.len() != 5 {
            return Err(SimError::Advance(
                self.id.clone(),
                format!("expected 5 state values, got {}", state.len()),
            ));
        }
        self.model = LandActiveTension {
            xs: state[0],
            xw: state[1],
            trpn: state[2],
            tm_blocked: state[3],
        };
        self.time = time;
        self.state = state.to_vec();
        Ok(())
    }
}

/// Time (ms) at which `series` peaks, with the peak value.
fn peak(series: &[(f64, f64)]) -> (f64, f64) {
    series.iter().copied().fold(
        (0.0, f64::MIN),
        |best, p| if p.1 > best.1 { p } else { best },
    )
}

fn main() {
    let mut sim = Simulation::new();
    sim.add_model(CellModel::new("electrical"))
        .expect("unique id");
    sim.add_model(ContractileModel::new("mechanics"))
        .expect("unique id");
    // The cell reports Ca²⁺ in mM; the contraction model takes µM.
    sim.add_coupling(Coupling::new("electrical", "mechanics", |src, input| {
        input[0] = src[CA_INDEX] * 1000.0;
    }));

    let mut voltage = Vec::new();
    let mut calcium = Vec::new();
    let mut tension = Vec::new();

    let mut t = 0.0;
    while t < DURATION {
        t += DT;
        sim.step_until(t).expect("coupled step");
        let cell = sim.model("electrical").expect("registered").state();
        let contraction = sim.model("mechanics").expect("registered").state();
        voltage.push((t, cell[V_INDEX]));
        calcium.push((t, cell[CA_INDEX]));
        tension.push((t, contraction[4]));
    }

    let (v_peak_t, v_peak) = peak(&voltage);
    let (ca_peak_t, ca_peak) = peak(&calcium);
    let (ta_peak_t, ta_peak) = peak(&tension);

    // Action-potential duration: time above −40 mV, from the first upstroke.
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

    println!("action potential");
    println!("  peak V            {v_peak:7.2} mV at {v_peak_t:6.1} ms");
    println!("  APD (-40 mV)      {apd:7.1} ms");
    println!("calcium transient");
    println!(
        "  peak Ca_i         {:7.3} µM at {ca_peak_t:6.1} ms",
        ca_peak * 1000.0
    );
    println!("contraction (Land et al. 2017, isometric)");
    println!("  peak active tension {ta_peak:7.2} kPa at {ta_peak_t:6.1} ms");
    println!("  lag after Ca peak   {:7.1} ms", ta_peak_t - ca_peak_t);

    // The checks. The cell must depolarize and plateau, the calcium transient must
    // follow the upstroke, and the tension must develop and lag the calcium peak.
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
    println!("all checks passed");
}
