//! Coupled scenario: an atmosphere box exchanging heat with an ocean mixed layer.
//!
//! The atmosphere has a small heat capacity and feels the CO₂ forcing and the
//! radiative balance. The ocean mixed layer has a large heat capacity and
//! responds slowly. Each box is a `SubModel` with its own ODE, and the two are
//! coupled through the orchestrator: each writes its temperature into the
//! other's input buffer, and that input enters the right-hand side on the next
//! step.
//!
//! The check is physical. Warming lags the forcing (the ocean takes centuries
//! to equilibrate), and the coupled system ends at the same equilibrium as the
//! uncoupled energy-balance model, because heat exchange does not change the
//! steady state.
//!
//! Run with: `cargo run --release --example ocean_atmosphere -p tpt-sci-sim-core`

use std::rc::Rc;

use tpt_sci_climate::EnergyBalanceModel;
use tpt_sci_ode::{Method, OdeProblemBuilder};
use tpt_sci_sim_core::{Coupling, SimError, Simulation, SubModel};

const SOLAR_TERM: f64 = 1361.0 / 4.0 * (1.0 - 0.3); // S(1 − α)/4, W/m²
const EMISSIVITY: f64 = 0.61;
const SIGMA: f64 = 5.670_374_419e-8; // W/m²/K⁴
const KAPPA: f64 = 1.0; // heat exchange between the boxes, W/m²/K
const CO2_FORCING: f64 = 3.7; // doubling, W/m²
const YEARS: f64 = 1500.0;
const DT: f64 = 1.0; // years

/// One well-mixed box of heat, with an external flux that depends on its own
/// temperature (radiation for the atmosphere, nothing for the ocean).
struct HeatBox {
    id: String,
    capacity: f64, // W·yr/m²/K
    temperature: f64,
    time: f64,
    /// Coupled input: the other box's temperature, written by the orchestrator.
    input: Vec<f64>,
    /// Net external flux, W/m², as a function of this box's temperature.
    external: Rc<dyn Fn(f64) -> f64>,
}

impl HeatBox {
    fn new(id: &str, capacity: f64, t0: f64, external: Rc<dyn Fn(f64) -> f64>) -> Self {
        Self {
            id: id.to_string(),
            capacity,
            temperature: t0,
            time: 0.0,
            input: vec![t0],
            external,
        }
    }
}

impl SubModel for HeatBox {
    fn id(&self) -> &str {
        &self.id
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn max_step(&self) -> f64 {
        f64::INFINITY
    }

    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        // dT/dt = [external(T) + κ·(T_other − T)] / C, with T_other from the input.
        let external = Rc::clone(&self.external);
        let other = self.input[0];
        let capacity = self.capacity;
        let rhs = move |_t: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = (external(y[0]) + KAPPA * (other - y[0])) / capacity;
        };
        let problem = OdeProblemBuilder::new(rhs, vec![self.temperature], self.time)
            .rtol(1e-9)
            .atol(1e-11)
            .build()
            .map_err(|e| SimError::Advance(self.id.clone(), e.to_string()))?;
        let next = problem
            .solve(Method::Tsit45, self.time + dt)
            .map_err(|e| SimError::Advance(self.id.clone(), e.to_string()))?;
        self.temperature = next[0];
        self.time += dt;
        Ok(())
    }

    fn state(&self) -> &[f64] {
        std::slice::from_ref(&self.temperature)
    }

    fn input_mut(&mut self) -> Option<&mut [f64]> {
        Some(&mut self.input)
    }

    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        if state.len() != 1 {
            return Err(SimError::Advance(
                self.id.clone(),
                format!("expected 1 state value, got {}", state.len()),
            ));
        }
        self.temperature = state[0];
        self.time = time;
        Ok(())
    }
}

fn main() {
    // Radiative balance of the atmosphere with CO₂ doubled: the forcing enters
    // as an extra flux.
    let radiation =
        Rc::new(move |t: f64| SOLAR_TERM + CO2_FORCING - EMISSIVITY * SIGMA * t.powi(4));
    let t0 = 288.0;
    let mut sim = Simulation::new();
    sim.add_model(HeatBox::new("atmosphere", 8.0, t0, radiation))
        .expect("unique id");
    sim.add_model(HeatBox::new("ocean", 100.0, t0, Rc::new(|_t| 0.0)))
        .expect("unique id");
    sim.add_coupling(Coupling::new("ocean", "atmosphere", |src, input| {
        input[0] = src[0]
    }));
    sim.add_coupling(Coupling::new("atmosphere", "ocean", |src, input| {
        input[0] = src[0]
    }));

    let temp = |sim: &Simulation, id: &str| sim.model(id).expect("registered").state()[0];

    let mut report = Vec::new();
    let mut t = 0.0;
    while t < YEARS {
        t += DT;
        sim.step_until(t).expect("coupled step");
        if (t as u32) % 250 == 0 {
            report.push((t, temp(&sim, "atmosphere"), temp(&sim, "ocean")));
        }
    }
    println!("year   atmosphere (K)   ocean (K)");
    for (year, ta, to) in &report {
        println!("{year:>5.0}   {ta:>13.3}   {to:>9.3}");
    }

    // Equilibrium of the uncoupled energy-balance model at the same forcing.
    let mut eq = EnergyBalanceModel::new(1.0, 0.3, EMISSIVITY, 280.0).expect("valid EBM");
    eq.co2 = 280.0 * 2.0;
    let t_eq = eq.equilibrium_temperature();
    let t_final = temp(&sim, "atmosphere");
    let t_ocean = temp(&sim, "ocean");
    println!("\nequilibrium (energy-balance model): {t_eq:.3} K");
    println!("coupled atmosphere after {YEARS:.0} years: {t_final:.3} K");
    println!("coupled ocean after {YEARS:.0} years:      {t_ocean:.3} K");

    // The ocean lags the atmosphere while warming, then both converge.
    let (_, ta_early, to_early) = report[0];
    assert!(
        to_early < ta_early,
        "the ocean must lag the atmosphere early on"
    );
    assert!(
        (t_final - t_eq).abs() < 0.2,
        "coupled system should reach the EBM equilibrium"
    );
    assert!(
        (t_ocean - t_final).abs() < 0.2,
        "the boxes should converge to one temperature"
    );
}
