//! Radiative-forcing and thermal-inertia tour of the `tpt-sci-climate` surface.
//!
//! The warming tour (`examples/warming.rs`) fits the logarithmic CO₂ sensitivity
//! and couples a CH₄ tracer. This example instead exercises three *different*
//! questions, all from the same public surface
//! ([`EnergyBalanceModel`], [`ClimateError`], [`SIGMA`], [`CO2_PREINDUSTRIAL`],
//! the re-exported constants):
//!
//! 1. **Forcing equivalence** — a CO₂ doubling warms by `ΔF = 5.35·ln 2`
//!    W/m². Since the TOA balance is `(1-α)·S/4 - εσT⁴ + ΔF = 0`, that warming
//!    can be *exactly cancelled* by lowering the solar constant by
//!    `ΔS = -4·ΔF/(1-α)`. We verify that the equilibrium temperature of the
//!    compensated model equals the unperturbed one: solar and CO₂ forcings are
//!    interchangeable once they share the same `ΔF`.
//! 2. **Two-axis sensitivity** — a small table of equilibrium temperature over
//!    CO₂ multipliers × solar factors, confirming monotonic response to both
//!    radiative controls.
//! 3. **Thermal inertia / lag** — the same 2×CO₂ step is integrated with three
//!    very different heat capacities (land-like, mixed-layer, deep-ocean-like).
//!    The e-folding relaxation time `τ = C / (4·εσ·T³)` grows with `C`, so a
//!    deep reservoir commits the planet to decades of *delayed* warming even
//!    after the forcing is applied.
//!
//! Everything is deterministic scalar arithmetic. Run with
//! `cargo run --example energy_balance -p tpt-sci-climate`.

use tpt_sci_climate::{CO2_PREINDUSTRIAL, ClimateError, EnergyBalanceModel, SIGMA};

/// Seconds in a day (explicit-Euler stride for the transient integrations).
const DAY: f64 = 86_400.0;
/// Pre-industrial albedo / emissivity used throughout.
const ALBEDO: f64 = 0.3;
const EMIS: f64 = 0.61;
/// Modern solar constant (W/m²), set by the constructor.
const S0: f64 = 1361.0;
/// Pre-industrial CO₂ (ppm).
const C0: f64 = CO2_PREINDUSTRIAL;

/// Loud assertion that `value` lies in `[lo, hi]` (both inclusive).
fn assert_within(label: &str, value: f64, lo: f64, hi: f64) {
    assert!(
        (lo..=hi).contains(&value),
        "{label}: {value} is outside the plausible range [{lo}, {hi}]"
    );
}

/// Loud assertion that `result` was rejected with [`ClimateError::InvalidModel`].
fn expect_invalid<T>(label: &str, result: Result<T, ClimateError>) {
    match result {
        Err(ClimateError::InvalidModel(msg)) => println!("  {label:<22} -> rejected: {msg}"),
        Ok(_) => panic!("{label}: expected an error, but the call succeeded"),
    }
}

/// Linear interpolation of the first time `target` is crossed, between the two
/// bracketing `(t, y)` samples.
fn interpolate(samples: &[(f64, f64)], target: f64) -> Option<f64> {
    for w in samples.windows(2) {
        let (t1, f1) = w[0];
        let (t2, f2) = w[1];
        if (f1 - target) * (f2 - target) <= 0.0 && f1 != f2 {
            let w = (target - f1) / (f2 - f1);
            return Some(t1 + w * (t2 - t1));
        }
    }
    None
}

/// Equilibrium temperature of a fresh EBM at `co2` ppm and solar constant `s`.
fn equilibrium(co2: f64, s: f64) -> f64 {
    let mut e = EnergyBalanceModel::new(1.0e7, ALBEDO, EMIS, co2).unwrap();
    e.solar_constant = s;
    e.equilibrium_temperature()
}

/// [1] Forcing equivalence between a CO₂ doubling and a solar reduction.
fn forcing_equivalence() {
    println!("\n[1] Forcing equivalence: a 2xCO2 warming == a -ΔS solar cut");

    let t_base = equilibrium(C0, S0);
    let f_co2 = 5.35 * (2.0_f64).ln(); // W/m^2 from doubling CO2
    // Cancel it with the Sun: (1-a)*ΔS/4 + F = 0  ->  ΔS = -4F/(1-a).
    let ds = -4.0 * f_co2 / (1.0 - ALBEDO);
    let s_comp = S0 + ds;

    let mut e = EnergyBalanceModel::new(1.0e7, ALBEDO, EMIS, 2.0 * C0).unwrap();
    e.solar_constant = s_comp;
    let t_comp = e.equilibrium_temperature();

    println!("  2xCO2 forcing F = {f_co2:.3} W/m^2  ->  compensate with ΔS = {ds:.2} W/m^2");
    println!(
        "  T_eq(base 1xCO2, S0)        = {t_base:.4} K\n  \
         T_eq(2xCO2, S0+ΔS)        = {t_comp:.4} K"
    );
    assert_within("base equilibrium T", t_base, 280.0, 296.0);
    assert!(
        (t_comp - t_base).abs() < 1e-6,
        "compensated model must have the same equilibrium temperature"
    );

    // The two fixed points also share the same TOA net flux *at equilibrium*:
    // `equilibrium_temperature` returns the balance temperature without mutating
    // the model, so pin the model's state to it before checking the flux.
    e.t = t_comp;
    assert!(
        e.net_flux().abs() < 1e-6,
        "compensated TOA flux must balance"
    );
    println!("  -> solar and CO2 forcings are interchangeable in equilibrium ✓");
}

/// [2] Equilibrium temperature over CO₂ multipliers × solar factors.
fn sensitivity_grid() {
    println!("\n[2] Two-axis sensitivity (equilibrium T over CO2 × solar)");

    let co2_mult = [0.5, 1.0, 2.0, 4.0];
    let solar_fac = [0.95, 1.0, 1.05];

    // Build the full 2-D table first, then assert monotonicity within each row
    // (over CO2 at fixed solar) and within each column (over solar at fixed CO2).
    let mut table = vec![vec![0.0; solar_fac.len()]; co2_mult.len()];
    for (i, &m) in co2_mult.iter().enumerate() {
        for (j, &sf) in solar_fac.iter().enumerate() {
            let t = equilibrium(C0 * m, S0 * sf);
            table[i][j] = t;
            println!("    CO2 x{m:<4} S x{sf:<5} -> T = {t:8.3} K");
        }
    }
    for j in 0..solar_fac.len() {
        for i in 1..co2_mult.len() {
            assert!(
                table[i][j] > table[i - 1][j],
                "T must rise with CO2 (mult {}, S x{})",
                co2_mult[i],
                solar_fac[j]
            );
        }
    }
    for i in 0..co2_mult.len() {
        for j in 1..solar_fac.len() {
            assert!(
                table[i][j] > table[i][j - 1],
                "T must rise with solar (mult {}, S x{})",
                co2_mult[i],
                solar_fac[j]
            );
        }
    }
    println!("  -> equilibrium T increases monotonically with both controls ✓");
}

/// [3] Transient lag across three heat capacities (land → deep ocean).
fn thermal_inertia() {
    println!("\n[3] Thermal inertia: e-folding lag τ = C / (4εσT³)");

    // Representative ocean-equivalent heat capacities (J/m²/K): a shallow
    // land slab, the ~70 m mixed layer, and a large deep reservoir.
    let capacities = [
        ("land slab", 1.0e7),
        ("mixed layer", 70.0 * 4.1816e6),
        ("deep reservoir", 1.0e9),
    ];

    let steps = 20_000usize; // ~54.8 yr at a 1-day stride
    println!(
        "  {:>15} {:>10} {:>12} {:>12} {:>12}",
        "reservoir", "C[J/m2/K]", "τ_analytic[d]", "τ_measured[d]", "2xCO2 warm[K]"
    );
    for (name, c) in capacities {
        let mut ebm = EnergyBalanceModel::new(c, ALBEDO, EMIS, C0).unwrap();
        let t_eq = ebm.equilibrium_temperature();
        ebm.co2 = 2.0 * C0;
        let t_target = ebm.equilibrium_temperature();
        let warming = t_target - t_eq;

        let lambda = 4.0 * EMIS * SIGMA * t_eq.powi(3);
        let tau_analytic = c / lambda; // seconds

        let mut samples: Vec<(f64, f64)> = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let frac = (ebm.temperature() - t_eq) / (t_target - t_eq);
            samples.push((i as f64 * DAY, frac));
            ebm.step(DAY);
        }
        let tau_meas = interpolate(&samples, 1.0 - 1.0 / std::f64::consts::E)
            .expect("1/e crossing must be sampled");

        println!(
            "  {:>15} {:>10.2e} {:>12.1} {:>12.1} {:>12.3}",
            name,
            c,
            tau_analytic / DAY,
            tau_meas / DAY,
            warming
        );
        assert_within(
            "measured e-folding",
            tau_meas,
            0.85 * tau_analytic,
            1.15 * tau_analytic,
        );
        assert_within("2xCO2 warming", warming, 1.0, 5.0);
    }
    println!("  -> deeper reservoirs lag longer (commit to delayed warming) ✓");
}

/// [4] Validation surface (a single rejected configuration).
fn validation() {
    println!("\n[4] Validation (ClimateError::InvalidModel)");
    expect_invalid(
        "co2 <= 0",
        EnergyBalanceModel::new(1.0e7, ALBEDO, EMIS, 0.0),
    );
}

fn main() {
    println!("=== tpt-sci-climate: forcing, sensitivity & thermal inertia ===");
    forcing_equivalence();
    sensitivity_grid();
    thermal_inertia();
    validation();
    println!("\nAll diagnostics and assertions passed.");
}
