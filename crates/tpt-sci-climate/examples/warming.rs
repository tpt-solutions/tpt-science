//! Tour of the `tpt-sci-climate` surface — a 0-D global energy-balance model
//! and its companions (grey radiative transfer, a tracer chemistry box), built
//! on [`tpt_sci_ode`] and [`tpt_math_linalg`].
//!
//! The planet is treated as a single well-mixed slab. The energy balance is
//!
//! ```text
//!     C dT/dt = (1 - alpha) S/4 - epsilon sigma T^4 + F_CO2,
//!     F_CO2   = 5.35 ln(C / C0)        W/m^2   (CO2 radiative forcing)
//! ```
//!
//! where `C` is the heat capacity, `alpha` the albedo, `epsilon` the effective
//! emissivity, `S` the solar constant and `sigma` the Stefan–Boltzmann
//! constant. We exercise the full public surface of the crate:
//!
//! * [`EnergyBalanceModel`] — constructor, accessors (`temperature`,
//!   `forcing`, `net_flux`), `step` (explicit Euler) and
//!   `equilibrium_temperature` (Newton fixpoint);
//! * the CO₂ forcing law and the logarithmic climate-sensitivity curve;
//! * a transient relaxation integrated both with the model's own `step` and
//!   with [`tpt_sci_ode`] (cross-checked, e-folding time reported);
//! * parameter sensitivity (albedo, solar constant) and a faint-Young-Sun
//!   inversion that requires an extreme CO₂ to stay warm;
//! * a water-vapour-like feedback (effective emissivity falling with `T`) that
//!   lifts equilibrium climate sensitivity into the observed range;
//! * [`grey_radiative_transfer`] — the single-layer grey atmosphere;
//! * [`ChemistryBox`] — a production–loss tracer, its steady state, and a
//!   coupling of CH₄ forcing back into the energy balance;
//! * the [`ClimateError`] validation surface.
//!
//! Everything is deterministic and fast (pure scalar loops). Run with
//! `cargo run --example warming -p tpt-sci-climate`.

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_sci_climate::{
    CO2_PREINDUSTRIAL, ChemistryBox, ClimateError, EnergyBalanceModel, SIGMA,
    grey_radiative_transfer,
};
use tpt_sci_ode::{Method, OdeProblem};

/// Seconds in a Julian year.
const YR: f64 = 31_557_600.0;
/// Seconds in a day (integration step for the model's explicit Euler).
const DAY: f64 = 86_400.0;
/// Realistic ocean mixed-layer heat capacity (J/m²/K), ~70 m of water.
const OCEAN_C: f64 = 70.0 * 4.1816e6;
/// Pre-industrial albedo / emissivity used throughout the tour.
const ALBEDO: f64 = 0.3;
const EMIS: f64 = 0.61;
/// Pre-industrial CO₂ (ppm) and reference surface temperature (K).
const C0: f64 = CO2_PREINDUSTRIAL;
const T0: f64 = 288.0;

// ---------------------------------------------------------------------------
// Self-checking diagnostics.
// ---------------------------------------------------------------------------

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
        Err(ClimateError::InvalidModel(msg)) => println!("  {label:<28} -> rejected: {msg}"),
        Ok(_) => panic!("{label}: expected an error, but the call succeeded"),
    }
}

/// Linear interpolation of the first time `frac` reaches `target`, between the
/// two bracketing samples.
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

// ---------------------------------------------------------------------------
// Sections.
// ---------------------------------------------------------------------------

/// Pre-industrial reference state and the grey-atmosphere consistency check.
fn baseline() {
    println!("\n[1] Baseline: pre-industrial equilibrium & grey-atmosphere check");

    let ebm = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0).unwrap();
    let t_eq = ebm.equilibrium_temperature();
    println!("  pre-industrial equilibrium T = {t_eq:.3} K  (reference {T0} K)");
    assert_within("pre-industrial T", t_eq, T0 - 1.0, T0 + 1.0);

    // A single grey atmospheric layer of emissivity eps gives an *effective*
    // planetary emissivity eps_eff = 1 - eps/2. The grey-atmosphere surface
    // temperature must equal the EBM equilibrium for that eps_eff.
    let eps_layer = 2.0 * (1.0 - EMIS);
    println!("  grey layer eps = {eps_layer:.4}  ->  eps_eff = 1 - eps/2 = {EMIS}");
    let t_grey = grey_radiative_transfer(ebm.solar_constant, ALBEDO, eps_layer);
    println!("  grey Ts = {t_grey:.3} K  vs EBM Teq = {t_eq:.3} K");
    assert!(
        (t_grey - t_eq).abs() < 1e-8,
        "grey transfer must match EBM equilibrium"
    );

    // Planck feedback: lambda = 4 eps sigma T^3 (W/m^2/K).
    let lambda = 4.0 * EMIS * SIGMA * t_eq.powi(3);
    println!("  Planck feedback lambda = {lambda:.3} W/m^2/K");
    assert_within("Planck lambda", lambda, 3.0, 3.8);

    // Pre-industrial forcing is ~0, so the surface is at radiative balance.
    println!("  pre-industrial forcing = {:.3} W/m^2", ebm.forcing());
    println!(
        "  net TOA flux at equilibrium = {:.3e} W/m^2",
        ebm.net_flux()
    );
    assert!(ebm.forcing().abs() < 1e-6);
}

/// CO₂ sensitivity sweep with a least-squares fit of ΔT to ln(C/C0).
fn co2_sweep() {
    println!("\n[2] CO2 sweep (0.5x..8x) and logarithmic sensitivity");

    let multipliers = [0.5, 1.0, 2.0, 4.0, 8.0];
    let mut temps = Vec::new();
    let mut prev = f64::NEG_INFINITY;

    println!(
        "  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "mult", "CO2[ppm]", "F[W/m2]", "Teq[K]", "dT[K]"
    );
    for &m in &multipliers {
        let ebm = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0 * m).unwrap();
        let t = ebm.equilibrium_temperature();
        temps.push(t);
        let dt = t - temps[0];
        println!(
            "  {:>6.2} {:>12.1} {:>12.3} {:>12.3} {:>12.3}",
            m,
            C0 * m,
            ebm.forcing(),
            t,
            dt
        );
        assert!(t > prev, "warming must increase with CO2 (mult {m})");
        prev = t;
    }

    // Fit dT[m] = a + b ln(m) with tpt-math-linalg (normal equations A^T A x = A^T y).
    // A has columns [1, ln m], so A^T A = [[sa, sb], [sb, sbb]] and
    // A^T y = [sum(y), sum(ln(m)*y)].
    let n = multipliers.len();
    let mut sa = 0.0;
    let mut sb = 0.0;
    let mut sy = 0.0;
    let mut sbb = 0.0;
    let mut sby = 0.0;
    for (i, &m) in multipliers.iter().enumerate() {
        let a = 1.0;
        let b = m.ln();
        let y = temps[i] - temps[0];
        sa += a;
        sb += b;
        sy += y;
        sbb += b * b;
        sby += b * y;
    }
    let gram = DMatrix::from_vec(2, 2, vec![sa, sb, sb, sbb]);
    let rhs = DVector::from_vec(vec![sy, sby]);
    let sol = gram
        .solve(&rhs)
        .expect("2x2 normal equations are well conditioned");
    let b = sol[1];
    println!("  least-squares: dT = a + b ln(mult), b = {b:.3} K per ln(CO2)");
    println!(
        "  -> ~{:.2} K per CO2 doubling (5.35 ln2 b)",
        b * 5.35_f64.ln()
    );
    assert_within("sensitivity b", b, 1.0, 2.0);

    // Sensitivity curve, normalised to the sweep.
    let lo = (multipliers[0].ln().min(multipliers[1].ln()) - 0.05).floor() - 0.1;
    let hi = (multipliers[n - 1].ln() + 0.05).ceil();
    let step = (hi - lo) / 49.0;
    let mut bars: Vec<(f64, f64)> = Vec::new();
    let mut x = lo;
    while x <= hi + 1e-9 {
        let t = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0 * x.exp())
            .unwrap()
            .equilibrium_temperature();
        bars.push((x, t - temps[0]));
        x += step;
    }
    println!("\n  sensitivity curve: vertical bar length = 1 K");
    println!("  ln(CO2/C0)");
    for (lx, dts) in &bars {
        let n_bars = (dts * 20.0).round() as usize;
        println!("  {:>6.2} |{:<-width$}", lx, "", width = n_bars);
    }
}

/// Transient relaxation toward 2xCO2, integrated two ways.
fn transient() {
    println!("\n[3] Transient relaxation to 2xCO2 (Euler vs tpt-sci-ode)");

    let mut ebm = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0).unwrap();
    let t_eq = ebm.equilibrium_temperature();
    ebm.co2 = 2.0 * C0;

    let t_target = ebm.equilibrium_temperature();
    let c = ebm.heat_capacity;
    let lambda = 4.0 * ebm.emissivity * SIGMA * t_eq.powi(3);
    let tau_analytic = c / lambda;
    println!(
        "  new equilibrium = {:.3} K (warming {:.3} K), analytic e-folding tau = {:.1} d",
        t_target,
        t_target - t_eq,
        tau_analytic / DAY
    );

    // Euler march with the model's own `step`, recording the fraction relaxed.
    // `samples` is recorded every step: the 1/e and 95% crossings below are
    // found by linear interpolation, so a coarse (multi-year) stride would
    // measurably bias the timings on an exponential curve.
    let n_steps = (20.0 * YR / DAY).round() as usize;
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut flux_samples: Vec<(f64, f64)> = Vec::new();
    for i in 0..=n_steps {
        let frac = (ebm.temperature() - t_eq) / (t_target - t_eq);
        samples.push((i as f64 * DAY, frac));
        if i % (n_steps / 10).max(1) == 0 {
            flux_samples.push((i as f64 * DAY, ebm.net_flux()));
        }
        ebm.step(DAY);
    }
    let t_euler = ebm.temperature();
    println!(
        "  explicit-Euler final T = {t_euler:.3} K, net TOA flux = {:.3e} W/m^2",
        ebm.net_flux()
    );
    assert!(
        ebm.net_flux().abs() < 1e-1,
        "should be near radiative balance"
    );

    let tau_euler = interpolate(&samples, 1.0 - 1.0 / std::f64::consts::E)
        .expect("1/e crossing must be sampled");
    let t95 = interpolate(&samples, 0.95).expect("95% crossing must be sampled");
    println!(
        "  measured e-folding time = {:.1} d  (95% equilibration {:.1} d)",
        tau_euler / DAY,
        t95 / DAY
    );
    assert_within(
        "e-folding time",
        tau_euler,
        0.9 * tau_analytic,
        1.1 * tau_analytic,
    );

    // Independent integration through tpt-sci-ode over the same window.
    let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        let f = (1.0 - ALBEDO) * 1361.0 / 4.0 - EMIS * SIGMA * y[0].powi(4)
            + 5.35 * (2.0 * C0 / C0).ln();
        dydt[0] = f / c;
    };
    let prob = OdeProblem::new(rhs, vec![t_eq], 0.0).unwrap();
    // `solve_dense` requires every t_eval strictly after t0, so the grid starts
    // at one stride in (not 0.0). It spans the same 20 yr window as the Euler
    // march above so the two final temperatures are directly comparable.
    let span = 20.0 * YR;
    let times: Vec<f64> = (1..=10).map(|i| i as f64 * span / 10.0).collect();
    let traj = prob
        .solve_dense(Method::Tsit45, &times)
        .expect("ODE integration of the EBM must succeed");
    let t_ode = *traj
        .last()
        .expect("dense output is non-empty")
        .first()
        .expect("1 state");
    println!("  tpt-sci-ode (Tsit45) final T = {t_ode:.3} K");
    assert!(
        (t_ode - t_euler).abs() < 5e-3,
        "Euler and ODE integrators must agree"
    );

    // Tabulate the relaxation.
    println!(
        "  {:>8} {:>10} {:>10} {:>10}",
        "t[yr]", "T[K]", "relaxed", "netF[W/m2]"
    );
    for (i, t) in times.iter().enumerate() {
        let frac = (traj[i][0] - t_eq) / (t_target - t_eq);
        let f = (1.0 - ALBEDO) * 1361.0 / 4.0 - EMIS * SIGMA * traj[i][0].powi(4)
            + 5.35 * (2.0 * C0 / C0).ln();
        println!(
            "  {:>8.1} {:>10.3} {:>10.3} {:>10.3}",
            t / YR,
            traj[i][0],
            frac,
            f
        );
    }
}

/// Parameter sensitivity and the faint-Young-Sun inversion.
fn sensitivity() {
    println!("\n[4] Parameter sensitivity (albedo, solar constant)");

    print_curve("albedo a", 0.25, 0.40, 0.03, |a| {
        EnergyBalanceModel::new(OCEAN_C, a, EMIS, C0)
            .unwrap()
            .equilibrium_temperature()
    });
    print_curve("solar S [W/m2]", 1320.0, 1400.0, 20.0, |s| {
        let mut e = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0).unwrap();
        e.solar_constant = s;
        e.equilibrium_temperature()
    });

    // Faint-Young-Sun: S = 0.75 S0. Inverting the forcing keeps T at T0.
    // Balance is (1-a)*S/4 + F = eps*sigma*T0^4, so the extra forcing needed is
    // F = eps*sigma*T0^4 - (1-a)*S/4.
    const S0: f64 = 1361.0;
    let s_young = 0.75 * S0;
    let f_needed = EMIS * SIGMA * T0.powi(4) - (1.0 - ALBEDO) * s_young / 4.0;
    let co2_young = C0 * (f_needed / 5.35).exp();
    // The reduced solar constant must be applied too, or the equilibrium below
    // would be solved against the modern Sun.
    let mut ebm_young = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, co2_young).unwrap();
    ebm_young.solar_constant = s_young;
    let t_young = ebm_young.equilibrium_temperature();
    println!("  faint-Young-Sun S = {s_young:.0} W/m^2 needs CO2 = {co2_young:.2e} ppm");
    println!(
        "  (equilibrium T = {t_young:.2} K)  -> ~{:.0}x pre-industrial!",
        co2_young / C0
    );
    assert_within("faint-Sun T", t_young, T0 - 0.1, T0 + 0.1);
}

/// Print a small monotonic curve of `f(x)` over [lo, hi].
fn print_curve(label: &str, lo: f64, hi: f64, step: f64, f: impl Fn(f64) -> f64) {
    println!("  {label}:");
    let mut x = lo;
    while x <= hi + 1e-9 {
        println!("    {:>8.2} -> T = {:>8.2} K", x, f(x));
        x += step;
    }
}

/// Water-vapour-like feedback: emissivity falling with `T` lifts ECS.
fn with_feedback() {
    println!("\n[5] Water-vapour-like feedback (emissivity ~ 1 - c(T - T0))");

    // `emissivity` scales the *outgoing* flux here, so a warming-induced
    // increase in greenhouse trapping is a *decrease* in effective emissivity.
    // Using +c would model a negative (damping) feedback instead.
    let c = 0.005;
    let bare = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0).unwrap();
    let t_bare = bare.equilibrium_temperature();
    let planck_ecs = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, 2.0 * C0)
        .unwrap()
        .equilibrium_temperature()
        - t_bare;

    let mut t = t_bare;
    for _ in 0..100 {
        let eps = (EMIS * (1.0 - c * (t - T0))).clamp(1e-3, 0.999);
        let f =
            (1.0 - ALBEDO) * 1361.0 / 4.0 - eps * SIGMA * t.powi(4) + 5.35 * (2.0 * C0 / C0).ln();
        // d/dT[-eps(T)*sigma*T^4] = -4*eps*sigma*T^3 - (deps/dT)*sigma*T^4,
        // with deps/dT = -c*EMIS.
        let df = -4.0 * eps * SIGMA * t.powi(3) + c * EMIS * SIGMA * t.powi(4);
        let t_new = (t - f / df).max(1.0);
        if (t_new - t).abs() < 1e-10 {
            t = t_new;
            break;
        }
        t = t_new;
    }
    let ecs = t - t_bare;
    println!("  bare 2xCO2 warming (Planck only)   = {planck_ecs:.3} K");
    println!("  with feedback 2xCO2 warming (ECS)  = {ecs:.3} K");
    println!(
        "  feedback gain g = (ECS - Planck)/ECS = {:.3}",
        (ecs - planck_ecs) / ecs
    );
    assert_within("ECS (with feedback)", ecs, planck_ecs, 5.0);
    assert!(ecs > planck_ecs, "feedback must amplify warming");
    assert_within("2xCO2 warming (range)", ecs, 1.5, 5.0);
}

/// Atmospheric-chemistry tracer and its coupling back to the energy balance.
fn chemistry() {
    println!("\n[6] ChemistryBox tracer (CH4-like) & forcing coupling");

    let lifetime_yr = 9.1;
    let k = 1.0 / (lifetime_yr * YR);
    // Choose the target steady state (present-day CH4 ~1800 ppb) and derive the
    // production that sustains it: C* = P/k  =>  P = C*·k. Start the box at the
    // pre-industrial ~700 ppb so the relaxation below is a real transient.
    let target = 1800.0;
    let mut ch4 = ChemistryBox::new(700.0, target * k, k).unwrap();
    let steady = ch4.steady_state();
    println!(
        "  steady-state C* = P/k = {:.1} ppb (loss lifetime {lifetime_yr} yr)",
        steady
    );
    assert_within("CH4 steady state", steady, 1700.0, 1900.0);

    let c0 = ch4.concentration;
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let n = (60.0 * YR / (30.0 * DAY)).round() as usize;
    for i in 0..=n {
        if i % (n / 6).max(1) == 0 {
            samples.push((
                i as f64 * 30.0 * DAY,
                (ch4.concentration - c0) / (steady - c0),
            ));
        }
        ch4.step(30.0 * DAY);
    }
    let gap = (ch4.concentration - steady).abs();
    println!(
        "  after 60 yr: C = {:.1} ppb, |C - C*| = {:.3} ppb",
        ch4.concentration, gap
    );
    assert_within("CH4 convergence gap", gap, 0.0, 5.0);

    let tau_meas = interpolate(&samples, 1.0 - 1.0 / std::f64::consts::E)
        .expect("tracer 1/e crossing must be sampled");
    println!(
        "  measured e-folding time = {:.1} yr (theory {lifetime_yr} yr)",
        tau_meas / YR
    );
    assert_within(
        "CH4 e-folding",
        tau_meas,
        0.9 * lifetime_yr * YR,
        1.1 * lifetime_yr * YR,
    );

    // Couple CH4 forcing into the EBM via a CO2-equivalent concentration.
    // IPCC-style simplified CH4 forcing, F = alpha * (sqrt(M) - sqrt(M0)) with
    // M in ppb. The CH4-N2O band-overlap correction is omitted: it needs an
    // N2O concentration, which this CO2/CH4-only example does not track.
    let f_ch4 = 0.036 * (1750.0_f64.sqrt() - 700.0_f64.sqrt());
    let co2_eq = C0 * (f_ch4 / 5.35_f64).exp();
    let t_ch4 = EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, co2_eq)
        .unwrap()
        .equilibrium_temperature();
    let dwarm = t_ch4
        - EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, C0)
            .unwrap()
            .equilibrium_temperature();
    println!("  CH4 forcing = {f_ch4:.3} W/m^2  ->  CO2-eq = {co2_eq:.1} ppm");
    println!("  equilibrium warming from CH4 alone = {dwarm:.3} K");
    assert_within("CO2-eq from CH4", co2_eq, 200.0, 400.0);
    assert_within("CH4 warming", dwarm, 0.05, 0.5);
}

/// The validation / error surface.
fn error_surface() {
    println!("\n[7] Validation (ClimateError::InvalidModel)");

    expect_invalid(
        "heat capacity <= 0",
        EnergyBalanceModel::new(0.0, ALBEDO, EMIS, C0),
    );
    expect_invalid(
        "albedo > 1",
        EnergyBalanceModel::new(OCEAN_C, 1.2, EMIS, C0),
    );
    expect_invalid(
        "emissivity <= 0",
        EnergyBalanceModel::new(OCEAN_C, ALBEDO, 0.0, C0),
    );
    expect_invalid(
        "CO2 <= 0",
        EnergyBalanceModel::new(OCEAN_C, ALBEDO, EMIS, 0.0),
    );

    expect_invalid("concentration < 0", ChemistryBox::new(-1.0, 1.0, 0.1));
    expect_invalid("loss < 0", ChemistryBox::new(0.0, 1.0, -0.1));

    // Zero loss yields an unbounded steady state.
    let box0 = ChemistryBox::new(1.0, 1.0, 0.0).unwrap();
    println!("  zero-loss steady state = {}", box0.steady_state());
    assert!(box0.steady_state().is_infinite());
}

fn main() {
    println!("=== tpt-sci-climate: a tour of the climate surface ===");
    baseline();
    co2_sweep();
    transient();
    sensitivity();
    with_feedback();
    chemistry();
    error_surface();
    println!("\nAll diagnostics and assertions passed.");
}
