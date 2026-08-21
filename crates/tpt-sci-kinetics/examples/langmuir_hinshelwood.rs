//! # Langmuir–Hinshelwood surface kinetics tour
//!
//! A second, complementary tour of the [`tpt_sci_kinetics`] public surface — the
//! first one (`reactor`) integrates a single coverage-dependent mechanism over
//! time and watches Arrhenius sensitivity / exothermic-adsorption coverage
//! collapses with temperature. This example instead probes the **kinetic
//! control** of surface processes, i.e. how *selectivity* and *steady-state
//! rate* are shaped by Arrhenius barriers and by the Langmuir site balance:
//!
//! * **Pressure-driven adsorption isotherm** — `θ_A(p)` from
//!   [`langmuir_hinshelwood_coverages`] for a single adsorbate, showing the
//!   Henry (low-`p`, `θ ∝ p`) and saturation (high-`p`, `θ → 1`) limits of the
//!   Langmuir isotherm (Section 1).
//! * **Temperature-dependent selectivity through a parallel network** — two
//!   competing desorption/product channels `A* → B*` and `A* → C*` with different
//!   Arrhenius parameters, resolved at several temperatures via
//!   [`KineticsProblem`] and driven with the [`tpt_sci_reaction_network`] ODE
//!   backend. Low `Ea` wins at low `T`; high-`A`/high-`Ea` wins at high `T`
//!   (Section 2).
//! * **Apparent-activation-energy / optimum-temperature of a LH surface step** —
//!   the observed rate `r(T) = k_surf(T)·θ_A(T)` combines an *increasing*
//!   Arrhenius factor with a *decreasing* (exothermic) coverage, producing the
//!   classic rate maximum as a function of temperature (Section 3).
//!
//! All scenarios use only the public kinetics + reaction-network + ODE API and
//! run deterministically in milliseconds. Run with:
//! `cargo run --example langmuir_hinshelwood -p tpt-sci-kinetics`

use tpt_sci_kinetics::{
    ArrheniusRate, KineticsError, KineticsProblem, R_GAS, langmuir_hinshelwood_coverages,
};
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::{ReactionNetwork, ReactionSystem};

/// Adsorption equilibrium constant `K(T) = exp(ΔS/R)·exp(-ΔH/(R·T))` for an
/// exothermic (`ΔH < 0`), entropy-losing (`ΔS < 0`) adsorbate.
fn adsorption_k(delta_h: f64, delta_s: f64, t: f64) -> f64 {
    (delta_s / R_GAS).exp() * (-delta_h / (R_GAS * t)).exp()
}

fn main() {
    println!("=== tpt-sci-kinetics: Langmuir–Hinshelwood surface kinetics ===\n");

    // -------------------------------------------------------------------
    // 1. Adsorption isotherm θ_A(p): Henry law at low p, saturation at high p.
    // -------------------------------------------------------------------
    // For a single adsorbate the closed form is θ = K·p / (1 + K·p). We verify
    // the API matches it across three decades of pressure, and that θ saturates
    // to 1 (the site budget) as p → ∞.
    let k_ads = 2.0; // adsorption equilibrium constant (K)
    println!("1) Langmuir isotherm  θ_A(p) = K·p/(1+K·p)   (K = {k_ads})");
    println!(
        "   {:>10} {:>10} {:>10} {:>10}",
        "p", "θ_A", "closed", "1-θ_A"
    );

    let mut prev = -1.0_f64;
    for &p in &[0.01, 0.05, 0.2, 1.0, 5.0, 50.0, 1000.0] {
        let theta = langmuir_hinshelwood_coverages(&[k_ads], &[p]).unwrap();
        let t = theta[0];
        let closed = k_ads * p / (1.0 + k_ads * p);
        // Monotonic increase and hard saturation at the site budget.
        assert!(t >= prev - 1e-12, "θ must be non-decreasing in p");
        assert!(t <= 1.0 + 1e-12, "θ cannot exceed the site budget");
        assert!((t - closed).abs() < 1e-12, "θ must match the closed form");
        prev = t;
        println!(
            "   {:>10.3} {:>10.4} {:>10.4} {:>10.4}",
            p,
            t,
            closed,
            1.0 - t
        );
    }

    // Henry (dilute) limit: θ ≈ K·p, i.e. linear in pressure.
    let p_h = 1.0e-4;
    let theta_h = langmuir_hinshelwood_coverages(&[k_ads], &[p_h]).unwrap()[0];
    assert!((theta_h - k_ads * p_h).abs() < 1e-6);
    // Saturation limit: at very high p the bare fraction (1-θ) is negligible.
    let theta_sat = langmuir_hinshelwood_coverages(&[k_ads], &[1.0e6]).unwrap()[0];
    assert!(theta_sat > 0.999_999);

    // Exercise the validation path: mismatched K/p lengths are rejected.
    assert!(matches!(
        langmuir_hinshelwood_coverages(&[1.0, 2.0], &[1.0]),
        Err(KineticsError::CoverageError(_))
    ));

    // -------------------------------------------------------------------
    // 2. Parallel surface network: selectivity flips with temperature.
    // -------------------------------------------------------------------
    // A* can follow two first-order paths:
    //   A* --k_B--> B*   (low Ea, modest pre-factor  -> favours B at low T)
    //   A* --k_C--> C*   (high Ea, large pre-factor  -> favours C at high T)
    // For parallel irreversible first-order channels the selectivity B/C equals
    // k_B/k_C exactly, so a temperature sweep reveals kinetic control.
    let rate_b = ArrheniusRate::new(1.0e9, 30_000.0).unwrap(); // low-Ea channel
    let rate_c = ArrheniusRate::new(1.0e16, 110_000.0).unwrap(); // high-Ea channel
    let kin = KineticsProblem::new(vec![rate_b, rate_c]).unwrap();
    assert_eq!(kin.rate_constants(800.0).len(), 2);

    let mut net: ReactionSystem = ReactionNetwork::from_dsl("k1, A --> B\nk2, A --> C").unwrap();
    let b_idx = net.species_index("B").unwrap();
    let c_idx = net.species_index("C").unwrap();

    println!("\n2) Parallel network A*→B* / A*→C* : selectivity B/C vs T");
    println!(
        "   {:>8} {:>10} {:>10} {:>10}",
        "T (K)", "k_B", "k_C", "B/C"
    );

    let mut sel_low = 0.0_f64;
    let mut sel_high = 0.0_f64;
    for &t in &[400.0, 600.0, 800.0, 1000.0] {
        let ks = kin.rate_constants(t);
        net.set_parameter("k1", ks[0]).unwrap();
        net.set_parameter("k2", ks[1]).unwrap();
        let y0 = net
            .initial_state(&[("A", 1.0), ("B", 0.0), ("C", 0.0)])
            .unwrap();
        let prob = net.to_ode_problem(&y0, 0.0).unwrap();
        let y = prob.solve(Method::Bdf, 1.0).unwrap();
        // Selectivity B/C (both channels deplete A, so the ratio is time-invariant).
        let sel = y[b_idx] / y[c_idx];
        assert!(y[b_idx].is_finite() && y[c_idx].is_finite());
        if t == 400.0 {
            sel_low = sel;
        }
        if t == 1000.0 {
            sel_high = sel;
        }
        println!(
            "   {:>8.0} {:>10.3e} {:>10.3e} {:>10.3e}",
            t, ks[0], ks[1], sel
        );
    }
    // Low-Ea path wins at low T, high-A/high-Ea path wins at high T.
    assert!(sel_low > 1.0, "low-Ea channel B should dominate at low T");
    assert!(
        sel_high < 1.0,
        "high-Ea channel C should dominate at high T"
    );

    // -------------------------------------------------------------------
    // 3. Observed LH rate r(T) = k_surf(T)·θ_A(T): a temperature optimum.
    // -------------------------------------------------------------------
    // A surface reaction whose rate is first order in the coverage of A* but
    // whose intrinsic barrier is Arrhenius. Because exothermic adsorption makes
    // θ_A(T) fall with T while k_surf(T) rises, r(T) passes through a maximum —
    // the textbook surface-kinetics "optimum temperature".
    let dh_a = -60_000.0; // exothermic adsorption (deeper than the barrier)
    let ds_a = -60.0; // entropy loss on binding
    let p_a = 1.0;
    // Ea < |ΔH| is required for an interior rate maximum: the rising Arrhenius
    // factor eventually loses to the collapsing coverage.
    let surf = ArrheniusRate::new(1.0e6, 40_000.0).unwrap();

    println!("\n3) Observed LH rate r(T) = k_surf(T)·θ_A(T)  (p_A = {p_a})");
    println!(
        "   {:>8} {:>10} {:>10} {:>10}",
        "T (K)", "k_surf", "θ_A", "r(T)"
    );

    let mut best_t = 0.0_f64;
    let mut best_r = 0.0_f64;
    let mut r_at_300 = 0.0_f64;
    let mut r_at_1200 = 0.0_f64;
    let mut prev_theta = f64::INFINITY;
    for t in (300..=1200).step_by(25) {
        let tk = t as f64;
        let k_a = adsorption_k(dh_a, ds_a, tk);
        let theta = langmuir_hinshelwood_coverages(&[k_a], &[p_a]).unwrap()[0];
        let r = surf.rate_constant(tk) * theta;
        assert!(
            theta <= prev_theta + 1e-12,
            "exothermic θ_A must fall with T"
        );
        prev_theta = theta;
        if r > best_r {
            best_r = r;
            best_t = tk;
        }
        if tk == 300.0 {
            r_at_300 = r;
        }
        if tk == 1200.0 {
            r_at_1200 = r;
        }
        println!(
            "   {:>8} {:>10.3e} {:>10.4} {:>10.3e}",
            tk,
            surf.rate_constant(tk),
            theta,
            r
        );
    }
    // There is a genuine interior optimum, bounded away from both endpoints.
    assert!(
        best_t > 300.0 && best_t < 1200.0,
        "rate optimum must be interior"
    );
    assert!(
        best_r > r_at_300 && best_r > r_at_1200,
        "optimum exceeds the endpoints"
    );
    println!(
        "   -> rate optimum near T = {:.0} K (rate-limited at low T, coverage-limited at high T)",
        best_t
    );

    println!("\nAll checks passed.");
}
