//! # Surface-kinetics reactor tour
//!
//! A guided tour of the [`tpt_sci_kinetics`] public surface, and of how it
//! plugs into the [`tpt_sci_reaction_network`] mass-action engine and the
//! [`tpt_sci_ode`] integrators it is built on.
//!
//! Catalysis / surface-reactor modelling needs two building blocks that plain
//! mass-action CRNs lack, and both live in this crate:
//!
//! * **Arrhenius temperature dependence** — `k(T) = A·exp(-Ea/(R·T))`, via
//!   [`ArrheniusRate`] and [`KineticsProblem`]. Watch the *exponential*
//!   sensitivity of a rate constant to temperature (Section 1).
//! * **Langmuir–Hinshelwood surface coverage** — fractional site coverages
//!   `θ_i = K_i·p_i / (1 + Σ K_j·p_j)` from an adsorption equilibrium, via
//!   [`langmuir_hinshelwood_coverages`]. The surface reaction rate is then
//!   first order in coverage rather than in gas concentration (Sections 2 & 4).
//!
//! What to observe when you run `cargo run --example reactor -p tpt-sci-kinetics`:
//! * A ~20–30× rate change for a modest 300 K temperature rise (Arrhenius).
//! * Coverage `θ_A` falling as temperature rises (adsorption is exothermic).
//! * A gas-phase reactant `A` being consumed through the coverage-dependent
//!   surface step, with conversion and `θ_A` reported over time.
//!
//! Everything is deterministic and runs in milliseconds.

use tpt_sci_kinetics::{
    ArrheniusRate, KineticsError, KineticsProblem, R_GAS, langmuir_hinshelwood_coverages,
};
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::{RateLaw, ReactionNetwork, ReactionSystem};

/// Adsorption equilibrium constant `K(T) = exp(ΔS/R)·exp(-ΔH/(R·T))` for a
/// (exothermic, `ΔH < 0`) adsorbate losing entropy (`ΔS < 0`) on binding.
fn adsorption_k(delta_h: f64, delta_s: f64, t: f64) -> f64 {
    (delta_s / R_GAS).exp() * (-delta_h / (R_GAS * t)).exp()
}

fn main() {
    println!("=== tpt-sci-kinetics: surface reactor tour ===\n");

    // -------------------------------------------------------------------
    // 1. Arrhenius temperature sensitivity (pure kinetics API).
    // -------------------------------------------------------------------
    // Surface step A* -> B* with a high activation barrier.
    let surface_step = ArrheniusRate::new(1.0e2, 50_000.0).unwrap();
    let t_low = 600.0;
    let t_high = 900.0;
    let k_low = surface_step.rate_constant(t_low);
    let k_high = surface_step.rate_constant(t_high);

    // The Arrhenius form predicts k(T2)/k(T1) = exp[ Ea/R · (1/T1 - 1/T2) ].
    let predicted_ratio = (surface_step.ea / R_GAS * (1.0 / t_low - 1.0 / t_high)).exp();

    println!(
        "1) Arrhenius surface step  (A = {:.0e}, Ea = {} J/mol)",
        surface_step.a, surface_step.ea as i32
    );
    println!("   k({t_low:.0} K) = {k_low:.3e}");
    println!("   k({t_high:.0} K) = {k_high:.3e}");
    println!(
        "   k(T2)/k(T1) = {:.3e}  (closed form = {:.3e})",
        k_high / k_low,
        predicted_ratio
    );

    assert!(
        k_high > k_low,
        "an Arrhenius rate must rise with temperature"
    );
    assert!(k_low.is_finite() && k_high.is_finite());
    assert!((k_high / k_low - predicted_ratio).abs() / predicted_ratio < 1e-9);

    // Exercise the validation path: bad A / negative Ea are rejected.
    assert!(matches!(
        ArrheniusRate::new(0.0, 1.0),
        Err(KineticsError::InvalidRate(_))
    ));
    assert!(matches!(
        ArrheniusRate::new(1.0, -1.0),
        Err(KineticsError::InvalidRate(_))
    ));

    // -------------------------------------------------------------------
    // 2. Langmuir–Hinshelwood coverage at two temperatures.
    // -------------------------------------------------------------------
    // Two adsorbates A* and B* compete for one site type. Adsorption is
    // exothermic, so lowering T strengthens adsorption and raises coverage.
    let (dh_a, ds_a) = (-40_000.0, -50.0);
    let (dh_b, ds_b) = (-30_000.0, -40.0);
    let pressures = [1.0_f64, 0.3]; // partial pressures of A and B

    let k_a_low = adsorption_k(dh_a, ds_a, t_low);
    let k_b_low = adsorption_k(dh_b, ds_b, t_low);
    let theta_low = langmuir_hinshelwood_coverages(&[k_a_low, k_b_low], &pressures).unwrap();

    let k_a_high = adsorption_k(dh_a, ds_a, t_high);
    let k_b_high = adsorption_k(dh_b, ds_b, t_high);
    let theta_high = langmuir_hinshelwood_coverages(&[k_a_high, k_b_high], &pressures).unwrap();

    println!("\n2) Langmuir–Hinshelwood coverages (A, B) at p = {pressures:?}");
    println!(
        "   {t_low:.0} K: θ_A = {:.3}, θ_B = {:.3}, bare = {:.3}",
        theta_low[0],
        theta_low[1],
        1.0 - theta_low[0] - theta_low[1]
    );
    println!(
        "   {t_high:.0} K: θ_A = {:.3}, θ_B = {:.3}, bare = {:.3}",
        theta_high[0],
        theta_high[1],
        1.0 - theta_high[0] - theta_high[1]
    );

    for theta in [&theta_low, &theta_high] {
        let sum: f64 = theta.iter().sum();
        assert!(sum <= 1.0 + 1e-9, "coverages cannot exceed the site budget");
        assert!(theta.iter().all(|&x| x.is_finite() && x >= 0.0));
    }
    // Exothermic adsorption => more coverage at lower temperature.
    assert!(theta_low[0] > theta_high[0]);

    // -------------------------------------------------------------------
    // 3. KineticsProblem binds Arrhenius rates to a reaction network.
    // -------------------------------------------------------------------
    // A gas-phase decay A -> B whose (mass-action) rate constant is itself an
    // Arrhenius expression resolved through KineticsProblem and then written
    // into the network as a parameter. This is the canonical "kinetics plug
    // into the reaction-network / ODE machinery" path.
    let t_op = 800.0;
    let decay = ArrheniusRate::new(1.0e6, 45_000.0).unwrap();
    let kin = KineticsProblem::new(vec![decay]).unwrap();
    let k_decay = kin.rate_constants(t_op)[0];
    assert!(k_decay > 0.0 && k_decay.is_finite());

    let mut net: ReactionSystem = ReactionNetwork::from_dsl("k, A --> B").unwrap();
    net.set_parameter("k", k_decay).unwrap();
    let y0_dec = net.initial_state(&[("A", 1.0), ("B", 0.0)]).unwrap();
    let prob_dec = net.to_ode_problem(&y0_dec, 0.0).unwrap();
    let y_dec = prob_dec.solve(Method::Tsit45, 1.0e-3).unwrap();
    let conv_dec = (1.0 - y_dec[0]) / 1.0;

    println!(
        "\n3) KineticsProblem -> reaction network (A -> B, Arrhenius k @ {t_op:.0} K = {k_decay:.3e})"
    );
    println!("   conversion after 1e-3 s = {conv_dec:.4}");
    assert!(conv_dec > 0.0 && conv_dec < 1.0);

    // -------------------------------------------------------------------
    // 4. Full Langmuir–Hinshelwood mechanism integrated over time.
    // -------------------------------------------------------------------
    // The *observed* rate of A consumption on the catalyst is
    //   r = k_surf(T) · θ_A,   θ_A = K_A·p_A / (1 + K_A·p_A + K_B·p_B),
    // with p_i ∝ [i]. We encode this as a custom rate law on a two-species
    // ReactionSystem and integrate it with tpt-sci-ode, then recompute the
    // coverage from the instantaneous concentrations at each output time.
    let k_surf = surface_step.rate_constant(t_op);
    let k_a = adsorption_k(dh_a, ds_a, t_op); // matches the K_A used below
    let k_b = adsorption_k(dh_b, ds_b, t_op); // matches the K_B used below

    let mut lh = ReactionNetwork::new();
    let a = lh.species("A");
    let b = lh.species("B");
    lh.parameter("k_surf", k_surf);
    lh.parameter("K_A", k_a);
    lh.parameter("K_B", k_b);
    lh.reaction(
        &[(a, 1.0)],
        &[(b, 1.0)],
        RateLaw::custom(|y, p| {
            let pa = y[0].max(0.0);
            let pb = y[1].max(0.0);
            let denom = 1.0 + p[1] * pa + p[2] * pb;
            p[0] * (p[1] * pa / denom) // k_surf · θ_A
        }),
    );
    let sys = lh.build().unwrap();

    // Inspect the compiled system (a slice of the reaction-network surface).
    println!("\n4) Langmuir–Hinshelwood mechanism integrated over time");
    println!(
        "   species = {:?}, n_reactions = {}",
        sys.species_names(),
        sys.n_reactions()
    );
    println!("   params  = {:?}", sys.parameter_names());
    println!(
        "   stoichiometry S (species × reaction) = {:?}",
        sys.stoichiometry_matrix()
    );
    let a_idx = sys.species_index("A").unwrap();
    let b_idx = sys.species_index("B").unwrap();
    let y0_lh = sys.initial_state(&[("A", 1.0), ("B", 0.0)]).unwrap();
    println!(
        "   r(y0) = {:?}  (via ReactionSystem::reaction_rates)",
        sys.reaction_rates(&y0_lh)
    );

    // Verify the ODE RHS matches the custom rate at the initial state.
    let mut dydt = vec![0.0; sys.n_species()];
    sys.eval_rhs(&y0_lh, &mut dydt);
    assert!((dydt[a_idx] + k_surf * (k_a * y0_lh[0] / (1.0 + k_a * y0_lh[0]))).abs() < 1e-9);

    // Sample strictly beyond t0 = 0 (solve_dense requires t_eval > t0); print
    // the initial state explicitly, then read conversion + coverage off the
    // integrated trajectory.
    println!("\n   {:>6} {:>10} {:>10} {:>10}", "t", "conv", "θ_A", "θ_B");
    let theta0 =
        langmuir_hinshelwood_coverages(&[k_a, k_b], &[y0_lh[a_idx], y0_lh[b_idx]]).unwrap();
    println!(
        "   {:>6.1} {:>10.4} {:>10.4} {:>10.4}",
        0.0, 0.0, theta0[0], theta0[1]
    );

    let oprob = sys.to_ode_problem(&y0_lh, 0.0).unwrap();
    let t_eval = [10.0, 25.0, 50.0, 100.0];
    let traj = oprob.solve_dense(Method::Tsit45, &t_eval).unwrap();
    assert_eq!(traj.len(), t_eval.len());

    let mut prev_conv = 0.0;
    for (t, y) in t_eval.iter().zip(traj.iter()) {
        assert!(
            y.iter().all(|v| v.is_finite()),
            "integration produced a finite state"
        );
        let theta = langmuir_hinshelwood_coverages(&[k_a, k_b], &[y[a_idx], y[b_idx]]).unwrap();
        let conv = (y0_lh[a_idx] - y[a_idx]) / y0_lh[a_idx];
        assert!(
            (-1e-9..=1.0 + 1e-9).contains(&conv),
            "conversion stays in [0, 1]"
        );
        assert!(
            conv >= prev_conv - 1e-9,
            "conversion is monotonic non-decreasing"
        );
        prev_conv = conv;
        println!(
            "   {:>6.1} {:>10.4} {:>10.4} {:>10.4}",
            t, conv, theta[0], theta[1]
        );
    }

    println!("\nAll checks passed.");
}
