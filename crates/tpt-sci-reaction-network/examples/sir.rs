//! # SIR epidemic — a tour of the `tpt-sci-reaction-network` public API
//!
//! The SIR model splits a closed population `N` into Susceptible (`S`),
//! Infectious (`I`), and Recovered (`R`):
//!
//! ```text
//! S + I --β--> 2 I      (infection, law of mass action)
//! I   --γ--> R          (recovery,  law of mass action)
//! ```
//!
//! This example is a *tour*: it builds the same model two equivalent ways (the
//! programmatic [`ReactionNetwork`] builder and the Catalyst.jl-style textual
//! DSL), dumps the compiled intermediate representation (species, parameters,
//! stoichiometry matrix, per-reaction rates, and the `dy/dt = S·r` right-hand
//! side — including a [`RateLaw::custom`] contrast), integrates it
//! deterministically through the `tpt-sci-ode` bridge, and finally compares
//! against exact stochastic Gillespie [`ReactionSystem::simulate_ssa`]
//! trajectories.
//!
//! What to observe:
//! - `S + I + R` stays equal to `N` to machine precision (conserved system),
//!   for both the deterministic and stochastic solvers.
//! - A clear infection peak followed by a long recovery tail.
//! - The stochastic mean final size tracks the deterministic one within noise.

use tpt_sci_ode::Method;
use tpt_sci_reaction_network::{RateLaw, ReactionNetwork, ReactionNetworkError, ReactionSystem};

// Deterministic parameters shared by both build paths.
const BETA: f64 = 1e-3;
const GAMMA: f64 = 0.05;
const N: f64 = 1000.0;
const T_MAX: f64 = 200.0;

fn main() {
    println!("=== SIR reaction-network API tour ===\n");

    // ---------------------------------------------------------------------
    // 1. Two equivalent ways to build the same model.
    // ---------------------------------------------------------------------
    let built = build_programmatic();
    let dsl = build_from_dsl();

    // The two compilers must agree on the intermediate representation.
    assert_eq!(built.species_names(), dsl.species_names());
    assert_eq!(built.parameter_names(), dsl.parameter_names());
    assert_eq!(built.stoichiometry_matrix(), dsl.stoichiometry_matrix());
    println!("programmatic builder and textual DSL produce identical IR");

    let sys = dsl; // use the DSL-compiled system for the rest of the tour

    // ---------------------------------------------------------------------
    // 2. Inspect the compiled intermediate representation.
    // ---------------------------------------------------------------------
    println!("\nspecies : {:?}", sys.species_names());
    println!("params  : {:?}", sys.parameter_names());
    println!(
        "rate beta = {}, gamma = {}",
        sys.parameter("beta").unwrap(),
        sys.parameter("gamma").unwrap()
    );

    // Stoichiometry matrix S[species][reaction] = net change when it fires.
    let s = sys.stoichiometry_matrix();
    println!("\nstoichiometry matrix S (rows = species, cols = reactions):");
    for (sp, row) in sys.species_names().iter().zip(s.iter()) {
        println!("  {sp:>3} : {row:?}");
    }
    assert_eq!(
        s,
        vec![
            vec![-1.0, 0.0], // S
            vec![1.0, -1.0], // I
            vec![0.0, 1.0],  // R
        ]
    );

    let y0 = sys
        .initial_state(&[("S", N - 1.0), ("I", 1.0), ("R", 0.0)])
        .unwrap();
    let rates = sys.reaction_rates(&y0);
    println!("\nreaction rates at t0: {rates:?}");
    let mut dydt = vec![0.0; sys.n_species()];
    sys.eval_rhs(&y0, &mut dydt);
    println!("dy/dt at t0        : {dydt:?}  (== S · r)");

    // ---------------------------------------------------------------------
    // 2b. Mass-action vs. a custom (non-mass-action) rate law.
    // ---------------------------------------------------------------------
    let mut custom = ReactionNetwork::new();
    let a = custom.species("A");
    custom.parameter("k", GAMMA);
    custom.reaction(&[(a, 1.0)], &[], RateLaw::mass_action("k"));
    custom.reaction(&[], &[(a, 1.0)], RateLaw::custom(|_y, p| p[0] * 2.0));
    let custom_sys = custom.build().expect("custom-law network should build");
    let cr = custom_sys.reaction_rates(&[5.0]);
    println!("\ncustom-law model rates at A=5: {cr:?}");
    assert!((cr[0] - GAMMA * 5.0).abs() < 1e-12);
    assert!((cr[1] - GAMMA * 2.0).abs() < 1e-12);

    // ---------------------------------------------------------------------
    // 3. Deterministic integration through the tpt-sci-ode bridge.
    // ---------------------------------------------------------------------
    let prob = sys.to_ode_problem(&y0, 0.0).expect("build ODE problem");
    let i = sys.species_index("I").unwrap();
    let s_idx = sys.species_index("S").unwrap();
    let r_idx = sys.species_index("R").unwrap();

    let times = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0, 150.0, 200.0];
    let mut peak = y0[i];
    let mut peak_t = 0.0_f64;
    println!("\ndeterministic trajectory (Esdirk34):");
    let mut y_end = y0.clone();
    for &t in &times {
        let y = prob.solve(Method::Esdirk34, t).expect("solve");
        y_end = y.clone();
        if y[i] > peak {
            peak = y[i];
            peak_t = t;
        }
        println!(
            "  t={t:6.1}  S={:8.2} I={:8.2} R={:8.2}",
            y[s_idx], y[i], y[r_idx]
        );
    }

    let final_s = y_end[s_idx];
    let final_i = y_end[i];
    let final_r = y_end[r_idx];

    // --- assertions on the deterministic solution ---
    let total = final_s + final_i + final_r;
    assert!(
        (total - N).abs() < 1e-4,
        "population not conserved: {total}"
    );
    assert!(peak > y0[i], "no infection peak observed");
    assert!(peak_t > 0.0, "peak should occur after t0");
    assert!(
        final_s <= y0[s_idx] + 1e-9,
        "susceptibles should not increase"
    );
    assert!(final_r >= y0[r_idx] - 1e-9, "recovered should not decrease");
    println!("\npeak infected = {peak:.2} at t = {peak_t:.1}");
    println!("final sizes   : S={final_s:.2} I={final_i:.2} R={final_r:.2}");

    // ---------------------------------------------------------------------
    // 4. Stochastic Gillespie trajectories via simulate_ssa.
    // ---------------------------------------------------------------------
    let y0_counts = sys
        .initial_state(&[("S", N - 1.0), ("I", 1.0), ("R", 0.0)])
        .unwrap();
    let realizations = 40;
    let mut rng = SplitMix64::new(0x1234_5678);
    let mut final_rs = Vec::with_capacity(realizations);
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for _ in 0..realizations {
        let traj = sys
            .simulate_ssa(&y0_counts, T_MAX, &mut || rng.next_f64())
            .expect("SSA simulate");
        // S + I + R is conserved at every event of the SSA chain.
        for st in &traj.states {
            debug_assert!((st[s_idx] + st[i] + st[r_idx] - N).abs() < 1e-9);
        }
        let fr = traj.final_state()[r_idx];
        final_rs.push(fr);
        min_r = min_r.min(fr);
        max_r = max_r.max(fr);
    }
    let mean_r: f64 = final_rs.iter().sum::<f64>() / realizations as f64;
    println!("\nstochastic SSA over {realizations} realizations (t_max = {T_MAX}):");
    println!("  mean final R = {mean_r:.2}  (min {min_r:.2}, max {max_r:.2})");

    // The stochastic mean final size must track the deterministic one.
    assert!(
        (mean_r - final_r).abs() < 50.0,
        "SSA mean final R {mean_r:.2} too far from deterministic {final_r:.2}"
    );

    // ---------------------------------------------------------------------
    // 5. Public error surface (ReactionNetworkError).
    // ---------------------------------------------------------------------
    demonstrate_errors();

    println!("\n=== tour complete: deterministic and stochastic agree, all assertions hold ===");
}

/// Build the SIR model with the programmatic [`ReactionNetwork`] builder.
fn build_programmatic() -> ReactionSystem {
    let mut net = ReactionNetwork::new();
    let s = net.species("S");
    let i = net.species("I");
    let r = net.species("R");
    net.parameter("beta", BETA);
    net.parameter("gamma", GAMMA);
    net.reaction(
        &[(s, 1.0), (i, 1.0)],
        &[(i, 2.0)],
        RateLaw::mass_action("beta"),
    );
    net.reaction(&[(i, 1.0)], &[(r, 1.0)], RateLaw::mass_action("gamma"));
    net.build().expect("programmatic SIR should build")
}

/// Build the SIR model with the textual Catalyst.jl-style DSL.
fn build_from_dsl() -> ReactionSystem {
    let mut sys = ReactionNetwork::from_dsl(
        "beta,  S + I --> 2 I
         gamma, I     --> R",
    )
    .expect("DSL SIR should parse");
    sys.set_parameter("beta", BETA).unwrap();
    sys.set_parameter("gamma", GAMMA).unwrap();
    sys
}

/// Exercise the [`ReactionNetworkError`] surface: undefined rate constants and
/// malformed DSL are rejected at build / parse time.
fn demonstrate_errors() {
    let mut net = ReactionNetwork::new();
    let a = net.species("A");
    net.reaction(&[(a, 1.0)], &[], RateLaw::mass_action("missing"));
    match net.build() {
        Err(ReactionNetworkError::UndefinedRateConstant(name)) => {
            println!("\nerror surface: caught UndefinedRateConstant({name})");
        }
        other => panic!("expected UndefinedRateConstant, got {other:?}"),
    }

    // An empty reaction (both sides blank) is rejected by the DSL parser.
    assert!(ReactionNetwork::from_dsl("k,  --> ").is_err());
    println!("error surface: caught Dsl parse error for empty reaction");
}

/// A tiny deterministic uniform RNG (`SplitMix64`) so the SSA tour does not
/// depend on an external RNG crate. Returns variates in `[0, 1)`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) / (1u64 << 53) as f64
    }
}
