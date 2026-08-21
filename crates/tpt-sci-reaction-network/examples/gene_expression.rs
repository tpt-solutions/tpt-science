//! # Gene expression — a non-SIR reaction network in `tpt-sci-reaction-network`.
//!
//! The SIR example built an epidemic compartmental model. This example instead
//! builds a small **gene-expression** network — the canonical stochastic
//! biochemical circuit — and shows how the same `tpt-sci-reaction-network` DSL
//! compiles law-of-mass-action kinetics into both a deterministic ODE and an
//! exact stochastic (Gillespie SSA) trajectory.
//!
//! The circuit models one gene that can be *off* (`G`) or *on* (`Gstar`), which
//! when active transcribes messenger RNA (`M`); mRNA is translated into protein
//! (`P`) and both degrade:
//!
//! ```text
//!     k_on,  G    --> Gstar        (activation)
//!     k_off, Gstar --> G           (deactivation)
//!     k_t,   Gstar --> Gstar + M   (transcription)
//!     k_dm,  M     --> 0           (mRNA degradation)
//!     k_tl,  M     --> M + P       (translation)
//!     k_dp,  P     --> 0           (protein degradation)
//! ```
//!
//! # What this example exercises
//!
//! * building the network with the Catalyst.jl-style textual [`ReactionNetwork`]
//!   DSL, and confirming the programmatic builder produces the *identical*
//!   intermediate representation (species order, stoichiometry matrix),
//! * the steady-state mass-action ODE `dy/dt = S·r` solved through the
//!   `tpt-sci-ode` bridge, compared against the *closed-form* steady state that
//!   the conservation + mass-action balance predicts,
//! * a **parameter scan** over the transcription rate `k_t`, demonstrating the
//!   linear dependence of steady-state protein on transcription — a genuinely
//!   different use case from SIR's epidemic curve,
//! * the conserved total gene copy number `G + Gstar = 1`,
//! * an exact stochastic [`ReactionSystem::simulate_ssa`] trajectory whose mean
//!   steady-state protein tracks the deterministic prediction.
//!
//! # What to observe in the output
//!
//! * total gene `G + Gstar` stays at `1.0` exactly (one gene copy is conserved),
//! * the deterministic steady-state protein matches the analytic
//!   `P_ss = k_tl·k_t·(k_on/(k_on+k_off)) / (k_dm·k_dp)` to a fraction of a
//!   percent,
//! * doubling `k_t` doubles `P_ss` (the parameter scan confirms the linear law),
//! * the stochastic mean tracks the deterministic steady state within noise.

use tpt_sci_ode::Method;
use tpt_sci_reaction_network::{RateLaw, ReactionNetwork, ReactionSystem};

// One gene copy; realistic rate constants (per arbitrary time unit).
const K_ON: f64 = 0.1; // gene activation
const K_OFF: f64 = 0.2; // gene deactivation
const K_T: f64 = 5.0; // transcription rate (scanned below)
const K_DM: f64 = 1.0; // mRNA degradation
const K_TL: f64 = 0.1; // translation
const K_DP: f64 = 0.05; // protein degradation

/// Build the gene-expression network from the textual DSL.
fn build_dsl(kt: f64) -> ReactionSystem {
    let mut sys = ReactionNetwork::from_dsl(
        "k_on,  G     --> Gstar
         k_off, Gstar --> G
         k_t,   Gstar --> Gstar + M
         k_dm,  M     --> 0
         k_tl,  M     --> M + P
         k_dp,  P     --> 0",
    )
    .expect("DSL gene-expression model should parse");
    sys.set_parameter("k_on", K_ON).unwrap();
    sys.set_parameter("k_off", K_OFF).unwrap();
    sys.set_parameter("k_t", kt).unwrap();
    sys.set_parameter("k_dm", K_DM).unwrap();
    sys.set_parameter("k_tl", K_TL).unwrap();
    sys.set_parameter("k_dp", K_DP).unwrap();
    sys
}

/// Build the identical network with the programmatic builder (used only to
/// confirm the two compilers agree on the IR).
fn build_programmatic() -> ReactionSystem {
    let mut net = ReactionNetwork::new();
    let g = net.species("G");
    let gstar = net.species("Gstar");
    let m = net.species("M");
    let p = net.species("P");
    net.parameter("k_on", K_ON);
    net.parameter("k_off", K_OFF);
    net.parameter("k_t", K_T);
    net.parameter("k_dm", K_DM);
    net.parameter("k_tl", K_TL);
    net.parameter("k_dp", K_DP);
    net.reaction(&[(g, 1.0)], &[(gstar, 1.0)], RateLaw::mass_action("k_on"));
    net.reaction(&[(gstar, 1.0)], &[(g, 1.0)], RateLaw::mass_action("k_off"));
    net.reaction(
        &[(gstar, 1.0)],
        &[(gstar, 1.0), (m, 1.0)],
        RateLaw::mass_action("k_t"),
    );
    net.reaction(&[(m, 1.0)], &[], RateLaw::mass_action("k_dm"));
    net.reaction(
        &[(m, 1.0)],
        &[(m, 1.0), (p, 1.0)],
        RateLaw::mass_action("k_tl"),
    );
    net.reaction(&[(p, 1.0)], &[], RateLaw::mass_action("k_dp"));
    net.build()
        .expect("programmatic gene-expression model should build")
}

/// Deterministic steady-state protein `P_ss = k_tl·k_t·(k_on/(k_on+k_off))/k_dm/k_dp`.
fn analytic_p_ss(kt: f64) -> f64 {
    let gstar_ss = K_ON / (K_ON + K_OFF); // fraction of time the gene is on
    (K_TL * kt * gstar_ss) / (K_DM * K_DP)
}

/// Integrate the deterministic network to steady state with one `k_t`.
fn steady_state(kt: f64) -> f64 {
    let sys = build_dsl(kt);
    let y0 = sys.initial_state(&[("G", 1.0)]).unwrap();
    let prob = sys.to_ode_problem(&y0, 0.0).expect("build ODE problem");
    let y = prob
        .solve(Method::Bdf, 300.0)
        .expect("integrate to steady state");
    let g = sys.species_index("G").unwrap();
    let gstar = sys.species_index("Gstar").unwrap();
    let p = sys.species_index("P").unwrap();
    // Conservation: exactly one gene copy.
    assert!(
        (y[g] + y[gstar] - 1.0).abs() < 1e-6,
        "total gene G + Gstar not conserved: {} + {}",
        y[g],
        y[gstar]
    );
    assert!(y[p] > 0.0, "protein should be produced");
    y[p]
}

fn main() {
    println!("=== tpt-sci-reaction-network: gene expression ===\n");

    // --- 1. Two compilers must agree on the IR -------------------------------
    let built = build_programmatic();
    let dsl = build_dsl(K_T);
    assert_eq!(built.species_names(), dsl.species_names());
    assert_eq!(built.parameter_names(), dsl.parameter_names());
    assert_eq!(built.stoichiometry_matrix(), dsl.stoichiometry_matrix());
    println!("programmatic builder and textual DSL produce identical IR");
    println!("species : {:?}", dsl.species_names());
    println!("params  : {:?}", dsl.parameter_names());

    let s = dsl.stoichiometry_matrix();
    println!("\nstoichiometry matrix S (rows = species, cols = reactions):");
    for (sp, row) in dsl.species_names().iter().zip(s.iter()) {
        println!("  {sp:>5} : {row:?}");
    }

    // --- 2. Deterministic steady state vs closed form ------------------------
    let p_ss = steady_state(K_T);
    let p_exact = analytic_p_ss(K_T);
    let rel = (p_ss - p_exact).abs() / p_exact;
    println!(
        "\n[deterministic] k_t = {K_T}: steady P = {p_ss:.4}, analytic = {p_exact:.4}, rel err = {rel:.3e}"
    );
    assert!(
        rel < 5e-3,
        "steady-state protein must match the analytic form"
    );

    // --- 3. Parameter scan: P_ss ∝ k_t (linear transcription law) ------------
    println!("\n[parameter scan] steady-state protein vs transcription rate k_t:");
    println!("  {:>6}  {:>10}  {:>12}", "k_t", "P_ss", "P_ss/k_t");
    let mut ratios = Vec::new();
    for &kt in &[2.0_f64, 5.0, 10.0] {
        let p = steady_state(kt);
        let ratio = p / kt;
        ratios.push(ratio);
        println!("  {kt:>6.1}  {:>10.4}  {:>12.4}", p, ratio);
        assert!((p - analytic_p_ss(kt)).abs() / analytic_p_ss(kt) < 5e-3);
    }
    // All three ratios must agree: P_ss is linear in k_t.
    let spread = ratios.iter().cloned().fold(0.0_f64, f64::max)
        - ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread / ratios[0] < 1e-2,
        "P_ss must scale linearly with k_t"
    );

    // --- 4. Exact stochastic SSA: mean tracks the deterministic steady state --
    let sys = build_dsl(K_T);
    let y0 = sys.initial_state(&[("G", 1.0)]).unwrap();
    let realizations = 60;
    let mut rng = SplitMix64::new(0xDEAD_BEEF);
    let mut final_ps = Vec::with_capacity(realizations);
    let p_idx = sys.species_index("P").unwrap();
    let g_idx = sys.species_index("G").unwrap();
    let gstar_idx = sys.species_index("Gstar").unwrap();
    for _ in 0..realizations {
        let traj = sys
            .simulate_ssa(&y0, 300.0, &mut || rng.next_f64())
            .expect("SSA simulate");
        for st in &traj.states {
            debug_assert!((st[g_idx] + st[gstar_idx] - 1.0).abs() < 1e-9);
        }
        final_ps.push(traj.final_state()[p_idx]);
    }
    let mean_p: f64 = final_ps.iter().sum::<f64>() / realizations as f64;
    println!(
        "\n[stochastic SSA] mean steady P over {realizations} realizations = {mean_p:.4}  (deterministic {p_ss:.4})"
    );
    assert!(
        (mean_p - p_ss).abs() < 0.5,
        "SSA mean steady protein {mean_p:.3} too far from deterministic {p_ss:.3}"
    );

    println!(
        "\n=== gene-expression tour complete: deterministic, analytic, and stochastic agree ==="
    );
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
