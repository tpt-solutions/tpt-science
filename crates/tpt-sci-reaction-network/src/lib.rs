//! # tpt-sci-reaction-network
//!
//! A **species / rate / stoichiometry DSL for compartmental models** — the
//! Rust equivalent of Julia's [Catalyst.jl](https://github.com/SciML/Catalyst.jl)
//! — built from scratch for the `tpt-science` pillar.
//!
//! Given a chemical reaction network (species, reactions, and rate laws), the
//! crate compiles the **law of mass action** into the deterministic ODE
//!
//! ```text
//! dy/dt = S · r(y, p)
//! ```
//!
//! where `S` is the stoichiometry matrix and `r` the per-reaction rate vector,
//! and hands the result to [`tpt_sci_ode`] for integration. It also exposes the
//! stoichiometry matrix and rate vector directly for inspection, parameter
//! estimation, or coupling into a larger multi-scale model.
//!
//! In addition to the deterministic mass-action ODE, the crate provides exact
//! stochastic trajectories via [`ReactionSystem::simulate_ssa`] (Gillespie's
//! direct method), so a network can be studied in either the deterministic or
//! stochastic regime from the same IR.
//!
//! This is the **v1** scope from `spec.txt` / `registry.toml`: a CRN IR plus a
//! mass-action ODE builder and a stochastic SSA backend. Still out of scope for
//! v1 (documented, not built): SDE/jump models, SBML I/O, network analysis, and
//! conservation-law elimination. (The previous plan deferred the stochastic
//! backend to wrapping `rebop`; it is now implemented from scratch instead.)
//!
//! ## Building models
//!
//! Two equivalent ways to define a model:
//!
//! 1. **Programmatically** with [`ReactionNetwork`], registering species and
//!    parameters by name and appending reactions with explicit indices.
//! 2. **Textually** with the Catalyst.jl-style [`ReactionNetwork::from_dsl`],
//!    e.g. `kB, S + E --> SE`.
//!
//! ## Example
//!
//! Michaelis–Menten enzyme kinetics (`S + E ⇌ SE → P + E`):
//!
//! ```
//! use tpt_sci_ode::Method;
//! use tpt_sci_reaction_network::ReactionNetwork;
//!
//! let model = ReactionNetwork::from_dsl(
//!     "kB, S + E --> SE
//!      kD, SE --> S + E
//!      kP, SE --> P + E",
//! ).unwrap();
//! let mut model = model;
//! model.set_parameter("kB", 0.01).unwrap();
//! model.set_parameter("kD", 0.1).unwrap();
//! model.set_parameter("kP", 0.1).unwrap();
//!
//! // E + SE is conserved: total enzyme stays at its initial 10.
//! let y0 = model.initial_state(&[("S", 50.0), ("E", 10.0)]).unwrap();
//! let prob = model.to_ode_problem(&y0, 0.0).unwrap();
//! let y = prob.solve(Method::Bdf, 200.0).unwrap();
//!
//! let e = model.species_index("E").unwrap();
//! let se = model.species_index("SE").unwrap();
//! assert!((y[e] + y[se] - 10.0).abs() < 1e-6);
//! ```
//!
//! Licensed under either of MIT or Apache-2.0 at your option.

pub mod error;
pub mod model;
pub mod ode;
pub mod ssa;

pub use error::ReactionNetworkError;
pub use model::{RateLaw, Reaction, ReactionNetwork, ReactionSystem, Term};
pub use ssa::SsaTrajectory;

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use tpt_sci_ode::Method;

    fn sir_builder() -> ReactionNetwork {
        let mut net = ReactionNetwork::new();
        let s = net.species("S");
        let i = net.species("I");
        let r = net.species("R");
        net.parameter("beta", 0.0);
        net.parameter("gamma", 0.0);
        net.reaction(
            &[(s, 1.0), (i, 1.0)],
            &[(i, 2.0)],
            RateLaw::mass_action("beta"),
        );
        net.reaction(&[(i, 1.0)], &[(r, 1.0)], RateLaw::mass_action("gamma"));
        net
    }

    #[test]
    fn stoichiometry_matrix_sir() {
        let sys = sir_builder().build().unwrap();
        let s = sys.stoichiometry_matrix();
        // columns are reaction net-change vectors; rows are species S, I, R
        assert_eq!(s, vec![vec![-1.0, 0.0], vec![1.0, -1.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn dsl_matches_builder_sir() {
        let built = sir_builder().build().unwrap();
        let dsl = ReactionNetwork::from_dsl(
            "beta, S + I --> 2 I
             gamma, I --> R",
        )
        .unwrap();
        assert_eq!(built.species_names(), dsl.species_names());
        assert_eq!(built.stoichiometry_matrix(), dsl.stoichiometry_matrix());
        assert_eq!(built.parameter_names(), dsl.parameter_names());
    }

    #[test]
    fn mass_action_rates_dimerization() {
        let mut net = ReactionNetwork::new();
        let a = net.species("A");
        let b = net.species("B");
        net.parameter("kf", 0.5);
        net.parameter("kr", 0.25);
        net.reaction(&[(a, 2.0)], &[(b, 1.0)], RateLaw::mass_action("kf"));
        net.reaction(&[(b, 1.0)], &[(a, 2.0)], RateLaw::mass_action("kr"));
        let sys = net.build().unwrap();
        // r_fwd = kf * A^2, r_rev = kr * B
        let rates = sys.reaction_rates(&[3.0, 4.0]);
        assert_abs_diff_eq!(rates[0], 0.5 * 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rates[1], 0.25 * 4.0, epsilon = 1e-12);
    }

    #[test]
    fn linear_production_decay_analytic() {
        // dA/dt = k1 - k2 A  ->  A(t) = k1/k2 + (A0 - k1/k2) e^{-k2 t}
        let mut model = ReactionNetwork::from_dsl(
            "k1, 0 --> A
             k2, A --> 0",
        )
        .unwrap();
        model.set_parameter("k1", 2.0).unwrap();
        model.set_parameter("k2", 1.0).unwrap();
        let y0 = model.initial_state(&[("A", 0.0)]).unwrap();
        let prob = model.to_ode_problem(&y0, 0.0).unwrap();
        // Use Esdirk34 (3rd order, stiff-capable) for better accuracy
        let y = prob.solve(Method::Esdirk34, 1.0).unwrap();
        let expected = 2.0 * (1.0 - (-1.0_f64).exp());
        assert_abs_diff_eq!(y[0], expected, epsilon = 1e-4);
    }

    #[test]
    fn reversible_dimer_conserves_a_plus_2b() {
        let mut model = ReactionNetwork::from_dsl(
            "kf, 2 A --> B
             kr, B --> 2 A",
        )
        .unwrap();
        model.set_parameter("kf", 1.0).unwrap();
        model.set_parameter("kr", 0.5).unwrap();
        let y0 = model.initial_state(&[("A", 1.0), ("B", 0.0)]).unwrap();
        let prob = model.to_ode_problem(&y0, 0.0).unwrap();
        let y = prob.solve(Method::Esdirk34, 5.0).unwrap();
        assert_abs_diff_eq!(y[0] + 2.0 * y[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn michaelis_menten_conserves_enzyme_and_substrate() {
        let mut model = ReactionNetwork::from_dsl(
            "kB, S + E --> SE
             kD, SE --> S + E
             kP, SE --> P + E",
        )
        .unwrap();
        model.set_parameter("kB", 0.01).unwrap();
        model.set_parameter("kD", 0.1).unwrap();
        model.set_parameter("kP", 0.1).unwrap();
        let y0 = model.initial_state(&[("S", 50.0), ("E", 10.0)]).unwrap();
        let prob = model.to_ode_problem(&y0, 0.0).unwrap();
        let y = prob.solve(Method::Esdirk34, 200.0).unwrap();

        let e = model.species_index("E").unwrap();
        let se = model.species_index("SE").unwrap();
        let s = model.species_index("S").unwrap();
        let p = model.species_index("P").unwrap();
        // enzyme conserved
        assert_abs_diff_eq!(y[e] + y[se], 10.0, epsilon = 1e-6);
        // substrate conserved
        assert_abs_diff_eq!(y[s] + y[se] + y[p], 50.0, epsilon = 1e-6);
        // product was formed
        assert!(y[p] > 0.0);
    }

    #[test]
    fn custom_rate_law_linear() {
        let mut net = ReactionNetwork::new();
        let a = net.species("A");
        net.parameter("k2", 1.0);
        // constant production via a custom rate law
        net.reaction(&[], &[(a, 1.0)], RateLaw::custom(|_y, _p| 2.0));
        net.reaction(&[(a, 1.0)], &[], RateLaw::mass_action("k2"));
        let sys = net.build().unwrap();
        let prob = sys.to_ode_problem(&[0.0], 0.0).unwrap();
        // Use Esdirk34 (3rd order, stiff-capable) for better accuracy
        let y = prob.solve(Method::Esdirk34, 1.0).unwrap();
        let expected = 2.0 * (1.0 - (-1.0_f64).exp());
        assert_abs_diff_eq!(y[0], expected, epsilon = 1e-4);
    }

    #[test]
    fn empty_reaction_rejected() {
        let mut net = ReactionNetwork::new();
        let a = net.species("A");
        net.reaction(&[], &[], RateLaw::mass_action("k"));
        assert!(matches!(
            net.build(),
            Err(ReactionNetworkError::EmptyReaction)
        ));
        // DSL: empty both sides
        assert!(ReactionNetwork::from_dsl("k,  --> ").is_err());
        let _ = a;
    }

    #[test]
    fn undefined_rate_constant_rejected() {
        let mut net = ReactionNetwork::new();
        let a = net.species("A");
        net.reaction(&[], &[(a, 1.0)], RateLaw::mass_action("missing"));
        assert!(matches!(
            net.build(),
            Err(ReactionNetworkError::UndefinedRateConstant(_))
        ));
    }

    #[test]
    fn state_dimension_mismatch_rejected() {
        let mut model = ReactionNetwork::from_dsl("k, A --> B").unwrap();
        model.set_parameter("k", 1.0).unwrap();
        // y0 has the wrong length (2 instead of 1 species)
        assert!(matches!(
            model.to_ode_problem(&[0.0], 0.0),
            Err(ReactionNetworkError::StateDimension { .. })
        ));
    }

    #[test]
    fn unknown_species_in_initial_state_rejected() {
        let model = ReactionNetwork::from_dsl("k, A --> B").unwrap();
        assert!(matches!(
            model.initial_state(&[("Z", 1.0)]),
            Err(ReactionNetworkError::UnknownSpecies(_))
        ));
    }

    #[test]
    fn sir_population_conserved_under_integration() {
        let mut model = sir_builder().build().unwrap();
        model.set_parameter("beta", 1e-4).unwrap();
        model.set_parameter("gamma", 0.01).unwrap();
        let y0 = model
            .initial_state(&[("S", 999.0), ("I", 1.0), ("R", 0.0)])
            .unwrap();
        let prob = model.to_ode_problem(&y0, 0.0).unwrap();
        let y = prob.solve(Method::Bdf, 250.0).unwrap();
        // S + I + R == N is conserved in the closed SIR model
        assert_abs_diff_eq!(y[0] + y[1] + y[2], 1000.0, epsilon = 1e-6);
        // susceptibles only ever decrease
        assert!(y[0] <= 999.0 + 1e-9);
        // recovered only ever increases
        assert!(y[2] >= -1e-9);
    }
}
