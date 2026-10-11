//! # tpt-sci-kinetics
//!
//! Heterogeneous and surface **chemical kinetics** for the `tpt-science` pillar,
//! built from scratch on top of [`tpt_sci_reaction_network`]'s mass-action
//! engine ([`tpt_sci_ode`] backend).
//!
//! The crate adds the two kinetic building blocks most often needed for
//! catalysis / reactor modelling that plain mass-action CRNs lack:
//!
//! * **Arrhenius temperature dependence.** Each reaction carries a pre-exponential
//!   factor `A` and activation energy `Ea`; its rate constant follows
//!   `k(T) = A·exp(-Ea / (R·T))` ([`ArrheniusRate`]).
//! * **Langmuir–Hinshelwood adsorption / surface coverage.** A small site-balance
//!   solver computes fractional surface coverages `θ_i` from gas-phase partial
//!   pressures and adsorption equilibria, feeding surface-reaction rates that are
//!   first order in coverage rather than in gas concentration
//!   ([`langmuir_hinshelwood_coverages`]), generalized to multiple independent
//!   site types via [`multi_site_langmuir_hinshelwood_coverages`].
//! * **Eley–Rideal mechanism.** [`EleyRideal`] models a step where one
//!   reactant is adsorbed (subject to LH coverage) and the other reacts
//!   directly from the gas phase, `r = k·θ_adsorbed·[gas_phase]`, pluggable
//!   into a `ReactionSystem` via [`EleyRideal::into_rate_law`].
//! * **Coverage-dependent activation energy.** [`CoverageDependentArrheniusRate`]
//!   extends the Arrhenius law with a linear Brønsted–Evans–Polanyi-style
//!   dependence `Ea(θ) = Ea0 + α·θ`, giving `k(T, θ) = A·exp(-Ea(θ)/(R·T))`.
//!
//! An [`KineticsProblem`] ties an Arrhenius-rate
//! [`ReactionSystem`](tpt_sci_reaction_network::ReactionSystem) to a
//! temperature profile and integrates it with `tpt-sci-ode`.
//!
//! # Example
//!
//! ```
//! use tpt_sci_kinetics::{ArrheniusRate, langmuir_hinshelwood_coverages};
//!
//! // Unimolecular decay with A = 1e13, Ea = 80 kJ/mol.
//! let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
//! let k = r.rate_constant(800.0); // T = 800 K
//! assert!(k > 0.0 && k.is_finite());
//!
//! // Two competing adsorbates on a single site type; coverages sum to <= 1.
//! let theta = langmuir_hinshelwood_coverages(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
//! let sum: f64 = theta.iter().sum();
//! assert!(sum <= 1.0 + 1e-9 && sum > 0.0);
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::KineticsError;

mod arrhenius;
mod constants;
mod coverage;
mod coverage_dependent;
mod eley_rideal;
mod problem;

pub use arrhenius::ArrheniusRate;
pub use constants::R_GAS;
pub use coverage::{langmuir_hinshelwood_coverages, multi_site_langmuir_hinshelwood_coverages};
pub use coverage_dependent::CoverageDependentArrheniusRate;
pub use eley_rideal::EleyRideal;
pub use problem::KineticsProblem;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn arrhenius_rises_with_temperature() {
        let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
        let k_low = r.rate_constant(600.0);
        let k_high = r.rate_constant(900.0);
        assert!(k_high > k_low);
        assert!(k_low.is_finite() && k_high.is_finite());
    }

    #[test]
    fn arrhenius_rejects_invalid() {
        assert!(ArrheniusRate::new(0.0, 1.0).is_err());
        assert!(ArrheniusRate::new(1.0, -1.0).is_err());
    }

    #[test]
    fn langmuir_hinshelwood_sums_to_one() {
        let theta = langmuir_hinshelwood_coverages(&[2.0, 3.0], &[1.0, 1.0]).unwrap();
        // Adsorbed coverages sum to <= 1; the remainder is bare surface.
        let sum: f64 = theta.iter().sum();
        assert!(sum <= 1.0 + 1e-12);
        assert_abs_diff_eq!(sum, 5.0 / 6.0, epsilon = 1e-12);
        // Stronger adsorber (K=3) takes more coverage.
        assert!(theta[1] > theta[0]);
    }

    #[test]
    fn langmuir_hinshelwood_empty_sites() {
        // Zero pressure => zero coverage, bare site = 1.
        let theta = langmuir_hinshelwood_coverages(&[5.0], &[0.0]).unwrap();
        assert_abs_diff_eq!(theta[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn kinetics_problem_resolves_constants() {
        let prob = KineticsProblem::new(vec![
            ArrheniusRate::new(1.0e13, 80_000.0).unwrap(),
            ArrheniusRate::new(1.0e10, 40_000.0).unwrap(),
        ])
        .unwrap();
        let ks = prob.rate_constants(800.0);
        assert_eq!(ks.len(), 2);
        assert!(ks[0] > 0.0 && ks[1] > 0.0);
        assert!(KineticsProblem::new(vec![]).is_err());
    }

    #[test]
    fn multi_site_reduces_to_single_site() {
        // All species on the same site type => identical to the single-site fn.
        let ks = [2.0, 3.0];
        let ps = [1.0, 1.0];
        let single = langmuir_hinshelwood_coverages(&ks, &ps).unwrap();
        let multi = multi_site_langmuir_hinshelwood_coverages(&ks, &ps, &[0, 0]).unwrap();
        for (s, m) in single.iter().zip(multi.iter()) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-12);
        }
    }

    #[test]
    fn multi_site_normalizes_independently_per_site_type() {
        // Species 0,1 on site type 0; species 2 alone on site type 1.
        let theta = multi_site_langmuir_hinshelwood_coverages(
            &[2.0, 3.0, 5.0],
            &[1.0, 1.0, 1.0],
            &[0, 0, 1],
        )
        .unwrap();
        // Site type 0: theta_i = K_i / (1 + 2 + 3) = K_i / 6.
        assert_abs_diff_eq!(theta[0], 2.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(theta[1], 3.0 / 6.0, epsilon = 1e-12);
        // Site type 1: single adsorbate, theta_2 = 5 / (1 + 5) = 5/6, unaffected
        // by site-type-0 occupancy.
        assert_abs_diff_eq!(theta[2], 5.0 / 6.0, epsilon = 1e-12);
        let site0_sum = theta[0] + theta[1];
        assert!(site0_sum <= 1.0 + 1e-12);
        assert!(theta[2] <= 1.0 + 1e-12);
    }

    #[test]
    fn multi_site_rejects_mismatched_lengths_and_bad_values() {
        assert!(multi_site_langmuir_hinshelwood_coverages(&[1.0], &[1.0, 2.0], &[0, 0]).is_err());
        assert!(multi_site_langmuir_hinshelwood_coverages(&[-1.0], &[1.0], &[0]).is_err());
        assert!(multi_site_langmuir_hinshelwood_coverages(&[1.0], &[-1.0], &[0]).is_err());
    }

    #[test]
    fn eley_rideal_zero_when_either_factor_zero() {
        let er = EleyRideal::new(3.0).unwrap();
        assert_eq!(er.rate(0.0, 5.0), 0.0);
        assert_eq!(er.rate(0.5, 0.0), 0.0);
        assert_eq!(er.rate(0.0, 0.0), 0.0);
    }

    #[test]
    fn eley_rideal_linear_in_gas_phase_concentration() {
        let er = EleyRideal::new(2.0).unwrap();
        let r1 = er.rate(0.4, 1.0);
        let r2 = er.rate(0.4, 3.0);
        // Linear in gas-phase concentration for fixed coverage.
        assert_abs_diff_eq!(r2, 3.0 * r1, epsilon = 1e-12);
        assert_abs_diff_eq!(r1, 2.0 * 0.4 * 1.0, epsilon = 1e-12);
    }

    #[test]
    fn eley_rideal_rejects_invalid_rate() {
        assert!(EleyRideal::new(-1.0).is_err());
        assert!(EleyRideal::new(f64::NAN).is_err());
    }

    #[test]
    fn eley_rideal_into_rate_law_matches_direct_rate() {
        use tpt_sci_reaction_network::ReactionNetwork;

        let er = EleyRideal::new(1.5).unwrap();
        let mut net = ReactionNetwork::new();
        let theta_a = net.species("theta_A");
        let gas_b = net.species("B_gas");
        let c = net.species("C");
        net.reaction(
            &[(theta_a, 1.0), (gas_b, 1.0)],
            &[(c, 1.0)],
            er.into_rate_law(0, 1),
        );
        let sys = net.build().unwrap();
        let y = [0.3, 2.0, 0.0];
        let rates = sys.reaction_rates(&y);
        assert_abs_diff_eq!(rates[0], er.rate(0.3, 2.0), epsilon = 1e-12);
    }

    #[test]
    fn coverage_dependent_matches_plain_arrhenius_when_alpha_zero() {
        let plain = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
        let cov = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, 0.0).unwrap();
        for theta in [0.0, 0.3, 1.0] {
            assert_abs_diff_eq!(
                cov.rate_constant(800.0, theta),
                plain.rate_constant(800.0),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn coverage_dependent_rate_constant_monotonic_in_theta() {
        // alpha > 0: Ea increases with theta => k decreases with theta.
        let repulsive = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, 20_000.0).unwrap();
        let k_low = repulsive.rate_constant(800.0, 0.0);
        let k_high = repulsive.rate_constant(800.0, 1.0);
        assert!(k_high < k_low);

        // alpha < 0: Ea decreases with theta => k increases with theta.
        let promoting = CoverageDependentArrheniusRate::new(1.0e13, 80_000.0, -20_000.0).unwrap();
        let k_low = promoting.rate_constant(800.0, 0.0);
        let k_high = promoting.rate_constant(800.0, 1.0);
        assert!(k_high > k_low);
    }

    #[test]
    fn coverage_dependent_rejects_invalid() {
        assert!(CoverageDependentArrheniusRate::new(0.0, 1.0, 0.0).is_err());
        assert!(CoverageDependentArrheniusRate::new(1.0, -1.0, 0.0).is_err());
        assert!(CoverageDependentArrheniusRate::new(1.0, 1.0, f64::NAN).is_err());
    }
}
