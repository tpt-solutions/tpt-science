//! Explicit tau-leaping and the Chemical Langevin Equation (CLE): two
//! approximate stochastic integrators that complement the exact SSA in
//! [`crate::ssa`].
//!
//! Both methods approximate the same continuous-time Markov jump process
//! that [`crate::ReactionSystem::simulate_ssa`] samples exactly, but advance
//! the state in coarser steps for speed:
//!
//! * **Tau-leaping** ([`crate::ReactionSystem::simulate_tau_leaping`]) fires
//!   each reaction `k_j ~ Poisson(a_j(y)·τ)` times over a step of length
//!   `τ`, independently per reaction (Gillespie, 2001, "Approximate
//!   accelerated stochastic simulation of chemically reacting systems").
//!   This is the primary, well-defined backend requested for this crate: it
//!   stays exact in the `τ → 0` limit and is valid for any (including very
//!   small) population.
//! * **CLE** ([`crate::ReactionSystem::simulate_cle`]) replaces the Poisson
//!   jumps with their diffusion (large-population) approximation — Gaussian
//!   increments integrated by Euler–Maruyama — which is faster still but
//!   only sensible once propensities are large enough that populations can
//!   be treated as continuous.
//!
//! Both return a [`crate::SsaTrajectory`], the same trajectory type produced
//! by [`crate::ReactionSystem::simulate_ssa`], so downstream code (plotting,
//! summary statistics, …) does not need to distinguish which backend
//! produced a given run.

/// Configuration for [`crate::ReactionSystem::simulate_tau_leaping`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauLeapConfig {
    /// If `Some`, use this fixed step size `τ` for every leap. If `None`
    /// (the default), select `τ` adaptively at each step from `epsilon`.
    pub fixed_tau: Option<f64>,
    /// Target bound on the relative change of each species' propensity-driven
    /// drift/diffusion per step (ignored when `fixed_tau` is set); a
    /// simplified version of the Cao–Gillespie–Petzold tau-selection rule
    /// (it omits the "highest order reaction" refinement `g_i`, using
    /// `g_i = 1` for every species, which is a conservative — i.e. safe but
    /// sometimes smaller-than-necessary — choice of `τ`). Smaller values are
    /// more accurate but slower; `0.03` is a typical default.
    pub epsilon: f64,
    /// If a leap would drive any species population negative — an artifact
    /// of the Poisson approximation, impossible in the exact SSA — the step
    /// is rejected and retried with `τ` halved, up to this many times,
    /// before falling back to clamping the offending species at zero.
    pub max_step_halvings: u32,
}

impl Default for TauLeapConfig {
    fn default() -> Self {
        Self {
            fixed_tau: None,
            epsilon: 0.03,
            max_step_halvings: 16,
        }
    }
}

/// Draw a `Poisson(lambda)`-distributed count from uniform `[0, 1)` variates
/// supplied by `rng`.
///
/// Uses Knuth's multiplication algorithm for moderate `lambda` (`<= 30`),
/// which is exact but `O(lambda)` in expectation; for larger `lambda`
/// (routine once populations are large, where tau-leaping is most useful)
/// falls back to rounding a `Normal(lambda, lambda)` approximation to the
/// nearest non-negative integer, which is standard practice in tau-leaping
/// implementations (the Poisson distribution is asymptotically normal, and
/// exact sampling would cost `O(lambda)` time per draw).
pub(crate) fn sample_poisson<F: FnMut() -> f64>(lambda: f64, rng: &mut F) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda <= 30.0 {
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0_f64;
        loop {
            k += 1;
            p *= (rng)().max(f64::MIN_POSITIVE);
            if p <= l {
                break;
            }
        }
        (k - 1) as f64
    } else {
        let z = sample_standard_normal(rng);
        (lambda + lambda.sqrt() * z).round().max(0.0)
    }
}

/// Draw a standard normal variate via the Box–Muller transform.
pub(crate) fn sample_standard_normal<F: FnMut() -> f64>(rng: &mut F) -> f64 {
    let u1 = (rng)().max(f64::MIN_POSITIVE);
    let u2 = (rng)();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use crate::{RateLaw, ReactionNetwork};

    /// A small deterministic uniform RNG (`SplitMix64`-style), matching the
    /// one used in `ssa`'s tests, so these tests do not depend on an
    /// external RNG crate.
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            (bits as f64) / (1u64 << 53) as f64
        }
    }

    fn birth_death() -> ReactionNetwork {
        let mut net = ReactionNetwork::new();
        let a = net.species("A");
        net.parameter("k1", 0.0);
        net.parameter("k2", 0.0);
        net.reaction(&[], &[(a, 1.0)], RateLaw::mass_action("k1"));
        net.reaction(&[(a, 1.0)], &[], RateLaw::mass_action("k2"));
        net
    }

    #[test]
    fn poisson_mean_matches_lambda() {
        let mut rng = TestRng::new(11);
        for &lambda in &[0.5, 5.0, 50.0, 500.0] {
            let n = 20_000;
            let mut sum = 0.0;
            for _ in 0..n {
                sum += super::sample_poisson(lambda, &mut || rng.next_f64());
            }
            let mean = sum / n as f64;
            // Mean should be within a few standard errors of lambda
            // (std error of the mean ~ sqrt(lambda / n)).
            let tol = 6.0 * (lambda / n as f64).sqrt() + 0.05;
            assert!(
                (mean - lambda).abs() < tol,
                "lambda={lambda} mean={mean} tol={tol}"
            );
        }
    }

    #[test]
    fn tau_leaping_matches_deterministic_steady_state() {
        // dA/dt = k1 - k2 A -> steady state A* = k1/k2. Run many replicate
        // tau-leaping trajectories at large populations (where the Poisson
        // approximation is very accurate) and check the ensemble mean at
        // t_max is close to the deterministic steady state.
        let mut sys = birth_death().build().unwrap();
        sys.set_parameter("k1", 100.0).unwrap();
        sys.set_parameter("k2", 1.0).unwrap();
        let y0 = sys.initial_state(&[("A", 100.0)]).unwrap();

        let mut rng = TestRng::new(42);
        let n_reps = 200;
        let mut sum_final = 0.0;
        for _ in 0..n_reps {
            let traj = sys
                .simulate_tau_leaping(
                    &y0,
                    20.0,
                    &crate::tau_leap::TauLeapConfig::default(),
                    &mut || rng.next_f64(),
                )
                .unwrap();
            sum_final += traj.final_state()[0];
        }
        let mean_final = sum_final / n_reps as f64;
        // Deterministic steady state is 100.0; allow generous stochastic
        // tolerance (population ~100 => sd ~10, sd of mean over 200 reps
        // ~0.7, so 5.0 is a very safe bound).
        assert!(
            (mean_final - 100.0).abs() < 8.0,
            "mean final A = {mean_final}, expected close to 100.0"
        );
    }

    #[test]
    fn tau_leaping_never_goes_negative() {
        let mut sys = birth_death().build().unwrap();
        sys.set_parameter("k1", 1.0).unwrap();
        sys.set_parameter("k2", 5.0).unwrap();
        let y0 = sys.initial_state(&[("A", 3.0)]).unwrap();
        let mut rng = TestRng::new(5);
        let config = crate::tau_leap::TauLeapConfig {
            fixed_tau: Some(0.5),
            ..Default::default()
        };
        let traj = sys
            .simulate_tau_leaping(&y0, 10.0, &config, &mut || rng.next_f64())
            .unwrap();
        for st in &traj.states {
            assert!(st[0] >= 0.0, "population went negative: {st:?}");
        }
    }

    #[test]
    fn tau_leaping_rejects_wrong_dimension() {
        let sys = birth_death().build().unwrap();
        let mut rng = TestRng::new(1);
        assert!(
            sys.simulate_tau_leaping(
                &[1.0, 2.0],
                10.0,
                &crate::tau_leap::TauLeapConfig::default(),
                &mut || rng.next_f64()
            )
            .is_err()
        );
    }

    #[test]
    fn cle_matches_deterministic_steady_state() {
        let mut sys = birth_death().build().unwrap();
        sys.set_parameter("k1", 100.0).unwrap();
        sys.set_parameter("k2", 1.0).unwrap();
        let y0 = sys.initial_state(&[("A", 100.0)]).unwrap();

        let mut rng = TestRng::new(99);
        let n_reps = 200;
        let mut sum_final = 0.0;
        for _ in 0..n_reps {
            let traj = sys
                .simulate_cle(&y0, 20.0, 0.01, &mut || rng.next_f64())
                .unwrap();
            sum_final += traj.final_state()[0];
        }
        let mean_final = sum_final / n_reps as f64;
        assert!(
            (mean_final - 100.0).abs() < 8.0,
            "mean final A = {mean_final}, expected close to 100.0"
        );
    }
}
