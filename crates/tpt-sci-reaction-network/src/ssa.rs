//! Stochastic simulation (Gillespie's direct method) for reaction networks.
//!
//! This fills the v1 gap noted in the crate overview: the deterministic
//! mass-action ODE (`dy/dt = S·r(y)`) is now complemented by a stochastic
//! trajectory sampler. Given integer-ish species counts, [`crate::ReactionSystem::simulate_ssa`]
//! draws an exact trajectory of the continuous-time Markov chain by firing
//! reactions at their (combinatorial) propensities. This is the in-house
//! equivalent of a Gillespie / SSA backend (the `rebop` wrap target remains the
//! recommended option for very large networks).

/// A stochastic trajectory produced by [`crate::ReactionSystem::simulate_ssa`].
///
/// `times[i]` is the time at which the state `states[i]` holds. `times[0] == 0`
/// with the initial state, every subsequent entry is the state immediately
/// after a reaction firing, and the final entry is the state clamped at
/// `t_max` (so `times.last() == t_max`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SsaTrajectory {
    /// Event times, starting at `0.0` and ending at `t_max`.
    pub times: Vec<f64>,
    /// Species state vector at each event time, parallel to [`SsaTrajectory::times`].
    pub states: Vec<Vec<f64>>,
}

impl SsaTrajectory {
    /// Number of recorded states (one per entry in [`SsaTrajectory::times`]).
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the trajectory is empty (never true for a successful simulation,
    /// which always records at least the initial and final states).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// The initial state (`states[0]`).
    #[must_use]
    pub fn initial(&self) -> &[f64] {
        &self.states[0]
    }

    /// The final state (`states.last()`).
    ///
    /// # Panics
    ///
    /// Panics if the trajectory is empty; a successful simulation always records
    /// at least the initial and final states, so this is unreachable in practice.
    #[must_use]
    pub fn final_state(&self) -> &[f64] {
        self.states
            .last()
            .expect("trajectory always has a final state")
    }
}

#[cfg(test)]
mod tests {
    use crate::ReactionNetwork;

    /// A small deterministic uniform RNG (`SplitMix64`-style) so the SSA tests do
    /// not depend on an external RNG crate.
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
            // 53 bits of mantissa, mapped to [0, 1).
            let bits = self.next_u64() >> 11;
            (bits as f64) / (1u64 << 53) as f64
        }
    }

    fn sir() -> ReactionNetwork {
        let mut net = ReactionNetwork::new();
        let s = net.species("S");
        let i = net.species("I");
        let r = net.species("R");
        net.parameter("beta", 0.0);
        net.parameter("gamma", 0.0);
        net.reaction(
            &[(s, 1.0), (i, 1.0)],
            &[(i, 2.0)],
            crate::RateLaw::mass_action("beta"),
        );
        net.reaction(
            &[(i, 1.0)],
            &[(r, 1.0)],
            crate::RateLaw::mass_action("gamma"),
        );
        net
    }

    #[test]
    fn ssa_conserves_sir_population() {
        let mut sys = sir().build().unwrap();
        sys.set_parameter("beta", 1e-3).unwrap();
        sys.set_parameter("gamma", 0.05).unwrap();
        let y0 = sys
            .initial_state(&[("S", 999.0), ("I", 1.0), ("R", 0.0)])
            .unwrap();
        let mut rng = TestRng::new(7);
        let traj = sys
            .simulate_ssa(&y0, 100.0, &mut || rng.next_f64())
            .unwrap();
        // S + I + R is conserved at every event.
        for st in &traj.states {
            let total = st[0] + st[1] + st[2];
            assert!(
                (total - 1000.0).abs() < 1e-9,
                "population {} not conserved",
                total
            );
        }
        // The epidemic must have run (some recoveries happened).
        let final_r = traj.final_state()[2];
        assert!(final_r > 0.0, "expected some recoveries");
        assert!(traj.times.last() == Some(&100.0));
    }

    #[test]
    fn ssa_rejects_wrong_dimension() {
        let sys = sir().build().unwrap();
        let mut rng = TestRng::new(1);
        assert!(
            sys.simulate_ssa(&[1.0], 10.0, &mut || rng.next_f64())
                .is_err()
        );
    }

    #[test]
    fn ssa_rejects_negative_counts() {
        let sys = sir().build().unwrap();
        let mut rng = TestRng::new(1);
        let y0 = sys
            .initial_state(&[("S", -1.0), ("I", 1.0), ("R", 0.0)])
            .unwrap();
        assert!(matches!(
            sys.simulate_ssa(&y0, 10.0, &mut || rng.next_f64()),
            Err(crate::ReactionNetworkError::InvalidState(_))
        ));
    }

    #[test]
    fn ssa_no_reactions_is_constant() {
        // A network with a parameter but no reactions has zero propensity and
        // should return the initial and final states unchanged.
        let mut net = ReactionNetwork::new();
        net.species("A");
        let sys = net.build().unwrap();
        let y0 = vec![5.0];
        let mut rng = TestRng::new(3);
        let traj = sys.simulate_ssa(&y0, 10.0, &mut || rng.next_f64()).unwrap();
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.final_state()[0], 5.0);
    }
}

use crate::{ReactionNetworkError, ReactionSystem};
use tpt_math_prob_core::{Rng, SplitMix64};

impl ReactionSystem {
    /// Run `n_runs` independent SSA trajectories from `y0` to `t_max`.
    ///
    /// Each run has its own generator, seeded from a stream derived from
    /// `base_seed`, so the ensemble is reproducible and does not depend on how
    /// the runs are scheduled. With the `rayon` feature the runs execute in
    /// parallel; the output is the same either way.
    ///
    /// # Errors
    ///
    /// Returns the first error from [`ReactionSystem::simulate_ssa`] (an invalid
    /// initial state, for example).
    pub fn simulate_ssa_ensemble(
        &self,
        y0: &[f64],
        t_max: f64,
        n_runs: usize,
        base_seed: u64,
    ) -> Result<Vec<SsaTrajectory>, ReactionNetworkError> {
        let mut seeder = SplitMix64::seed_from_u64(base_seed);
        let seeds: Vec<u64> = (0..n_runs).map(|_| seeder.next_u64()).collect();
        let run = |seed: &u64| {
            let mut rng = SplitMix64::seed_from_u64(*seed);
            self.simulate_ssa(y0, t_max, &mut || rng.next_f64())
        };
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            seeds.par_iter().map(run).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            seeds.iter().map(run).collect()
        }
    }
}

#[cfg(test)]
mod ensemble_tests {
    use crate::{RateLaw, ReactionNetwork};

    #[test]
    fn ensemble_is_reproducible_and_seed_dependent() {
        // ∅ → X at rate k = 1: a pure production process.
        let mut net = ReactionNetwork::new();
        let x = net.species("X");
        let k = net.parameter("k", 1.0);
        let _ = k;
        net.reaction(&[], &[(x, 1.0)], RateLaw::MassAction { k: "k".into() });
        let sys = net.build().unwrap();
        let a = sys.simulate_ssa_ensemble(&[0.0], 5.0, 8, 42).unwrap();
        let b = sys.simulate_ssa_ensemble(&[0.0], 5.0, 8, 42).unwrap();
        let c = sys.simulate_ssa_ensemble(&[0.0], 5.0, 8, 43).unwrap();
        assert_eq!(a.len(), 8);
        assert!(
            a.iter()
                .zip(&b)
                .all(|(x, y)| x.states == y.states && x.times == y.times)
        );
        assert!(a.iter().zip(&c).any(|(x, y)| x.states != y.states));
    }
}
