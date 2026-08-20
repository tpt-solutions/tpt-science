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
