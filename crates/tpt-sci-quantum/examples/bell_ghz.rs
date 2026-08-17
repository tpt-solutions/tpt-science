//! Measurement statistics for Bell and GHZ states, including a collapse-aware
//! multi-shot GHZ experiment that uses `measure_collapsing`.
use tpt_math_prob_core::SplitMix64;
use tpt_sci_quantum::State;

fn main() {
    // Bell state |00> + |11>.
    let mut bell = State::new(2).unwrap();
    bell.h(0).unwrap();
    bell.cnot(0, 1).unwrap();
    let mut rng = SplitMix64::seed_from_u64(2024);
    let trials = 5000;
    let mut correlated = 0usize;
    for _ in 0..trials {
        let s = bell.measure(&mut rng);
        if s == 0 || s == 3 {
            correlated += 1;
        }
    }
    println!(
        "Bell: {:.1}% of outcomes are |00> or |11>",
        100.0 * correlated as f64 / trials as f64
    );

    // GHZ state over 3 qubits; collapse-aware multi-shot.
    let mut ghz = State::new(3).unwrap();
    ghz.h(0).unwrap();
    ghz.cnot(0, 1).unwrap();
    ghz.cnot(1, 2).unwrap();
    let mut rng2 = SplitMix64::seed_from_u64(7);
    let mut all_same = 0usize;
    for _ in 0..trials {
        let mut g = ghz.clone();
        let a = g.measure_collapsing(&mut rng2);
        let b = g.measure_collapsing(&mut rng2);
        let c = g.measure_collapsing(&mut rng2);
        if a == b && b == c {
            all_same += 1;
        }
    }
    println!(
        "GHZ: {:.1}% of multi-shots have all-equal bits",
        100.0 * all_same as f64 / trials as f64
    );
}
