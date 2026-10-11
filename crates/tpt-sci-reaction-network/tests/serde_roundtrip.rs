//! `SsaTrajectory` serialization round-trip (requires the `serde` feature).

use tpt_sci_reaction_network::SsaTrajectory;

#[test]
fn ssa_trajectory_round_trips_through_json() {
    let tr = SsaTrajectory {
        times: vec![0.0, 0.7, 1.0],
        states: vec![vec![10.0, 0.0], vec![9.0, 1.0], vec![9.0, 1.0]],
    };
    let json = serde_json::to_string(&tr).unwrap();
    let back: SsaTrajectory = serde_json::from_str(&json).unwrap();
    assert_eq!(back.times, tr.times);
    assert_eq!(back.states, tr.states);
}
