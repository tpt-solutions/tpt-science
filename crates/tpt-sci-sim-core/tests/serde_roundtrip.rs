//! Checkpoint serialization round-trip (requires the `serde` feature).

use tpt_sci_sim_core::Checkpoint;

#[test]
fn checkpoint_round_trips_through_json() {
    let cp = Checkpoint {
        global_time: 1.5,
        models: vec![("decay".into(), vec![0.25, 0.5], 1.5)],
    };
    let json = serde_json::to_string(&cp).unwrap();
    let back: Checkpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(back.global_time, cp.global_time);
    assert_eq!(back.models, cp.models);
}
