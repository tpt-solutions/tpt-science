//! Public-API invariants for the Hodgkin–Huxley cell.

use tpt_sci_electrophys::HodgkinHuxley;

#[test]
fn resting_cell_is_stable_without_stimulus() {
    let mut hh = HodgkinHuxley::resting();
    let v_rest = hh.voltage();
    for _ in 0..5000 {
        hh.step(0.01);
    }
    assert!(hh.voltage().is_finite());
    assert!(
        (hh.voltage() - v_rest).abs() < 0.5,
        "drifted to {} mV",
        hh.voltage()
    );
}
