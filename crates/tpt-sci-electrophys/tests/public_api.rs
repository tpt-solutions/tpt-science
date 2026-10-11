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

#[test]
fn one_row_bidomain_cable_solves_without_panicking() {
    use tpt_sci_electrophys::{Diffusivity, Tissue};
    let mut cable =
        Tissue::with_model(30, 1, Diffusivity::Scalar(1.0), HodgkinHuxley::resting()).unwrap();
    cable.enable_bidomain(1.0);
    for i in 0..4 {
        cable.stimulate(i, 0, 40.0);
    }
    for _ in 0..200 {
        cable.bidomain_step(0.01).unwrap();
    }
    let phi = cable.extracellular_potential();
    assert_eq!(phi.len(), 30);
    assert!(phi.iter().all(|v| v.is_finite()));
    // The Neumann-problem solution is zero-mean.
    let mean = phi.iter().sum::<f64>() / phi.len() as f64;
    assert!(mean.abs() < 1e-6, "mean {mean}");
}
