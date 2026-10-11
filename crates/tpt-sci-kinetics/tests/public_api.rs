//! Public-API invariants for surface kinetics.

use tpt_sci_kinetics::{ArrheniusRate, langmuir_hinshelwood_coverages};

#[test]
fn arrhenius_rate_matches_closed_form_and_increases_with_temperature() {
    let r = ArrheniusRate::new(1.0e13, 80_000.0).unwrap();
    let r_gas = 8.314_462_618;
    let expected = 1.0e13 * (-80_000.0_f64 / (r_gas * 800.0)).exp();
    let k = r.rate_constant(800.0);
    assert!((k - expected).abs() < 1e-9 * expected);
    assert!(r.rate_constant(900.0) > k);
}

#[test]
fn coverages_are_bounded_and_grow_with_pressure() {
    let low = langmuir_hinshelwood_coverages(&[1.0, 2.0], &[0.1, 0.1]).unwrap();
    let high = langmuir_hinshelwood_coverages(&[1.0, 2.0], &[1.0, 1.0]).unwrap();
    for th in low.iter().chain(&high) {
        assert!((0.0..=1.0).contains(th));
    }
    assert!(high[0] > low[0]);
    assert!(high.iter().sum::<f64>() <= 1.0 + 1e-9);
}
