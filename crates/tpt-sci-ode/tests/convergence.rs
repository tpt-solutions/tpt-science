//! Tolerance-convergence gate for every integrator.
//!
//! An adaptive method's global error must fall as its tolerance tightens. A
//! scheme with inconsistent coefficients (wrong row sums or quadrature weights)
//! converges to the wrong answer and shows a flat error floor instead, so this
//! catches that class of bug that single-tolerance checks miss.

use std::f64::consts::PI;

use tpt_sci_ode::{Method, OdeProblemBuilder};

const METHODS: [Method; 4] = [
    Method::Tsit45,
    Method::TrBdf2,
    Method::Esdirk34,
    Method::Bdf,
];

/// Global error at `t = 4π` for `y'' = -y` (exact `y = cos t`), at relative
/// tolerance `rtol`. `None` if the solver fails.
fn error_at(method: Method, rtol: f64) -> Option<f64> {
    let t_end = 4.0 * PI;
    let prob = OdeProblemBuilder::new(
        |_t, y: &[f64], dy: &mut [f64]| {
            dy[0] = y[1];
            dy[1] = -y[0];
        },
        vec![1.0, 0.0],
        0.0,
    )
    .rtol(rtol)
    .atol(rtol * 1e-2)
    .build()
    .ok()?;
    let y = prob.solve(method, t_end).ok()?;
    Some((y[0] - t_end.cos()).abs())
}

#[test]
fn error_decreases_as_tolerance_tightens() {
    for method in METHODS {
        let errs: Vec<f64> = [1e-4, 1e-6, 1e-8]
            .iter()
            .map(|&rtol| {
                error_at(method, rtol)
                    .unwrap_or_else(|| panic!("{method:?} failed at rtol = {rtol:e}"))
            })
            .collect();
        assert!(
            errs[0] > errs[1] && errs[1] > errs[2],
            "{method:?}: error not monotone in tolerance: {errs:?}"
        );
        // Two decades of tolerance should buy at least one decade of accuracy.
        assert!(
            errs[2] < errs[0] / 10.0,
            "{method:?}: insufficient error reduction: {errs:?}"
        );
    }
}

#[test]
fn second_order_tr_bdf2_reaches_tight_accuracy() {
    // Regression for an inconsistent TR-BDF2 tableau that plateaued near 4e-4
    // regardless of tolerance.
    let err = error_at(Method::TrBdf2, 1e-8).expect("TR-BDF2 must converge");
    assert!(err < 1e-7, "TR-BDF2 error {err:e} at rtol 1e-8");
}

#[test]
fn esdirk34_reaches_tight_accuracy() {
    // Regression for ESDIRK34 weights that violated the third-order conditions.
    let err = error_at(Method::Esdirk34, 1e-8).expect("ESDIRK34 must converge");
    assert!(err < 1e-5, "ESDIRK34 error {err:e} at rtol 1e-8");
}
