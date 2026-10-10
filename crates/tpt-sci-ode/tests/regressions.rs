//! Regression tests for solver robustness: RHS error propagation and dense
//! output landing exactly on requested times.

use std::cell::Cell;

use tpt_sci_ode::{Method, OdeError, OdeProblem, RhsCallable};

/// RHS for `dy/dt = -y` that succeeds for the first `ok_calls` evaluations and
/// then reports an error.
struct FailAfter {
    n: usize,
    ok_calls: usize,
    calls: Cell<usize>,
}

impl RhsCallable for FailAfter {
    fn nstates(&self) -> usize {
        self.n
    }

    fn call(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), OdeError> {
        let c = self.calls.get();
        self.calls.set(c + 1);
        if c >= self.ok_calls {
            return Err(OdeError::Invalid("rhs failed".to_string()));
        }
        for (d, yi) in dydt.iter_mut().zip(y) {
            *d = -yi;
        }
        Ok(())
    }
}

fn failing(n: usize, ok_calls: usize) -> OdeProblem {
    OdeProblem::from_rhs(
        FailAfter {
            n,
            ok_calls,
            calls: Cell::new(0),
        },
        vec![1.0; n],
        0.0,
    )
    .unwrap()
}

#[test]
fn rhs_error_propagates_from_explicit_method() {
    let prob = failing(1, 0);
    assert!(matches!(
        prob.solve(Method::Tsit45, 1.0),
        Err(OdeError::Invalid(_))
    ));
}

#[test]
fn rhs_error_propagates_from_implicit_methods_dense_path() {
    for m in [Method::Bdf, Method::TrBdf2, Method::Esdirk34] {
        let prob = failing(1, 5);
        assert!(
            matches!(prob.solve(m, 1.0), Err(OdeError::Invalid(_))),
            "{m:?} did not propagate the RHS error"
        );
    }
}

#[test]
fn rhs_error_propagates_from_implicit_methods_sparse_path() {
    // n = 80 routes the Newton solves through the sparse path.
    for m in [Method::Bdf, Method::TrBdf2, Method::Esdirk34] {
        let prob = failing(80, 20);
        assert!(
            matches!(prob.solve(m, 1.0), Err(OdeError::Invalid(_))),
            "{m:?} did not propagate the RHS error on the sparse path"
        );
    }
}

#[test]
fn dense_output_lands_exactly_on_uniform_grids() {
    // Grids whose spacing is not exactly representable make `t + (te - t)`
    // land an ulp short of `te`. The driver must still hit every point.
    let decay = |_t: f64, y: &[f64], dydt: &mut [f64]| dydt[0] = -y[0];
    for m in [
        Method::Tsit45,
        Method::Bdf,
        Method::TrBdf2,
        Method::Esdirk34,
    ] {
        for i in 1..=60 {
            let dt = i as f64 / 37.0;
            let grid: Vec<f64> = (1..=30).map(|k| k as f64 * dt).collect();
            let prob = OdeProblem::new(decay, vec![1.0], 0.0).unwrap();
            let out = prob
                .solve_dense(m, &grid)
                .unwrap_or_else(|e| panic!("{m:?} dt={dt}: {e}"));
            assert_eq!(out.len(), grid.len(), "{m:?} dt={dt}");
            for (te, y) in grid.iter().zip(&out) {
                let exact = (-te).exp();
                assert!(
                    (y[0] - exact).abs() < 1e-3,
                    "{m:?} dt={dt} t={te}: {} vs {exact}",
                    y[0]
                );
            }
        }
    }
}

#[test]
fn solve_reaches_uniform_final_times() {
    let decay = |_t: f64, y: &[f64], dydt: &mut [f64]| dydt[0] = -y[0];
    for m in [Method::Tsit45, Method::Bdf] {
        for k in 1..=40 {
            let t_final = k as f64 * 0.1;
            let prob = OdeProblem::new(decay, vec![1.0], 0.0).unwrap();
            let y = prob
                .solve(m, t_final)
                .unwrap_or_else(|e| panic!("{m:?} t_final={t_final}: {e}"));
            assert!((y[0] - (-t_final).exp()).abs() < 1e-3);
        }
    }
}

#[test]
fn bdf_matches_closed_forms() {
    // Exponential decay: y(1) = exp(-2).
    let k = 2.0;
    let prob = OdeProblem::new(
        move |_t: f64, y: &[f64], d: &mut [f64]| d[0] = -k * y[0],
        vec![1.0],
        0.0,
    )
    .unwrap();
    let y = prob.solve(Method::Bdf, 1.0).unwrap();
    assert!((y[0] - (-k).exp()).abs() < 1e-5, "decay: {}", y[0]);

    // Harmonic oscillator: after a quarter period, (x, v) = (1, 0) -> (0, -1)
    // from x(0)=0, v(0)=1 gives x = sin t, v = cos t, so at t = π/2: (1, 0).
    let prob = OdeProblem::new(
        |_t: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[1];
            d[1] = -y[0];
        },
        vec![0.0, 1.0],
        0.0,
    )
    .unwrap();
    let y = prob
        .solve(Method::Bdf, std::f64::consts::FRAC_PI_2)
        .unwrap();
    assert!(
        (y[0] - 1.0).abs() < 1e-4 && y[1].abs() < 1e-4,
        "oscillator: {y:?}"
    );
}

#[test]
fn sir_model_conserves_population_and_methods_agree() {
    // Nonlinear, non-stiff model; every method should conserve S + I + R and
    // agree with the others to well within the tolerance.
    let rhs = |_t: f64, y: &[f64], d: &mut [f64]| {
        let inf = 0.6 * y[0] * y[1];
        d[0] = -inf;
        d[1] = inf - 0.2 * y[1];
        d[2] = 0.2 * y[1];
    };
    let mut results = Vec::new();
    for m in [
        Method::Tsit45,
        Method::Bdf,
        Method::TrBdf2,
        Method::Esdirk34,
    ] {
        let prob = tpt_sci_ode::OdeProblemBuilder::new(rhs, vec![0.99, 0.01, 0.0], 0.0)
            .rtol(1e-8)
            .atol(1e-8)
            .build()
            .unwrap();
        let y = prob.solve(m, 20.0).unwrap();
        let total: f64 = y.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{m:?} population {total}");
        results.push((m, y));
    }
    let reference = &results[0].1;
    for (m, y) in &results[1..] {
        for (a, b) in y.iter().zip(reference) {
            assert!(
                (a - b).abs() < 1e-4,
                "{m:?} disagrees with Tsit45: {a} vs {b}"
            );
        }
    }
}

#[test]
fn implicit_methods_converge_at_large_magnitudes() {
    // y' = -1000 (y - 1e6), y(0) = 0. Exact: y(t) = 1e6 (1 - exp(-1000 t)).
    // The Newton test must scale with the magnitude of y, not use an absolute
    // residual threshold.
    let t = 0.01;
    let exact = 1e6 * (1.0 - (-1000.0_f64 * t).exp());
    for m in [Method::Bdf, Method::TrBdf2, Method::Esdirk34] {
        let prob = OdeProblem::new(
            |_t: f64, y: &[f64], d: &mut [f64]| d[0] = -1000.0 * (y[0] - 1e6),
            vec![0.0],
            0.0,
        )
        .unwrap();
        let y = prob
            .solve(m, t)
            .unwrap_or_else(|e| panic!("{m:?} failed at large magnitude: {e}"));
        assert!(
            (y[0] - exact).abs() / 1e6 < 1e-5,
            "{m:?}: {} vs {exact}",
            y[0]
        );
    }
}

#[test]
fn t_eval_must_be_strictly_increasing_and_finite() {
    let decay = |_t: f64, y: &[f64], d: &mut [f64]| d[0] = -y[0];
    for bad in [
        vec![0.5, 0.4],
        vec![0.5, 0.5],
        vec![0.2, f64::NAN],
        vec![-0.1, 0.5],
    ] {
        let prob = OdeProblem::new(decay, vec![1.0], 0.0).unwrap();
        assert!(
            matches!(
                prob.solve_dense(Method::Tsit45, &bad),
                Err(OdeError::Invalid(_))
            ),
            "t_eval {bad:?} should be rejected"
        );
    }
}
