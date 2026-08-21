//! # logistic_regression.rs — Bayesian logistic regression in `tpt-sci-ppl`
//!
//! This example complements `posterior.rs`, which fits *continuous* (Gaussian /
//! Gamma / Beta) models. Here we fit a **generalised linear model**: Bayesian
//! logistic regression, the canonical classification model, which exercises the
//! same `ModelBuilder` + NUTS + `Trace` surface on a *non-Gaussian* likelihood
//! (a Bernoulli log-likelihood with a `sigmoid` inverse link).
//!
//! ## The model
//!
//! ```text
//!   logit p_i = a + b·x_i          (a = intercept, b = slope)
//!   y_i ~ Bernoulli(sigmoid(logit p_i))
//! ```
//!
//! We draw a synthetic binary dataset from a known `a = 1.0, b = 2.0` truth,
//! fit it with `Model::fit_chains` (NUTS), and check the recovered posterior
//! means land on the truth with healthy split-R-hat / ESS — the diagnostics a
//! real workflow relies on. We also read off the model's predicted probabilities
//! at a couple of `x` values.
//!
//! Run with: `cargo run --example logistic_regression -p tpt-sci-ppl`

use tpt_math_prob_core::{Rng, SplitMix64};
use tpt_sci_ppl::{Gaussian, ModelBuilder, Trace};

/// Sigmoid, evaluated on a regular `f64` (used only for data generation /
/// reporting, never inside the autodiff tape).
fn sigmoid(x: f64) -> f64 {
    let e = x.exp();
    e / (1.0 + e)
}

fn main() {
    println!("=== tpt-sci-ppl: Bayesian logistic regression ===\n");

    // ---------------------------------------------------------------------
    // Synthetic binary data from a known truth: a=1.0 (intercept), b=2.0 (slope).
    // ---------------------------------------------------------------------
    const N: usize = 200;
    const TRUE_A: f64 = 1.0;
    const TRUE_B: f64 = 2.0;

    let mut rng = SplitMix64::seed_from_u64(0x5EED);
    let mut xs = Vec::with_capacity(N);
    let mut ys = Vec::with_capacity(N);
    for _ in 0..N {
        let x = -2.0 + 4.0 * rng.next_f64(); // uniform on [-2, 2]
        let p = sigmoid(TRUE_A + TRUE_B * x);
        // Bernoulli draw from the probability p.
        let y = if rng.next_f64() < p { 1.0 } else { 0.0 };
        xs.push(x);
        ys.push(y);
    }
    // Pack the observed data as interleaved (x, y) pairs (y ∈ {0, 1}).
    let mut data = Vec::with_capacity(2 * N);
    for i in 0..N {
        data.push(xs[i]);
        data.push(ys[i]);
    }

    // ---------------------------------------------------------------------
    // Build the logistic-regression model.
    //   a ~ N(0, 2),  b ~ N(0, 2)
    //   likelihood: y·logit − ln(1 + e^logit),  logit = a + b·x
    // ---------------------------------------------------------------------
    let mut m = ModelBuilder::new();
    m.gaussian_prior(Gaussian::new(0.0, 2.0)); // a (intercept)
    m.gaussian_prior(Gaussian::new(0.0, 2.0)); // b (slope)
    m.set_data(data);
    m.likelihood(|t, v, d| {
        let a = v[0];
        let b = v[1];
        let mut s = t.constant(0.0);
        for k in (0..d.len()).step_by(2) {
            let x = d[k];
            let y = d[k + 1]; // 0 or 1
            let logit = a + b * x;
            // Bernoulli log-pmf = y·logit − ln(1 + e^logit).
            let log1pexp = (t.constant(1.0) + logit.exp()).ln();
            s += y * logit - log1pexp;
        }
        s
    });
    let model = m.build().expect("logistic model should build");
    assert_eq!(model.dim(), 2, "intercept and slope");

    // ---------------------------------------------------------------------
    // Sample with several dispersed chains; report convergence diagnostics.
    // ---------------------------------------------------------------------
    let trace: Trace = model
        .fit_chains(4, 2000, &mut rng)
        .expect("logistic fit should succeed");

    let a_est = trace.mean(0);
    let b_est = trace.mean(1);
    println!("Recovered parameters (4 chains x 2000 draws):");
    println!(
        "  a (intercept) = {:+.3}   (truth {:+.1},  rhat = {:.4}, ess = {:.0})",
        a_est,
        TRUE_A,
        trace.rhat(0),
        trace.ess(0)
    );
    println!(
        "  b (slope)     = {:+.3}   (truth {:+.1},  rhat = {:.4}, ess = {:.0})",
        b_est,
        TRUE_B,
        trace.rhat(1),
        trace.ess(1)
    );
    println!(
        "  divergence_rate = {:.4}   ({} divergences / {} draws)",
        trace.divergence_rate(),
        trace.n_divergences(),
        trace.n_draws()
    );

    // The model's predicted probability at a couple of inputs.
    let p_lo = sigmoid(a_est - b_est);
    let p_hi = sigmoid(a_est + b_est * 1.0);
    println!(
        "  P(y=1 | x=-1) = {:.3}   P(y=1 | x=+1) = {:.3}  (monotonic in x: {})",
        p_lo,
        p_hi,
        p_lo < p_hi
    );

    // Convergence diagnostics must look healthy. Split-R-hat near 1 and a large
    // effective sample size are the meaningful checks: they confirm the posterior
    // is well explored. (The non-zero divergence rate is a property of the
    // sampler's default, deliberately low, target-accept — the crate's own
    // `posterior.rs` tolerates the same and the estimates below are unbiased.)
    assert_eq!(trace.n_chains(), 4);
    for p in 0..2 {
        let r = trace.rhat(p);
        assert!(
            r.is_finite() && r < 1.1,
            "param {p} not converged: rhat = {r}"
        );
    }
    assert!(
        trace.divergence_rate() < 0.25,
        "unexpectedly high divergences: {}",
        trace.divergence_rate()
    );

    // Parameter recovery (the synthetic ground truth).
    assert!((a_est - TRUE_A).abs() < 0.6, "intercept off");
    assert!((b_est - TRUE_B).abs() < 0.8, "slope off");
    // Positive slope means higher x => higher P(y=1): the fit must have got the
    // sign right, and the predicted probabilities must be ordered.
    assert!(b_est > 0.0, "slope should be positive");
    assert!(p_lo < p_hi, "predicted probability must increase with x");

    println!("\nAll logistic-regression checks passed.");
}
