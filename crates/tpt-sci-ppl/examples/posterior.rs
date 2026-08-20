//! # posterior.rs — a tour of the `tpt-sci-ppl` public API
//!
//! This example exercises a broad slice of the probabilistic-programming
//! surface on top of the from-scratch NUTS sampler:
//!
//! 1. **Model / DSL** — `ModelBuilder` with parameters declared several ways:
//!    a custom differentiable prior closure (`parameter`), a typed
//!    [`Gaussian`] prior (`gaussian_prior`), and bounded-support typed priors
//!    (`beta_prior`, `gamma_prior`). Likelihoods are written in terms of the
//!    `autodiff` `Variable`s and the observed `data` (`set_data` + `likelihood`).
//! 2. **Samplers** — `Model::fit` (2 dispersed chains), `Model::fit_chains`
//!    (N chains), and `Model::fit_from` (a single explicitly-seeded chain),
//!    plus the raw [`nuts_with_options`] entry point with a custom
//!    [`NutsOptions`] target.
//! 3. **Diagnostics** — the [`Trace`] API: `rhat` (split-R-hat), `ess`
//!    (effective sample size), `divergence_rate`, `mean`, `std`, `n_chains`.
//!
//! Everything is deterministic (fixed seeds) and fast. Watch for the recovered
//! posterior means of the linear-regression slope/intercept landing on the
//! synthetic ground truth, and the multi-chain split-R-hat sitting at ~1.0.

use tpt_math_prob_core::{Rng, SplitMix64};
use tpt_sci_ppl::{
    Beta, Gamma, Gaussian, GradientTape, ModelBuilder, NutsOptions, Trace, Variable,
    nuts_with_options,
};

/// Standard-normal draw via Box–Muller (the sampler's own `randn` is internal).
fn randn(rng: &mut impl Rng) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Differentiable N(0, 1) target for the raw NUTS demo (a HRTB function item,
/// which satisfies the `Target` trait unlike a closure).
fn std_normal_target<'t>(t: &'t GradientTape, v: &'t [Variable<'t>]) -> Variable<'t> {
    let mut s = t.constant(0.0);
    s += -0.5 * v[0] * v[0];
    s
}

fn main() {
    println!("=== tpt-sci-ppl API tour ===\n");

    // ---------------------------------------------------------------------
    // 1. Bayesian linear regression via the DSL.
    //
    //   y_i = a*x_i + b + noise,  noise ~ N(0, SIGMA)
    //   ground truth: a = 2.0 (slope), b = 1.0 (intercept)
    // ---------------------------------------------------------------------
    const N: usize = 40;
    const TRUE_SLOPE: f64 = 2.0;
    const TRUE_INTERCEPT: f64 = 1.0;
    const SIGMA: f64 = 0.5;

    let mut rng = SplitMix64::seed_from_u64(0x5EED);
    let xs: Vec<f64> = (0..N)
        .map(|i| -2.0 + 4.0 * i as f64 / (N - 1) as f64)
        .collect();
    // Pack the observed data as interleaved (x, y) pairs.
    let mut data = Vec::with_capacity(2 * N);
    for &x in &xs {
        let y = TRUE_SLOPE * x + TRUE_INTERCEPT + SIGMA * randn(&mut rng);
        data.push(x);
        data.push(y);
    }

    let mut m = ModelBuilder::new();
    // Slope: a weak Gaussian prior written by hand as a differentiable closure.
    m.parameter(|x: &Variable<'_>| {
        let x = *x;
        let z = (x - 0.0) / 5.0;
        -0.5 * z * z
    });
    // Intercept: the same idea, but from a typed `Gaussian` distribution.
    m.gaussian_prior(Gaussian::new(0.0, 5.0));
    m.set_data(data);
    m.likelihood(|t, v, d| {
        let slope = v[0];
        let intercept = v[1];
        let mut s = t.constant(0.0);
        for k in (0..d.len()).step_by(2) {
            let x = d[k];
            let y = d[k + 1];
            let pred = x * slope + intercept; // f64*Variable + Variable
            let z = (y - pred) / SIGMA; // f64 - Variable, then / f64
            s += -0.5 * z * z;
        }
        s
    });
    let model = m.build().expect("model should build");
    assert_eq!(model.dim(), 2);

    // --- 1a. Default fit (2 dispersed chains) -> meaningful split-R-hat. ---
    let trace: Trace = model.fit(&mut rng, 800).expect("fit should succeed");
    println!("Linear regression (default 2-chain fit):");
    println!(
        "  slope    = {:.3}  (truth {:.1},  ess = {:.0})",
        trace.mean(0),
        TRUE_SLOPE,
        trace.ess(0)
    );
    println!(
        "  intercept= {:.3}  (truth {:.1},  ess = {:.0})",
        trace.mean(1),
        TRUE_INTERCEPT,
        trace.ess(1)
    );
    println!(
        "  rhat     = [{:.4}, {:.4}]   divergence_rate = {:.3}",
        trace.rhat(0),
        trace.rhat(1),
        trace.divergence_rate()
    );
    println!(
        "  std      = [{:.3}, {:.3}]   chains = {}",
        trace.std(0),
        trace.std(1),
        trace.n_chains()
    );
    assert!((trace.mean(0) - TRUE_SLOPE).abs() < 0.3, "slope off");
    assert!(
        (trace.mean(1) - TRUE_INTERCEPT).abs() < 0.3,
        "intercept off"
    );

    // --- 1b. Explicit multi-chain fit, assert convergence. ---
    let multi: Trace = model
        .fit_chains(4, 800, &mut rng)
        .expect("multi-chain fit should succeed");
    let rhat_slope = multi.rhat(0);
    let rhat_int = multi.rhat(1);
    println!(
        "\nMulti-chain (4 chains x 800): rhat = [{:.4}, {:.4}], draws = {}, divergences = {}",
        rhat_slope,
        rhat_int,
        multi.n_draws(),
        multi.n_divergences()
    );
    assert!(rhat_slope < 1.05 && rhat_int < 1.05, "chains not converged");
    assert!((multi.mean(0) - TRUE_SLOPE).abs() < 0.3, "slope off");
    assert!(
        (multi.mean(1) - TRUE_INTERCEPT).abs() < 0.3,
        "intercept off"
    );

    // --- 1c. Single explicitly-seeded chain via fit_from. ---
    let single: Trace = model
        .fit_from(&[0.0, 0.0], &mut rng, 800)
        .expect("fit_from should succeed");
    println!(
        "\nSingle chain (fit_from): n_chains = {}, rhat = {:.3} (NaN expected for 1 chain), mean = [{:.3}, {:.3}]",
        single.n_chains(),
        single.rhat(0),
        single.mean(0),
        single.mean(1)
    );
    assert_eq!(single.n_chains(), 1);
    assert!(single.rhat(0).is_nan(), "rhat undefined for one chain");

    // ---------------------------------------------------------------------
    // 2. Bounded-support typed priors: Beta-Binomial posterior.
    //    7 successes, 3 failures, uniform Beta(1,1) prior -> mean 0.7.
    // ---------------------------------------------------------------------
    let mut bm = ModelBuilder::new();
    bm.beta_prior(Beta::uniform());
    bm.set_data(vec![7.0, 3.0]);
    bm.likelihood(|_t, v, d| {
        let s = d[0];
        let f = d[1];
        s * v[0].ln() + f * (1.0_f64 - v[0]).ln()
    });
    let beta_model = bm.build().expect("beta model should build");
    let beta_trace = beta_model
        .fit(&mut rng, 800)
        .expect("beta fit should succeed");
    println!(
        "\nBeta-Binomial: p = {:.3} (truth 0.70), rhat = {:.4}, ess = {:.0}",
        beta_trace.mean(0),
        beta_trace.rhat(0),
        beta_trace.ess(0)
    );
    assert!(
        (beta_trace.mean(0) - 0.7).abs() < 0.06,
        "beta posterior off"
    );

    // ---------------------------------------------------------------------
    // 3. Gamma prior on a rate with an Exponential likelihood.
    //    ground-truth rate = 3.0; Gamma(2,1) prior, Exponential data.
    // ---------------------------------------------------------------------
    const TRUE_RATE: f64 = 3.0;
    let mut grng = SplitMix64::seed_from_u64(0xBEEF);
    let exp_data: Vec<f64> = (0..30).map(|_| -grng.next_f64().ln() / TRUE_RATE).collect();
    let mut gm = ModelBuilder::new();
    gm.gamma_prior(Gamma::new(2.0, 1.0));
    gm.set_data(exp_data);
    gm.likelihood(|t, v, d| {
        let lam = v[0];
        let mut s = t.constant(0.0);
        for &x in d {
            s += lam.ln() - x * lam; // Variable - (f64 * Variable)
        }
        s
    });
    let gamma_model = gm.build().expect("gamma model should build");
    let gamma_trace = gamma_model
        .fit(&mut rng, 800)
        .expect("gamma fit should succeed");
    println!(
        "Gamma-Exponential: rate = {:.3} (truth {:.1}), rhat = {:.4}",
        gamma_trace.mean(0),
        TRUE_RATE,
        gamma_trace.rhat(0)
    );
    assert!(
        (gamma_trace.mean(0) - TRUE_RATE).abs() < 0.6,
        "gamma posterior off"
    );

    // ---------------------------------------------------------------------
    // 4. Raw sampler surface: NUTS directly on a differentiable target with
    //    custom NutsOptions (no DSL, no data).
    // ---------------------------------------------------------------------
    let opts = NutsOptions {
        step_size: 0.5,
        max_depth: 8,
        warmup: 200,
        target_accept: 0.65,
    };
    let mut rraw = SplitMix64::seed_from_u64(0xCAFE);
    let raw = nuts_with_options(&std_normal_target, &[0.0], &mut rraw, 500, opts)
        .expect("nuts should succeed");
    let raw_mean: f64 = raw.iter().map(|s| s[0]).sum::<f64>() / raw.len() as f64;
    println!(
        "\nRaw NUTS (NutsOptions, N(0,1) target): mean = {:.3} (truth 0.0), draws = {}",
        raw_mean,
        raw.len()
    );
    assert!(raw_mean.abs() < 0.15, "raw NUTS mean off");

    println!("\nAll API exercises passed.");
}
