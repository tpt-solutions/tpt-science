//! Bayesian inference for an ODE model: the likelihood gradient comes from
//! forward sensitivities, and NUTS samples the posterior of the decay rate.

use tpt_math_prob_core::SplitMix64;
use tpt_sci_ode::{Method, forward_sensitivities};
use tpt_sci_ppl::gradient::fit_with_gradient;

/// Observation times and noisy data for `y' = -k y`, `y(0) = 1`, `k = 0.8`.
fn synthetic_data(sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let times: Vec<f64> = (1..=20).map(|i| 0.25 * i as f64).collect();
    // Deterministic pseudo-noise with unit variance, centred to exactly zero mean
    // so that it does not bias the fit. The data are reproducible without
    // exposing the sampler's internal normal generator.
    let raw: Vec<f64> = (0..times.len())
        .map(|i| 3f64.sqrt() * (2.0 * (i as f64 * 12.9898).sin().fract() - 1.0))
        .collect();
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    let data = times
        .iter()
        .zip(&raw)
        .map(|(&t, &e)| (-0.8 * t).exp() + sigma * (e - mean))
        .collect();
    (times, data)
}

/// Log-posterior and gradient in the unconstrained parameter `θ = ln k`, with
/// the prior `θ ~ N(0, 2²)`. The chain rule maps `∂/∂k` to `∂/∂θ` via `k`.
fn log_posterior(
    theta: &[f64],
    times: &[f64],
    data: &[f64],
    sigma: f64,
) -> Option<(f64, Vec<f64>)> {
    let k = theta[0].exp();
    let t_last = *times.last()?;
    let sens = forward_sensitivities(
        |_t, y, p, dy| dy[0] = -p[0] * y[0],
        &[1.0],
        &[k],
        0.0,
        t_last,
        Method::Tsit45,
        1e-8,
        1e-10,
        Some(times),
    )
    .ok()?;
    let mut loglik = 0.0;
    let mut dloglik_dk = 0.0;
    for ((d, y), s) in data.iter().zip(&sens.states).zip(&sens.sensitivities) {
        let r = d - y[0];
        loglik += -0.5 * r * r / (sigma * sigma);
        dloglik_dk += r / (sigma * sigma) * s[0];
    }
    let prior = -0.125 * theta[0] * theta[0];
    let grad = dloglik_dk * k - 0.25 * theta[0];
    Some((loglik + prior, vec![grad]))
}

#[test]
fn posterior_of_decay_rate_recovers_truth() {
    let sigma = 0.05;
    let (times, data) = synthetic_data(sigma);
    let mut rng = SplitMix64::seed_from_u64(5);
    let trace = fit_with_gradient(
        1,
        |theta| log_posterior(theta, &times, &data, sigma),
        4,
        800,
        &mut rng,
    )
    .unwrap();

    // Posterior is for θ = ln k; the truth is ln 0.8.
    let truth = 0.8f64.ln();
    let mean = trace.mean(0);
    assert!(
        (mean - truth).abs() < 0.1,
        "posterior mean θ = {mean}, truth {truth}"
    );
    let rhat = trace.rhat(0);
    assert!(rhat < 1.1, "chains disagree: rhat {rhat}");
    assert!(
        trace.divergence_rate() < 0.05,
        "divergence rate {}",
        trace.divergence_rate()
    );
}
