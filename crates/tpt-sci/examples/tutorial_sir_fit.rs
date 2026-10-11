//! Tutorial: fit an SIR epidemic model to noisy case counts with NUTS.
//!
//! The model is
//!
//! ```text
//! S' = -β·S·I/N,    I' = β·S·I/N − γ·I,    R' = γ·I
//! ```
//!
//! Only the infected count `I(t)` is observed, with Gaussian noise. We infer
//! `β` (transmission) and `γ` (recovery) by sampling the posterior with NUTS.
//!
//! The sampler needs the gradient of the log-posterior with respect to the
//! parameters. Forward sensitivity analysis gives `∂I/∂β` and `∂I/∂γ` alongside
//! the ODE solution, and the chain rule then gives the gradient in the
//! log-parameters we sample over.
//!
//! Run with: `cargo run --release --example tutorial_sir_fit -p tpt-sci --features full`

use tpt_math_prob_core::SplitMix64;
use tpt_sci::ode::{Method, forward_sensitivities};
use tpt_sci::ppl::gradient::fit_with_gradient;

const POPULATION: f64 = 1000.0;
const SIGMA: f64 = 5.0; // observation noise on the case count

/// Infected-count trajectory and sensitivities for parameters `p = [β, γ]`.
///
/// Returns `(I(t_k), ∂I/∂β, ∂I/∂γ)` at each observation time.
fn infected_and_sensitivities(p: &[f64], times: &[f64]) -> Option<Vec<(f64, f64, f64)>> {
    let y0 = [POPULATION - 10.0, 10.0, 0.0];
    let t_last = *times.last()?;
    let sol = forward_sensitivities(
        |_t, y, p, dy| {
            let (s, i) = (y[0], y[1]);
            let infection = p[0] * s * i / POPULATION;
            dy[0] = -infection;
            dy[1] = infection - p[1] * i;
            dy[2] = p[1] * i;
        },
        &y0,
        p,
        0.0,
        t_last,
        Method::Tsit45,
        1e-8,
        1e-10,
        Some(times),
    )
    .ok()?;
    let np = p.len();
    Some(
        sol.states
            .iter()
            .zip(&sol.sensitivities)
            .map(|(y, s)| (y[1], s[np], s[np + 1]))
            .collect(),
    )
}

/// Log-posterior and gradient in `θ = (ln β, ln γ)`.
///
/// Priors: `θ ~ N(0, 2²)` on each log-parameter. The chain rule maps
/// `∂/∂p` to `∂/∂θ` by multiplying by `p`.
fn log_posterior(theta: &[f64], times: &[f64], counts: &[f64]) -> Option<(f64, Vec<f64>)> {
    let p = [theta[0].exp(), theta[1].exp()];
    let traj = infected_and_sensitivities(&p, times)?;
    let mut loglik = 0.0;
    let mut grad_p = [0.0, 0.0];
    for ((i_model, di_dbeta, di_dgamma), observed) in traj.iter().zip(counts) {
        let r = observed - i_model;
        loglik += -0.5 * r * r / (SIGMA * SIGMA);
        grad_p[0] += r / (SIGMA * SIGMA) * di_dbeta;
        grad_p[1] += r / (SIGMA * SIGMA) * di_dgamma;
    }
    let prior: f64 = theta.iter().map(|t| -0.125 * t * t).sum();
    let grad: Vec<f64> = (0..2).map(|j| grad_p[j] * p[j] - 0.25 * theta[j]).collect();
    Some((loglik + prior, grad))
}

/// Simulate noisy case counts from the true parameters `β = 0.4`, `γ = 0.2`.
fn synthetic_counts(times: &[f64]) -> Vec<f64> {
    let truth = infected_and_sensitivities(&[0.4, 0.2], times).expect("forward model should solve");
    // Deterministic, zero-mean pseudo-noise (see the ppl ODE-fit test for why
    // the mean is removed).
    let raw: Vec<f64> = (0..times.len())
        .map(|i| 3f64.sqrt() * (2.0 * (i as f64 * 12.9898).sin().fract() - 1.0))
        .collect();
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    truth
        .iter()
        .zip(&raw)
        .map(|((i, _, _), e)| i + SIGMA * (e - mean))
        .collect()
}

fn main() {
    // Step 1 — data. Weekly-style counts over 30 time units.
    let times: Vec<f64> = (1..=30).map(|d| d as f64).collect();
    let counts = synthetic_counts(&times);
    println!("observed infected counts (every 5th day):");
    for (t, c) in times.iter().zip(&counts).step_by(5) {
        println!("  day {t:>4.0}: {c:>7.1}");
    }

    // Step 2 — posterior. Four chains, 1000 draws each after warm-up.
    let mut rng = SplitMix64::seed_from_u64(2024);
    let trace = fit_with_gradient(
        2,
        |theta| log_posterior(theta, &times, &counts),
        4,
        1000,
        &mut rng,
    )
    .expect("sampling should succeed");

    // Step 3 — summarise on the natural scale. Means are of θ = ln p, so the
    // posterior median of p is exp(mean θ).
    let beta = trace.mean(0).exp();
    let gamma = trace.mean(1).exp();
    println!("\nposterior (θ = ln p, reported as exp of the mean):");
    println!("  β (transmission) ≈ {beta:.3}   (true 0.400)");
    println!("  γ (recovery)     ≈ {gamma:.3}   (true 0.200)");
    println!("  R₀ = β/γ         ≈ {:.2}   (true 2.00)", beta / gamma);
    println!(
        "  rhat = [{:.4}, {:.4}], divergence rate = {:.4}",
        trace.rhat(0),
        trace.rhat(1),
        trace.divergence_rate()
    );
}
