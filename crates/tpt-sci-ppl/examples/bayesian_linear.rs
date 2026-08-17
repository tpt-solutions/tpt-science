//! Bayesian linear regression: recover the slope and intercept of a noisy line
//! with the from-scratch NUTS sampler.
//!
//! Run with: `cargo run --example bayesian_linear -p tpt-sci-ppl`

use tpt_math_prob_core::SplitMix64;
use tpt_sci_ppl::{Gaussian, ModelBuilder};

fn main() {
    // y = 2 x + 1 + noise. Pack each observation as [x, y].
    let mut data = Vec::new();
    for x in 0..20 {
        let x = x as f64;
        let y = 2.0 * x + 1.0 + 0.2 * (x - 9.5).sin();
        data.push(x);
        data.push(y);
    }

    let mut m = ModelBuilder::new();
    m.gaussian_prior(Gaussian::new(0.0, 5.0)); // slope
    m.gaussian_prior(Gaussian::new(0.0, 5.0)); // intercept
    m.set_data(data);
    m.likelihood(|_t, v, d| {
        let mut s = _t.constant(0.0);
        // Fixed observation noise sigma = 0.5.
        for pair in d.chunks(2) {
            let x = pair[0];
            let y = pair[1];
            let pred = v[0] * x + v[1];
            let z = (y - pred) / 0.5;
            s += -0.5 * z * z;
        }
        s
    });

    let model = m.build().unwrap();
    let mut rng = SplitMix64::seed_from_u64(7);
    let trace = model.fit(&mut rng, 1000).unwrap();
    let samples = trace.combined_samples();

    let slope: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
    let intercept: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / samples.len() as f64;
    println!("recovered slope = {slope:.3} (truth 2.0)");
    println!("recovered intercept = {intercept:.3} (truth 1.0)");
}
