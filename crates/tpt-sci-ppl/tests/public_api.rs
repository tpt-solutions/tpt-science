//! End-to-end inference check: NUTS recovers the posterior mean of a Gaussian
//! location model with a known closed form.

use tpt_math_prob_core::SplitMix64;
use tpt_sci_ppl::ModelBuilder;

#[test]
fn gaussian_location_posterior_mean_matches_closed_form() {
    // mu ~ N(0, 5²), data_i | mu ~ N(mu, 1). Posterior mean is
    // (Σx / σ² + 0 / 25) / (n / σ² + 1 / 25) with σ = 1.
    let data = [2.0, 3.0, 2.5, 3.5, 3.0];
    let n = data.len() as f64;
    let sum: f64 = data.iter().sum();
    let exact_mean = (sum / 1.0) / (n / 1.0 + 1.0 / 25.0);

    let mut m = ModelBuilder::new();
    m.gaussian_parameter(0.0, 5.0);
    m.set_data(data.to_vec());
    m.likelihood(|t, v, d| {
        let mut s = t.constant(0.0);
        for &x in d.iter() {
            let z = v[0] - x;
            s += -0.5 * z * z;
        }
        s
    });
    let model = m.build().unwrap();
    let mut rng = SplitMix64::seed_from_u64(1);
    let trace = model.fit(&mut rng, 1000).unwrap();
    let samples = trace.combined_samples();
    let mean = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
    assert!(
        (mean - exact_mean).abs() < 0.15,
        "mean {mean} vs exact {exact_mean}"
    );
    let rhat = trace.rhat(0);
    assert!(rhat < 1.1, "chains did not mix: rhat {rhat}");
}
