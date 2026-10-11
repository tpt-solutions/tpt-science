//! Bayesian inference with NUTS: estimate the mean of a Gaussian from data.
//!
//! Model: `μ ~ N(0, 5²)`, and each observation is `x_i ~ N(μ, 1)`. The
//! posterior mean has a closed form, so the sampler's answer can be checked.

use tpt_math_prob_core::SplitMix64;
use tpt_sci::ppl::ModelBuilder;

fn main() {
    let data = vec![2.0, 3.0, 2.5, 3.5, 3.0];

    let mut model = ModelBuilder::new();
    model.gaussian_parameter(0.0, 5.0);
    model.set_data(data.clone());
    model.likelihood(|tape, vars, obs| {
        let mut s = tape.constant(0.0);
        for &x in obs {
            let z = vars[0] - x;
            s += -0.5 * z * z;
        }
        s
    });
    let model = model.build().expect("valid model");

    let mut rng = SplitMix64::seed_from_u64(1);
    let trace = model.fit(&mut rng, 1000).expect("sampling succeeds");

    // Closed-form posterior mean for a N(0, 5²) prior and unit noise.
    let n = data.len() as f64;
    let exact = data.iter().sum::<f64>() / (n + 1.0 / 25.0);

    println!("posterior mean of μ: {:.3} (exact {exact:.3})", trace.mean(0));
    println!("posterior sd of μ:   {:.3}", trace.std(0));
    println!("split R-hat:         {:.4}", trace.rhat(0));
    println!("divergence rate:     {:.4}", trace.divergence_rate());
    assert!((trace.mean(0) - exact).abs() < 0.15, "posterior mean drifted");
}
