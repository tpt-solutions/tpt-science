//! Gaussian posterior via NUTS, with convergence diagnostics.
use tpt_math_prob_core::SplitMix64;
use tpt_sci_ppl::{ModelBuilder, Trace};

fn main() {
    let data = [2.0, 3.0, 2.5, 3.5, 3.0];
    let mut m = ModelBuilder::new();
    m.gaussian_parameter(0.0, 5.0);
    m.set_data(data.to_vec());
    m.likelihood(|t, v, d| {
        let mut s = t.constant(0.0);
        for &x in d.iter() {
            let z = (v[0] - x) / 1.0;
            s += -0.5 * z * z;
        }
        s
    });
    let model = m.build().unwrap();
    let mut rng = SplitMix64::seed_from_u64(1);

    let trace: Trace = model.fit(&mut rng, 1000).unwrap();
    println!(
        "Posterior mean ~= {:.3} (data mean = 2.8), ESS = {:.1}",
        trace.mean(0),
        trace.ess(0)
    );

    let multi = model.fit_chains(4, 1000, &mut rng).unwrap();
    println!(
        "4-chain R-hat = {:.4} (~1 means converged), divergence rate = {:.3}",
        multi.rhat(0),
        multi.divergence_rate()
    );
}
