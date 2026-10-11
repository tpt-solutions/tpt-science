//! Parallel NUTS chains recover the same posterior as the serial sampler
//! (requires `rayon`).
#![cfg(feature = "rayon")]

use tpt_math_prob_core::SplitMix64;
use tpt_sci_ppl::gradient::fit_with_gradient_parallel;

#[test]
fn parallel_chains_recover_gaussian() {
    let mut rng = SplitMix64::seed_from_u64(21);
    let sd = 0.5;
    let trace = fit_with_gradient_parallel(
        1,
        move |th: &[f64]| Some((-0.5 * (th[0] / sd).powi(2), vec![-th[0] / (sd * sd)])),
        4,
        1000,
        &mut rng,
    )
    .unwrap();
    assert_eq!(trace.n_chains(), 4);
    assert!(trace.mean(0).abs() < 0.1, "mean {}", trace.mean(0));
    assert!((trace.std(0) - sd).abs() < 0.05, "std {}", trace.std(0));
    assert!(trace.rhat(0) < 1.05);
}
