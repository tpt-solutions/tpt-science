# tpt-sci-ppl

Probabilistic programming for the `tpt-science` pillar: a small model/DSL
layer and a from-scratch NUTS (No-U-Turn Sampler) Hamiltonian Monte Carlo
backend.

The original `spec.txt` planned to wrap `nuts-rs`; per the project's "build our
own internals" direction this crate implements NUTS directly on top of the
`tpt-math` substrate — reverse-mode gradients come from `tpt-math-autodiff-rev`
and randomness from `tpt-math-prob` — so there is no external sampler
dependency to audit or license-check.

Priors are expressed as reverse-mode differentiable closures (so the NUTS
gradient is exact). They can be written directly with the parameter `autodiff`
variable, or derived from a typed distribution in `tpt-math-prob-bayes`
(`Gaussian`, `Beta`, `Gamma`).

Depends on `tpt-math-prob-bayes`, `tpt-math-prob-core`, `tpt-math-autodiff-rev`
(all published).

## Example

```rust
use tpt_sci_ppl::ModelBuilder;
use tpt_math_prob_core::SplitMix64;

// mu ~ N(0, 5),  data | mu ~ N(mu, 1)
let data = [2.0, 3.0, 2.5, 3.5, 3.0];
let mut m = ModelBuilder::new();
m.gaussian_parameter(0.0, 5.0);
m.set_data(data.to_vec());
m.likelihood(|t, v, d| {
    let mut s = t.constant(0.0);
    for &x in d.iter() {
        let z = (v[0] - x) / 1.0;
        s = s + (-0.5 * z * z);
    }
    s
});
let model = m.build().unwrap();
let mut rng = SplitMix64::seed_from_u64(1);
let samples = model.fit(&mut rng, 500).unwrap();
let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
assert!((mean - 2.8).abs() < 0.6);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
