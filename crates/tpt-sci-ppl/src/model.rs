//! A small probabilistic-programming model/DSL built on top of the from-scratch
//! [`crate::nuts()`] sampler.
//!
//! A [`ModelBuilder`] collects parameter priors (expressed as reverse-mode
//! differentiable closures, so the NUTS gradient is exact) and a likelihood,
//! then [`ModelBuilder::build`] freezes them into a model and [`Model::fit`] runs NUTS
//! to draw posterior samples.
//!
//! ```rust
//! use tpt_sci_ppl::{ModelBuilder, Rng};
//! use tpt_math_prob_core::SplitMix64;
//!
//! // Posterior of mu given data ~ N(mu, 1) with a weak N(0, 5) prior.
//! let data = [2.0, 3.0, 2.5, 3.5, 3.0];
//! let mut m = ModelBuilder::new();
//! m.gaussian_parameter(0.0, 5.0);
//! m.set_data(data.to_vec());
//! m.likelihood(|_tape, v, d| {
//!     let mut s = _tape.constant(0.0);
//!     for &x in d.iter() {
//!         let z = (v[0] - x) / 1.0;
//!         s = s + (-0.5 * z * z);
//!     }
//!     s
//! });
//! let model = m.build().unwrap();
//! let mut rng = SplitMix64::seed_from_u64(1);
//! let trace = model.fit(&mut rng, 500).unwrap();
//! let samples = trace.combined_samples();
//! let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
//! assert!((mean - 2.8).abs() < 0.6);
//! ```

use tpt_math_autodiff_rev::{GradientTape, Variable};
use tpt_math_prob_core::Rng;

use crate::error::PplError;
use crate::nuts::{nuts_with_divergences, randn};
use crate::trace::Trace;
use crate::{Beta, Gamma, Gaussian};

/// A prior over one scalar parameter, written in terms of its `autodiff`
/// variable. Stored boxed so a `Model` can hold a heterogeneous list.
type PriorFn = Box<dyn for<'t> Fn(&'t Variable<'t>) -> Variable<'t>>;

/// A likelihood over all parameters and the observed data, written in terms of
/// `autodiff` variables.
type LikelihoodFn = Box<dyn for<'t> Fn(&'t GradientTape, &[Variable<'t>], &[f64]) -> Variable<'t>>;

/// Builder for a [`Model`].
pub struct ModelBuilder {
    dim: usize,
    priors: Vec<PriorFn>,
    likelihood: Option<LikelihoodFn>,
    data: Vec<f64>,
}

impl ModelBuilder {
    /// Create an empty model (zero parameters, no data).
    pub fn new() -> Self {
        Self {
            dim: 0,
            priors: Vec::new(),
            likelihood: None,
            data: Vec::new(),
        }
    }

    /// Add a parameter with an arbitrary differentiable prior closure.
    pub fn parameter<F>(&mut self, prior: F) -> &mut Self
    where
        F: for<'t> Fn(&'t Variable<'t>) -> Variable<'t> + 'static,
    {
        self.priors.push(Box::new(prior));
        self.dim += 1;
        self
    }

    /// Add a Gaussian prior `N(mean, std²)` (constant terms dropped — they do
    /// not affect sampling).
    pub fn gaussian_parameter(&mut self, mean: f64, std: f64) -> &mut Self {
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            let z = (x - mean) / std;
            -0.5 * z * z
        })
    }

    /// Add a Beta prior `Beta(a, b)` (for a parameter in `(0, 1)`).
    pub fn beta_parameter(&mut self, a: f64, b: f64) -> &mut Self {
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            (a - 1.0) * x.ln() + (b - 1.0) * (1.0_f64 - x).ln()
        })
    }

    /// Add a Gamma prior `Gamma(a, b)` (rate parameterisation, for a parameter
    /// in `(0, ∞)`).
    pub fn gamma_parameter(&mut self, a: f64, b: f64) -> &mut Self {
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            (a - 1.0) * x.ln() - b * x
        })
    }

    /// Add a Gaussian prior derived from a typed [`Gaussian`] distribution
    /// `N(mean, std²)` (from `tpt-math-prob-bayes`). Equivalent to
    /// [`ModelBuilder::gaussian_parameter`] but takes the distribution struct
    /// directly, so a prior can be constructed and validated once and reused
    /// across models.
    pub fn gaussian_prior(&mut self, d: Gaussian) -> &mut Self {
        let (mean, std) = (d.mean(), d.std());
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            let z = (x - mean) / std;
            -0.5 * z * z
        })
    }

    /// Add a Beta prior derived from a typed [`Beta`] distribution on `(0, 1)`
    /// (from `tpt-math-prob-bayes`). Equivalent to
    /// [`ModelBuilder::beta_parameter`] but takes the distribution struct.
    pub fn beta_prior(&mut self, d: Beta) -> &mut Self {
        let (a, b) = (d.alpha(), d.beta());
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            (a - 1.0) * x.ln() + (b - 1.0) * (1.0_f64 - x).ln()
        })
    }

    /// Add a Gamma prior derived from a typed [`Gamma`] distribution in
    /// shape/rate parameterisation on `(0, ∞)` (from `tpt-math-prob-bayes`).
    /// Equivalent to [`ModelBuilder::gamma_parameter`] but takes the
    /// distribution struct.
    pub fn gamma_prior(&mut self, d: Gamma) -> &mut Self {
        let (a, b) = (d.shape(), d.rate());
        self.parameter(move |x: &Variable<'_>| {
            let x = *x;
            (a - 1.0) * x.ln() - b * x
        })
    }

    /// Set the observed data passed to the likelihood closure.
    pub fn set_data(&mut self, data: Vec<f64>) -> &mut Self {
        self.data = data;
        self
    }

    /// Set the likelihood `log p(data | params)`, written in terms of the
    /// parameter `autodiff` variables and the observed `data`.
    pub fn likelihood<F>(&mut self, f: F) -> &mut Self
    where
        F: for<'t> Fn(&'t GradientTape, &[Variable<'t>], &[f64]) -> Variable<'t> + 'static,
    {
        self.likelihood = Some(Box::new(f));
        self
    }

    /// Freeze the model into a [`Model`].
    ///
    /// # Errors
    ///
    /// Returns [`PplError::InvalidOption`] if no parameters were added.
    pub fn build(self) -> Result<Model, PplError> {
        if self.dim == 0 {
            return Err(PplError::InvalidOption("model has no parameters".into()));
        }
        Ok(Model {
            dim: self.dim,
            priors: self.priors,
            likelihood: self.likelihood,
            data: self.data,
        })
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A frozen probabilistic model: parameter priors, likelihood, and data.
pub struct Model {
    dim: usize,
    priors: Vec<PriorFn>,
    likelihood: Option<LikelihoodFn>,
    data: Vec<f64>,
}

impl Model {
    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Build the differentiable target and draw `n_samples` posterior samples
    /// with NUTS, starting from `0.5` in every coordinate (a safe interior
    /// point for both bounded and unbounded priors).
    ///
    /// This is a convenience wrapper that runs [`Model::fit_chains`] with two
    /// dispersed chains, so the returned [`Trace`] always carries a meaningful
    /// split-R-hat ([`Trace::rhat`] is `NaN` for a single chain). Call
    /// `fit_chains` directly for finer control over the number of chains or
    /// `fit_from` for a single, explicitly-seeded chain.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying sampler.
    pub fn fit(&self, rng: &mut impl Rng, n_samples: usize) -> Result<Trace, PplError> {
        self.fit_chains(2, n_samples, rng)
    }

    /// Draw samples starting from an explicit initial point.
    ///
    /// # Errors
    ///
    /// Returns [`PplError::DimensionMismatch`] if `initial.len()` does not
    /// match the model dimension.
    pub fn fit_from(
        &self,
        initial: &[f64],
        rng: &mut impl Rng,
        n_samples: usize,
    ) -> Result<Trace, PplError> {
        if initial.len() != self.dim {
            return Err(PplError::DimensionMismatch {
                expected: self.dim,
                got: initial.len(),
            });
        }
        // A callable that is higher-ranked over the tape lifetime: it can be
        // invoked with `&'t GradientTape` and `&'t [Variable<'t>]` for *any*
        // `'t`, which is exactly what the NUTS gradient machinery requires.
        let target = ModelTarget {
            priors: &self.priors,
            likelihood: &self.likelihood,
            data: &self.data,
        };
        let (samples, n_div) = nuts_with_divergences(&target, initial, rng, n_samples)?;
        Ok(Trace::new(vec![samples], n_div))
    }

    /// Draw `n_samples` posterior samples from `n_chains` independently
    /// initialised NUTS chains. The chains start from dispersed points around
    /// `0.5` so that [`Trace::rhat`] (split-R-hat across chains) is a meaningful
    /// convergence diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`PplError::InvalidOption`] if `n_chains == 0`, or any error from
    /// the underlying sampler.
    pub fn fit_chains(
        &self,
        n_chains: usize,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<Trace, PplError> {
        if n_chains == 0 {
            return Err(PplError::InvalidOption("n_chains must be > 0".into()));
        }
        let base = vec![0.5; self.dim];
        let target = ModelTarget {
            priors: &self.priors,
            likelihood: &self.likelihood,
            data: &self.data,
        };
        let mut chains = Vec::with_capacity(n_chains);
        let mut n_div = 0usize;
        for _ in 0..n_chains {
            // Disperse the start so the chains explore from different regions;
            // a single shared start would make R-hat trivially ~1.
            let start: Vec<f64> = base.iter().map(|&v| v + 0.1 * randn(rng)).collect();
            let (samples, d) = nuts_with_divergences(&target, &start, rng, n_samples)?;
            n_div += d;
            chains.push(samples);
        }
        Ok(Trace::new(chains, n_div))
    }
}

/// Bridges a frozen model's priors/likelihood/data into a differentiable
/// target usable by [`nuts`]. The [`Target`] trait is implemented with a
/// single higher-ranked method so the autodiff tape lifetime is free, which
/// keeps this on stable Rust (no manual `Fn`/`extern "rust-call"` impls).
struct ModelTarget<'a> {
    priors: &'a [PriorFn],
    likelihood: &'a Option<LikelihoodFn>,
    data: &'a [f64],
}

impl<'a> crate::nuts::Target for ModelTarget<'a> {
    fn eval<'t>(&self, tape: &'t GradientTape, vars: &'t [Variable<'t>]) -> Variable<'t> {
        let mut s = tape.constant(0.0);
        for (i, p) in self.priors.iter().enumerate() {
            s += p(&vars[i]);
        }
        if let Some(l) = self.likelihood {
            s += l(tape, vars, self.data);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_prob_core::SplitMix64;

    #[test]
    fn gaussian_model_recovers_posterior_mean() {
        // mu ~ N(0, 5),  data | mu ~ N(mu, 1)
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
        let trace = model.fit(&mut rng, 800).unwrap();
        let samples = trace.combined_samples();
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        // Data mean is 2.8; weak prior lets the posterior sit near it.
        assert_abs_diff_eq!(mean, 2.8, epsilon = 0.6);
        assert!(mean > 1.0 && mean < 4.0);
    }

    #[test]
    fn fit_from_rejects_wrong_dimension() {
        let mut m = ModelBuilder::new();
        m.gaussian_parameter(0.0, 1.0);
        let model = m.build().unwrap();
        let mut rng = SplitMix64::seed_from_u64(1);
        assert!(matches!(
            model.fit_from(&[0.0, 0.0], &mut rng, 10),
            Err(PplError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn typed_gaussian_prior_matches_raw_prior() {
        // A Gaussian prior built from the typed `tpt-math-prob-bayes`
        // distribution must recover the same posterior as the raw
        // `(mean, std)` prior.
        let data = [2.0, 3.0, 2.5, 3.5, 3.0];
        let mut m = ModelBuilder::new();
        m.gaussian_prior(Gaussian::new(0.0, 5.0));
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
        let trace = model.fit(&mut rng, 800).unwrap();
        let samples = trace.combined_samples();
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert_abs_diff_eq!(mean, 2.8, epsilon = 0.6);
    }

    #[test]
    fn typed_beta_prior_recovers_posterior() {
        // Posterior of p (a rate in (0, 1)) given 7 successes / 3 failures
        // under a uniform Beta(1, 1) prior. The posterior mean is 0.7.
        let mut m = ModelBuilder::new();
        m.beta_prior(Beta::uniform());
        m.set_data(vec![7.0, 3.0]);
        m.likelihood(|_t, v, d| {
            let s = d[0];
            let f = d[1];
            s * v[0].ln() + f * (1.0_f64 - v[0]).ln()
        });
        let model = m.build().unwrap();
        let mut rng = SplitMix64::seed_from_u64(42);
        let trace = model.fit(&mut rng, 2000).unwrap();
        let samples = trace.combined_samples();
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert_abs_diff_eq!(mean, 0.7, epsilon = 0.06);
    }

    #[test]
    fn multi_chain_fit_reports_diagnostics() {
        // mu ~ N(0, 5), data | mu ~ N(mu, 1), data mean 2.8.
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
        let mut rng = SplitMix64::seed_from_u64(123);
        let trace = model.fit_chains(4, 800, &mut rng).unwrap();
        assert_eq!(trace.n_chains(), 4);
        assert_eq!(trace.dim(), 1);
        // Converged chains give split-R-hat near 1.
        let rhat = trace.rhat(0);
        assert!(rhat.is_finite() && rhat < 1.1, "rhat = {rhat}");
        // ESS should be a positive fraction of the raw draw count.
        let ess = trace.ess(0);
        assert!(
            ess > 0.0 && ess < trace.n_draws() as f64 + 1.0,
            "ess = {ess}"
        );
        // Divergence rate is well defined (0.0 for this well-behaved target).
        assert!((0.0..=1.0).contains(&trace.divergence_rate()));
        // The posterior mean is recovered.
        assert_abs_diff_eq!(trace.mean(0), 2.8, epsilon = 0.6);
    }

    #[test]
    fn fit_chains_rejects_zero_chains() {
        let mut m = ModelBuilder::new();
        m.gaussian_parameter(0.0, 1.0);
        let model = m.build().unwrap();
        let mut rng = SplitMix64::seed_from_u64(1);
        assert!(matches!(
            model.fit_chains(0, 10, &mut rng),
            Err(PplError::InvalidOption { .. })
        ));
    }
}
