//! Sampling from log-densities whose gradients come from outside the tape.
//!
//! The autodiff tape records only elementary operations, so a quantity computed
//! by a black-box solver (an ODE integration, say) has no tape node for its
//! parameter dependence. If the caller can supply the value *and* the gradient
//! of the log-density, it can still be sampled with NUTS by **gradient
//! injection**: at each evaluation, with `g = ∂L/∂θ` and `L` the value,
//!
//! ```text
//! surrogate(θ) = (L − g·θ₀) + g·θ        (constant offset + linear term)
//! ```
//!
//! has value `L` at the current `θ` and gradient exactly `g`, so the sampler
//! sees the right density and the right Hamiltonian dynamics. The offset is a
//! tape constant, so it contributes nothing to the derivative.
//!
//! [`fit_with_gradient`] runs this for several dispersed chains. The typical
//! source of `g` is forward sensitivity analysis:
//! `tpt_sci_ode::forward_sensitivities` returns `∂y/∂p`, and the chain rule
//! gives the likelihood gradient.

use tpt_math_autodiff_rev::{GradientTape, Variable};

use crate::PplError;
use crate::nuts::{Target, nuts_with_divergences, randn};
use crate::trace::Trace;
use tpt_math_prob_core::Rng;

/// A log-density, with value and gradient, as a [`Target`].
///
/// `logp_and_grad` returns `Some((L, ∂L/∂θ))`, or `None` where the density is
/// not defined (outside the support, or a failed inner solve). `None` is
/// treated as a point of zero probability, which NUTS rejects.
struct GradientTarget<F> {
    logp_and_grad: F,
}

impl<F> Target for GradientTarget<F>
where
    F: Fn(&[f64]) -> Option<(f64, Vec<f64>)>,
{
    fn eval<'t>(&self, tape: &'t GradientTape, vars: &'t [Variable<'t>]) -> Variable<'t> {
        let theta: Vec<f64> = vars.iter().map(|v| v.value()).collect();
        let (value, grad) = match (self.logp_and_grad)(&theta) {
            Some((value, grad)) => (value, grad),
            None => (f64::NEG_INFINITY, vec![0.0; theta.len()]),
        };
        let linear_at_theta: f64 = theta.iter().zip(&grad).map(|(t, g)| t * g).sum();
        let mut s = tape.constant(value - linear_at_theta);
        for (v, g) in vars.iter().zip(&grad) {
            s += *v * *g;
        }
        s
    }
}

/// Sample `n_samples` draws from `n_chains` NUTS chains on a log-density given
/// by `logp_and_grad`, returning a [`Trace`] with the usual diagnostics
/// (split-R-hat, ESS, divergences).
///
/// `dim` is the number of unconstrained parameters. Chains start dispersed
/// around `0`, as in [`Model::fit_chains`](crate::Model::fit_chains).
///
/// # Errors
///
/// Returns [`PplError::InvalidOption`] if `n_chains == 0`, or any error from
/// the sampler (for example a non-finite starting density).
///
/// # Examples
///
/// Sampling a standard normal from its value and gradient:
///
/// ```
/// use tpt_math_prob_core::SplitMix64;
/// use tpt_sci_ppl::gradient::fit_with_gradient;
///
/// let mut rng = SplitMix64::seed_from_u64(3);
/// let trace = fit_with_gradient(
///     1,
///     |theta: &[f64]| Some((-0.5 * theta[0] * theta[0], vec![-theta[0]])),
///     2,
///     500,
///     &mut rng,
/// )
/// .unwrap();
/// assert!(trace.mean(0).abs() < 0.3);
/// ```
pub fn fit_with_gradient<F>(
    dim: usize,
    logp_and_grad: F,
    n_chains: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Result<Trace, PplError>
where
    F: Fn(&[f64]) -> Option<(f64, Vec<f64>)>,
{
    if n_chains == 0 {
        return Err(PplError::InvalidOption("n_chains must be > 0".into()));
    }
    let target = GradientTarget { logp_and_grad };
    let mut chains = Vec::with_capacity(n_chains);
    let mut n_div = 0usize;
    for _ in 0..n_chains {
        // Disperse the starts so that R-hat is informative.
        let start: Vec<f64> = (0..dim).map(|_| 0.1 * randn(rng)).collect();
        let (samples, d) = nuts_with_divergences(&target, &start, rng, n_samples)?;
        n_div += d;
        chains.push(samples);
    }
    Ok(Trace::new(chains, n_div))
}

/// Samples and divergence count from one chain.
#[cfg(feature = "rayon")]
type ChainResult = (Vec<Vec<f64>>, usize);

/// Parallel version of [`fit_with_gradient`]: the chains run on separate
/// threads. Each chain's generator is seeded from `rng` before any chain starts,
/// so the result is deterministic for a given seed. It does not depend on
/// scheduling, but it differs from [`fit_with_gradient`], which threads one
/// generator through every chain in turn.
///
/// `logp_and_grad` must be `Sync`, since every thread evaluates it.
///
/// # Errors
///
/// As [`fit_with_gradient`].
#[cfg(feature = "rayon")]
pub fn fit_with_gradient_parallel<F>(
    dim: usize,
    logp_and_grad: F,
    n_chains: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Result<Trace, PplError>
where
    F: Fn(&[f64]) -> Option<(f64, Vec<f64>)> + Sync,
{
    use rayon::prelude::*;
    use tpt_math_prob_core::SplitMix64;

    if n_chains == 0 {
        return Err(PplError::InvalidOption("n_chains must be > 0".into()));
    }
    let seeds: Vec<u64> = (0..n_chains).map(|_| rng.next_u64()).collect();
    let results: Result<Vec<ChainResult>, PplError> = seeds
        .par_iter()
        .map(|&seed| {
            let mut chain_rng = SplitMix64::seed_from_u64(seed);
            let target = GradientTarget {
                logp_and_grad: &logp_and_grad,
            };
            let start: Vec<f64> = (0..dim).map(|_| 0.1 * randn(&mut chain_rng)).collect();
            nuts_with_divergences(&target, &start, &mut chain_rng, n_samples)
        })
        .collect();
    let mut chains = Vec::with_capacity(n_chains);
    let mut n_div = 0usize;
    for (samples, d) in results? {
        n_div += d;
        chains.push(samples);
    }
    Ok(Trace::new(chains, n_div))
}
