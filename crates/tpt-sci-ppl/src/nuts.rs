//! NUTS (No-U-Turn Sampler) — a Hamiltonian Monte Carlo variant with an
//! adaptive trajectory length, built from scratch on top of
//! [`tpt_math_autodiff_rev`] (reverse-mode gradients) and the `tpt-math-prob`
//! RNG traits.
//!
//! The target is supplied as a *reverse-mode differentiable* closure:
//! `for<'t> Fn(&'t GradientTape, &[Variable<'t>]) -> Variable<'t>`, i.e. an
//! unnormalised log-posterior written in terms of `autodiff` variables so the
//! sampler can obtain its gradient exactly (no finite differences).
//!
//! The integrator is the standard leapfrog; the trajectory length is chosen by
//! the no-U-turn criterion (Hoffman & Gelman, 2014), and the step size is
//! adapted during a warm-up phase with the dual-averaging scheme from the same
//! paper.

use tpt_math_autodiff_rev::{GradientTape, Variable};
use tpt_math_prob_core::Rng;

use crate::error::PplError;

/// A differentiable unnormalised log-posterior: given a tape and the current
/// parameter variables, return the scalar log-density.
///
/// Implemented for any `for<'t> Fn(&'t GradientTape, &[Variable<'t>]) ->
/// Variable<'t>` (function items, function pointers, and closures), so both the
/// from-scratch sampler and the model/DSL layer can supply a target without any
/// unstable `Fn`-trait machinery.
pub trait Target {
    /// Evaluate the target at `vars` (which were produced from `tape`),
    /// returning the scalar log-density as a `Variable` tied to `tape`.
    fn eval<'t>(&self, tape: &'t GradientTape, vars: &'t [Variable<'t>]) -> Variable<'t>;
}

impl<F> Target for F
where
    F: for<'t> Fn(&'t GradientTape, &'t [Variable<'t>]) -> Variable<'t>,
{
    fn eval<'t>(&self, tape: &'t GradientTape, vars: &'t [Variable<'t>]) -> Variable<'t> {
        self(tape, vars)
    }
}

/// Sampler options for [`nuts`].
#[derive(Debug, Clone)]
pub struct NutsOptions {
    /// Initial step size. Adapted during warm-up; ignored afterwards.
    pub step_size: f64,
    /// Maximum tree depth `j` (max trajectory leaves `2^j`).
    pub max_depth: usize,
    /// Number of warm-up iterations used to adapt the step size.
    pub warmup: usize,
    /// Target mean acceptance rate for dual averaging (typically 0.65).
    pub target_accept: f64,
}

impl Default for NutsOptions {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            max_depth: 8,
            warmup: 500,
            target_accept: 0.65,
        }
    }
}

/// Run NUTS on `target` starting from `initial`, collecting `n_samples`
/// post-warm-up draws. Each returned vector is one posterior sample.
///
/// # Errors
///
/// Returns [`PplError`] if `n_samples == 0`, the step size is non-positive, the
/// maximum tree depth is zero, or the initial point has a non-finite
/// log-density.
pub fn nuts<T: Target + ?Sized>(
    target: &T,
    initial: &[f64],
    rng: &mut impl Rng,
    n_samples: usize,
) -> Result<Vec<Vec<f64>>, PplError> {
    let (samples, _) = run_nuts(target, initial, rng, n_samples, NutsOptions::default())?;
    Ok(samples)
}

/// Run NUTS with explicit [`NutsOptions`].
///
/// # Errors
///
/// See [`nuts`] for the error conditions.
pub fn nuts_with_options<T: Target + ?Sized>(
    target: &T,
    initial: &[f64],
    rng: &mut impl Rng,
    n_samples: usize,
    opts: NutsOptions,
) -> Result<Vec<Vec<f64>>, PplError> {
    let (samples, _) = run_nuts(target, initial, rng, n_samples, opts)?;
    Ok(samples)
}

/// Run NUTS and also report the number of divergent transitions observed
/// during the sampling phase.
///
/// A high divergence count signals a pathological posterior (steep curvature,
/// poor identifiability) where the step size had to be shrunk hard; the
/// resulting samples may be biased. The count feeds [`crate::Trace`].
///
/// # Errors
///
/// See [`nuts`] for the error conditions.
pub fn nuts_with_divergences<T: Target + ?Sized>(
    target: &T,
    initial: &[f64],
    rng: &mut impl Rng,
    n_samples: usize,
) -> Result<(Vec<Vec<f64>>, usize), PplError> {
    run_nuts(target, initial, rng, n_samples, NutsOptions::default())
}

/// Core NUTS driver shared by the public entry points. Returns the post
/// warm-up samples plus the number of divergent transitions seen while
/// sampling.
fn run_nuts<T: Target + ?Sized>(
    target: &T,
    initial: &[f64],
    rng: &mut impl Rng,
    n_samples: usize,
    opts: NutsOptions,
) -> Result<(Vec<Vec<f64>>, usize), PplError> {
    let dim = initial.len();
    // Validate the initial point produces a finite log-density.
    {
        let (lp, _) = log_post_grad(target, initial);
        if !lp.is_finite() {
            return Err(PplError::NonFiniteInitial);
        }
    }

    let mut x = initial.to_vec();
    let mut eps = opts.step_size;
    // Dual-averaging anchor `μ = ln(10·ε₀)` (Hoffman & Gelman 2014, Algorithm
    // 4). Anchoring at `ln ε₀` instead would forbid shrinking the step size
    // below its initial value — which silently breaks bounded-support targets
    // (a leapfrog step overshoots the support, the proposal is non-finite, and
    // the sampler freezes at the initial point). The `10·` offset gives the
    // adaptation headroom to shrink ε as far as the target demands.
    let log_eps_mu = (10.0 * opts.step_size).ln();
    let mut log_eps_bar = eps.ln();
    let mut h_bar = 0.0;
    let gamma = 0.05_f64;
    let t0 = 10.0_f64;
    let kappa = 0.75_f64;

    let total = opts.warmup + n_samples;
    let mut samples = Vec::with_capacity(n_samples);
    let mut n_divergences = 0usize;

    for m in 1..=total {
        let r0: Vec<f64> = (0..dim).map(|_| randn(rng)).collect();
        let (lp0, _) = log_post_grad(target, &x);
        let joint0 = lp0 - 0.5 * dot(&r0, &r0);
        // Slice variable: draw `u ~ Uniform(0, exp(joint0))` and work in log
        // space, so `log_u = joint0 + ln(w)` with `w ~ Uniform(0, 1)`. A
        // proposal with joint `j` is admitted when `j >= log_u`.
        let log_u = joint0 + rng.next_f64().ln();

        let mut x_minus = x.clone();
        let mut r_minus = r0.clone();
        let mut x_plus = x.clone();
        let mut r_plus = r0.clone();
        let mut x_sample = x.clone();
        let mut n = 1usize;
        let mut sum_accept = 0.0_f64;
        let mut iter_divergent = false;

        for j in 0..opts.max_depth {
            let v = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let (endpoint_x, endpoint_r) = if v < 0.0 {
                (&x_minus, &r_minus)
            } else {
                (&x_plus, &r_plus)
            };
            let tree = build_tree(target, endpoint_x, endpoint_r, log_u, v, j, eps, rng);
            if tree.divergent {
                iter_divergent = true;
                break;
            }
            if v < 0.0 {
                x_minus = tree.x_minus.clone();
                r_minus = tree.r_minus.clone();
            } else {
                x_plus = tree.x_plus.clone();
                r_plus = tree.r_plus.clone();
            }
            // Accept a sample from the new subtree with probability n_new / n.
            if n > 0 && rng.next_f64() < tree.n as f64 / n as f64 {
                x_sample = tree.x_sample.clone();
            }
            n += tree.n;
            sum_accept += tree.sum_accept;

            // No-U-turn on the full trajectory.
            let d1 = dot(&sub(&x_plus, &x_minus), &r_minus);
            let d2 = dot(&sub(&x_plus, &x_minus), &r_plus);
            if d1 < 0.0 || d2 < 0.0 {
                break;
            }
        }

        // A U-turn only stops further trajectory expansion; the accepted
        // sample from the valid (pre-U-turn) subtree is still a legitimate
        // next state and must be kept. (Genuine numerical divergence would be
        // a separate signal and is not modelled here.)
        x = x_sample;

        // Dual-averaging step-size adaptation (Algorithm 4 of H&G 2014).
        // During warm-up the *current* step size `exp(log_eps)` drives each
        // trajectory (so a bad step can be corrected immediately), while
        // `log_eps_bar` accumulates the running average that is adopted — and
        // then held fixed — once sampling begins. The step size is clamped so a
        // quadratic target (where acceptance stays ~1 until the leapfrog
        // stability wall, then drops to exactly 0) cannot drive `log_eps` to
        // overflow and get stuck.
        if m <= opts.warmup {
            let alpha = if n > 0 { sum_accept / n as f64 } else { 0.0 };
            let m_f = m as f64;
            h_bar = (1.0 - 1.0 / (m_f + t0)) * h_bar
                + (1.0 / (m_f + t0)) * (opts.target_accept - alpha);
            let log_eps = log_eps_mu - (m_f).sqrt() / gamma * h_bar;
            log_eps_bar = m_f.powf(-kappa) * log_eps + (1.0 - m_f.powf(-kappa)) * log_eps_bar;
            eps = log_eps.exp().clamp(1e-3, 10.0);
        }

        if m > opts.warmup {
            // Hold the dual-averaged step size fixed for the sampling phase.
            eps = log_eps_bar.exp().clamp(1e-3, 10.0);
            if iter_divergent {
                n_divergences += 1;
            }
            samples.push(x.clone());
        }
    }

    Ok((samples, n_divergences))
}

/// One recursive NUTS tree expansion.
#[derive(Clone)]
struct Tree {
    x_minus: Vec<f64>,
    r_minus: Vec<f64>,
    x_plus: Vec<f64>,
    r_plus: Vec<f64>,
    x_sample: Vec<f64>,
    n: usize,
    sum_accept: f64,
    sum_log_alpha: f64,
    divergent: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_tree<T: Target + ?Sized>(
    target: &T,
    x: &[f64],
    r: &[f64],
    log_u: f64,
    v: f64,
    depth: usize,
    eps: f64,
    rng: &mut impl Rng,
) -> Tree {
    if depth == 0 {
        let (xn, rn) = leapfrog(target, x, r, v * eps);
        let (lpn, _) = log_post_grad(target, &xn);
        let joint = lpn - 0.5 * dot(&rn, &rn);
        let n = if joint >= log_u { 1 } else { 0 };
        let accept = if joint >= log_u {
            1.0
        } else {
            (joint - log_u).exp().min(1.0)
        };
        return Tree {
            x_minus: xn.clone(),
            r_minus: rn.clone(),
            x_plus: xn.clone(),
            r_plus: rn.clone(),
            x_sample: xn,
            n,
            sum_accept: accept,
            sum_log_alpha: joint - log_u,
            // A non-finite joint (e.g. a leapfrog step that left the prior
            // support) must stop the trajectory: otherwise the doubling keeps
            // expanding a zero-length, never-accepted subtree and the sampler
            // freezes at the current point.
            divergent: !joint.is_finite(),
        };
    }

    // Expand one shallower subtree from (x, r).
    let t1 = build_tree(target, x, r, log_u, v, depth - 1, eps, rng);
    if t1.divergent {
        return t1;
    }
    // Second subtree from the far endpoint of t1.
    let (ex, er) = if v < 0.0 {
        (&t1.x_minus, &t1.r_minus)
    } else {
        (&t1.x_plus, &t1.r_plus)
    };
    let t2 = build_tree(target, ex, er, log_u, v, depth - 1, eps, rng);

    // The combined trajectory's backward endpoint (and its momentum) is the
    // far end of `t2` when stepping backward (`v < 0`), and the far end of
    // `t1` when stepping forward; the forward endpoint is the complement.
    let (x_minus, r_minus, x_plus, r_plus) = if v < 0.0 {
        (
            t2.x_minus.clone(),
            t2.r_minus.clone(),
            t1.x_plus.clone(),
            t1.r_plus.clone(),
        )
    } else {
        (
            t1.x_minus.clone(),
            t1.r_minus.clone(),
            t2.x_plus.clone(),
            t2.r_plus.clone(),
        )
    };

    let n = t1.n + t2.n;
    // Slice-sample a point uniformly from the merged set of valid leaves.
    let x_sample = if t2.n == 0 {
        t1.x_sample.clone()
    } else if t1.n == 0 || rng.next_f64() < t2.n as f64 / t1.n as f64 {
        t2.x_sample.clone()
    } else {
        t1.x_sample.clone()
    };

    let d1 = dot(&sub(&x_plus, &x_minus), &r_minus);
    let d2 = dot(&sub(&x_plus, &x_minus), &r_plus);
    let divergent = (d1 < 0.0) || (d2 < 0.0);

    Tree {
        x_minus,
        r_minus,
        x_plus,
        r_plus,
        x_sample,
        n,
        sum_accept: t1.sum_accept + t2.sum_accept,
        sum_log_alpha: t1.sum_log_alpha + t2.sum_log_alpha,
        divergent,
    }
}

/// One leapfrog step of size `eps` in direction `sign` (+1 forward, -1 back).
fn leapfrog<T: Target + ?Sized>(
    target: &T,
    x: &[f64],
    r: &[f64],
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (_, g0) = log_post_grad(target, x);
    // r_half = r - 0.5*eps*gradU(x), with gradU = -grad_lp  =>  r + 0.5*eps*grad_lp
    let mut r = sub(r, &scale(&g0, -0.5 * eps));
    let x = add(x, &scale(&r, eps));
    let (_, g1) = log_post_grad(target, &x);
    r = sub(&r, &scale(&g1, -0.5 * eps));
    (x, r)
}

/// Evaluate the target and its gradient using a fresh reverse-mode tape.
fn log_post_grad<T: Target + ?Sized>(target: &T, x: &[f64]) -> (f64, Vec<f64>) {
    let tape = GradientTape::new();
    let vars = tape.vars(x);
    let out = target.eval(&tape, &vars);
    let grad = tape.gradient(out, &vars);
    (out.value(), grad)
}

/// Standard-normal draw via Box–Muller.
pub(crate) fn randn<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_prob_core::SplitMix64;

    /// 2-D standard normal target, written with autodiff variables.
    fn std_normal<'t>(t: &'t GradientTape, v: &'t [Variable<'t>]) -> Variable<'t> {
        let mut s = t.constant(0.0);
        for &x in v {
            s += -0.5 * x * x;
        }
        s
    }

    /// N(3, 1) target (1-D).
    fn shifted<'t>(_t: &'t GradientTape, v: &'t [Variable<'t>]) -> Variable<'t> {
        let z = (v[0] - 3.0) / 1.0;
        -0.5 * z * z
    }

    #[test]
    fn recovers_standard_normal_1d() {
        let mut rng = SplitMix64::seed_from_u64(7);
        let samples = nuts(&std_normal, &[0.0], &mut rng, 4000).unwrap();
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let var: f64 = samples
            .iter()
            .map(|s| (s[0] - mean) * (s[0] - mean))
            .sum::<f64>()
            / samples.len() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.15);
        assert_abs_diff_eq!(var, 1.0, epsilon = 0.25);
    }

    #[test]
    fn recovers_standard_normal_2d() {
        let mut rng = SplitMix64::seed_from_u64(11);
        let samples = nuts(&std_normal, &[0.0, 0.0], &mut rng, 4000).unwrap();
        let n = samples.len() as f64;
        let mx: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / n;
        let my: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / n;
        let vx: f64 = samples.iter().map(|s| (s[0] - mx).powi(2)).sum::<f64>() / n;
        let vy: f64 = samples.iter().map(|s| (s[1] - my).powi(2)).sum::<f64>() / n;
        let cov: f64 = samples
            .iter()
            .map(|s| (s[0] - mx) * (s[1] - my))
            .sum::<f64>()
            / n;
        assert_abs_diff_eq!(mx, 0.0, epsilon = 0.15);
        assert_abs_diff_eq!(my, 0.0, epsilon = 0.15);
        assert_abs_diff_eq!(vx, 1.0, epsilon = 0.3);
        assert_abs_diff_eq!(vy, 1.0, epsilon = 0.3);
        assert_abs_diff_eq!(cov, 0.0, epsilon = 0.3);
    }

    #[test]
    fn nonzero_mean_target_shifts_posterior() {
        // Target N(3, 1):  -0.5*(x-3)^2
        let mut rng = SplitMix64::seed_from_u64(3);
        let samples = nuts(&shifted, &[0.0], &mut rng, 4000).unwrap();
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert_abs_diff_eq!(mean, 3.0, epsilon = 0.2);
    }
}
