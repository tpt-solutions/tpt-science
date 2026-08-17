//! Convergence diagnostics and the return type of [`crate::Model::fit`].
//!
//! [`Trace`] collects posterior draws (optionally from several chains) together
//! with the diagnostics a real Bayesian workflow needs: an effective sample
//! size (ESS), a split-R-hat, and the divergence rate. These were previously
//! silently discarded by the sampler.

/// Posterior draws from one or more NUTS chains, with convergence diagnostics.
///
/// Layout is `chains[chain][draw][parameter]`. A single-chain [`crate::Model::fit`]
/// produces one chain; [`crate::Model::fit_chains`] produces several so that
/// [`Trace::rhat`] is well defined.
pub struct Trace {
    /// One vector of draws per chain.
    chains: Vec<Vec<Vec<f64>>>,
    /// Total divergent transitions recorded during the sampling phase across
    /// all chains (a sampler health signal — a high rate points at a poorly
    /// identified or steep posterior).
    n_divergences: usize,
}

impl Trace {
    /// Build a trace from per-chain draws and a divergence count.
    pub(crate) fn new(chains: Vec<Vec<Vec<f64>>>, n_divergences: usize) -> Self {
        Self {
            chains,
            n_divergences,
        }
    }

    /// Number of chains.
    #[must_use]
    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }

    /// Total number of draws across all chains.
    #[must_use]
    pub fn n_draws(&self) -> usize {
        self.chains.iter().map(Vec::len).sum()
    }

    /// Number of parameters (length of a draw vector).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.chains
            .first()
            .and_then(|c| c.first())
            .map_or(0, Vec::len)
    }

    /// The raw per-chain samples.
    #[must_use]
    pub fn chains(&self) -> &[Vec<Vec<f64>>] {
        &self.chains
    }

    /// All draws flattened across chains (chain order, then draw order).
    #[must_use]
    pub fn combined_samples(&self) -> Vec<Vec<f64>> {
        self.chains.iter().flatten().cloned().collect()
    }

    /// Total divergent transitions recorded while sampling.
    #[must_use]
    pub fn n_divergences(&self) -> usize {
        self.n_divergences
    }

    /// Fraction of sampling iterations that ended in a divergence.
    #[must_use]
    pub fn divergence_rate(&self) -> f64 {
        let total = self.n_draws();
        if total == 0 {
            0.0
        } else {
            self.n_divergences as f64 / total as f64
        }
    }

    /// Posterior mean of `param` over all draws.
    #[must_use]
    pub fn mean(&self, param: usize) -> f64 {
        let (s, n) = self
            .chains
            .iter()
            .flatten()
            .fold((0.0_f64, 0usize), |(s, n), draw| (s + draw[param], n + 1));
        if n == 0 { 0.0 } else { s / n as f64 }
    }

    /// Posterior standard deviation of `param` over all draws.
    #[must_use]
    pub fn std(&self, param: usize) -> f64 {
        let mean = self.mean(param);
        let (acc, n) = self
            .chains
            .iter()
            .flatten()
            .fold((0.0_f64, 0usize), |(acc, n), draw| {
                let d = draw[param] - mean;
                (acc + d * d, n + 1)
            });
        if n <= 1 {
            0.0
        } else {
            (acc / (n - 1) as f64).sqrt()
        }
    }

    /// Effective sample size of `param`: the sum of the per-chain ESS estimates
    /// (Geyer initial-monotone-sequence estimator) across all chains. Smaller
    /// than the raw draw count when draws are autocorrelated.
    #[must_use]
    pub fn ess(&self, param: usize) -> f64 {
        self.chains
            .iter()
            .map(|chain| geyer_ess(&param_column(chain, param)))
            .sum()
    }

    /// Split-R-hat of `param` (each chain split in half; Vehtari et al. 2021).
    /// than two chains (R-hat is undefined for a single chain). Values near `1`
    /// indicate convergence; values above `~1.01` are a cause for concern.
    #[must_use]
    pub fn rhat(&self, param: usize) -> f64 {
        if self.chains.len() < 2 {
            return f64::NAN;
        }
        split_rhat(&self.chains, param)
    }
}

// --- internal helpers -------------------------------------------------------

/// Extract one parameter's column from a single chain.
fn param_column(chain: &[Vec<f64>], param: usize) -> Vec<f64> {
    chain.iter().map(|draw| draw[param]).collect()
}

/// Geyer (1992) initial-monotone-sequence ESS for a single chain.
fn geyer_ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return n as f64;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var == 0.0 {
        return n as f64;
    }
    // P_t = rho_{2t} + rho_{2t+1}; truncate at the first non-positive P_t.
    let mut p = vec![1.0_f64];
    let mut t = 1usize;
    loop {
        let r2 = autocovariance(x, mean, var, 2 * t);
        let r1 = autocovariance(x, mean, var, 2 * t + 1);
        let pt = r2 + r1;
        if pt <= 0.0 || 2 * t + 1 >= n {
            break;
        }
        p.push(pt);
        t += 1;
    }
    // Monotone sequence: enforce a non-increasing P from the end backwards.
    for i in (1..p.len()).rev() {
        if p[i] > p[i - 1] {
            p[i] = p[i - 1];
        }
    }
    let tau = (-1.0 + 2.0 * p.iter().sum::<f64>()).max(1.0);
    n as f64 / tau
}

/// Lag-`lag` autocorrelation of `x` (normalised by its variance).
fn autocovariance(x: &[f64], mean: f64, var: f64, lag: usize) -> f64 {
    let n = x.len();
    if lag >= n {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..(n - lag) {
        s += (x[i] - mean) * (x[i + lag] - mean);
    }
    s / (n as f64 * var)
}

/// Split each chain in half and return the between/within R-hat on the raw
/// values (Vehtari et al. 2021, split-R-hat). `chains` must contain at least
/// two chains.
fn split_rhat(chains: &[Vec<Vec<f64>>], param: usize) -> f64 {
    // Split every chain into two halves -> 2·n_chains shorter chains.
    let mut split: Vec<Vec<f64>> = Vec::new();
    for c in chains {
        let n = c.len();
        let h = n / 2;
        if h == 0 {
            continue;
        }
        split.push(c[..h].iter().map(|d| d[param]).collect());
        split.push(c[h..].iter().map(|d| d[param]).collect());
    }
    let m = split.len() as f64;
    let n = split[0].len() as f64;
    let chain_means: Vec<f64> = split.iter().map(|s| s.iter().sum::<f64>() / n).collect();
    let grand = chain_means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|cm| (cm - grand).powi(2))
            .sum::<f64>();
    let mut w = 0.0;
    for (s, cm) in split.iter().zip(&chain_means) {
        let v = s.iter().map(|v| (v - cm).powi(2)).sum::<f64>() / (n - 1.0);
        w += v;
    }
    w /= m;
    if w == 0.0 {
        return 1.0;
    }
    let var_hat = (n - 1.0) / n * w + b / n;
    (var_hat / w).sqrt()
}
