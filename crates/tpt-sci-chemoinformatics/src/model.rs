//! Lightweight regression models and validation: ridge regression on
//! standardised descriptors, Tanimoto k-nearest-neighbours on fingerprints,
//! regression metrics, and k-fold / leave-one-out cross-validation.

use crate::error::{ChemoinformaticsError, Result};
use crate::fingerprint::Fingerprint;
use tpt_math_prob_core::{Rng, SplitMix64};

fn model_err(msg: impl Into<String>) -> ChemoinformaticsError {
    ChemoinformaticsError::Model(msg.into())
}

/// Regression quality metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Coefficient of determination (can be negative for poor models).
    pub r2: f64,
    /// Root-mean-square error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Number of samples.
    pub n: usize,
}

impl Metrics {
    /// Compute metrics for predictions against observations.
    ///
    /// # Errors
    /// If the slices differ in length or are empty.
    pub fn compute(observed: &[f64], predicted: &[f64]) -> Result<Metrics> {
        if observed.len() != predicted.len() || observed.is_empty() {
            return Err(model_err(
                "observed/predicted must be non-empty and equal length",
            ));
        }
        let n = observed.len() as f64;
        let mean = observed.iter().sum::<f64>() / n;
        let ss_res: f64 = observed
            .iter()
            .zip(predicted)
            .map(|(o, p)| (o - p).powi(2))
            .sum();
        let ss_tot: f64 = observed.iter().map(|o| (o - mean).powi(2)).sum();
        let mae = observed
            .iter()
            .zip(predicted)
            .map(|(o, p)| (o - p).abs())
            .sum::<f64>()
            / n;
        Ok(Metrics {
            r2: if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                f64::NAN
            },
            rmse: (ss_res / n).sqrt(),
            mae,
            n: observed.len(),
        })
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` (row-major `n×n`) via
/// Cholesky factorisation.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-14 {
                    return Err(model_err("normal matrix is not positive definite"));
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    Ok(x)
}

/// Ridge regression on standardised features.
#[derive(Debug, Clone, PartialEq)]
pub struct Ridge {
    lambda: f64,
    mean: Vec<f64>,
    std: Vec<f64>,
    weights: Vec<f64>,
    y_mean: f64,
}

impl Ridge {
    /// Fit on rows `x` (one per sample) with labels `y` and L2 penalty `lambda > 0`.
    ///
    /// Features are z-scored (constant columns get zero weight) and the
    /// intercept is unpenalised.
    ///
    /// # Errors
    /// On empty/ragged/non-finite input or `lambda <= 0`.
    pub fn fit(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Result<Ridge> {
        if !(lambda > 0.0 && lambda.is_finite()) {
            return Err(ChemoinformaticsError::InvalidParameter(
                "ridge lambda must be positive and finite".into(),
            ));
        }
        if x.is_empty() || x.len() != y.len() {
            return Err(model_err("x and y must be non-empty and of equal length"));
        }
        let p = x[0].len();
        if p == 0 || x.iter().any(|r| r.len() != p) {
            return Err(model_err(
                "feature rows must be non-empty and of equal length",
            ));
        }
        if x.iter().flatten().chain(y).any(|v| !v.is_finite()) {
            return Err(model_err("non-finite value in training data"));
        }
        let n = x.len() as f64;
        let mean: Vec<f64> = (0..p)
            .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n)
            .collect();
        let std: Vec<f64> = (0..p)
            .map(|j| {
                let v = x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n;
                if v.sqrt() < 1e-12 { 1.0 } else { v.sqrt() }
            })
            .collect();
        let y_mean = y.iter().sum::<f64>() / n;
        let z: Vec<Vec<f64>> = x
            .iter()
            .map(|r| (0..p).map(|j| (r[j] - mean[j]) / std[j]).collect())
            .collect();
        let mut a = vec![0.0; p * p];
        let mut b = vec![0.0; p];
        for (row, yi) in z.iter().zip(y) {
            for i in 0..p {
                b[i] += row[i] * (yi - y_mean);
                for j in 0..p {
                    a[i * p + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..p {
            a[i * p + i] += lambda;
        }
        let weights = cholesky_solve(&a, &b, p)?;
        Ok(Ridge {
            lambda,
            mean,
            std,
            weights,
            y_mean,
        })
    }

    /// Predict one sample.
    ///
    /// # Errors
    /// If the feature length differs from training.
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.weights.len() {
            return Err(model_err(format!(
                "expected {} features, got {}",
                self.weights.len(),
                x.len()
            )));
        }
        Ok(self.y_mean
            + x.iter()
                .enumerate()
                .map(|(j, v)| self.weights[j] * (v - self.mean[j]) / self.std[j])
                .sum::<f64>())
    }

    /// Weights on the standardised features.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The L2 penalty used.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Similarity-weighted Tanimoto k-nearest-neighbour regressor.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnRegressor {
    k: usize,
    fingerprints: Vec<Fingerprint>,
    y: Vec<f64>,
}

impl KnnRegressor {
    /// Store the training set. `k` is clamped to the number of samples.
    ///
    /// # Errors
    /// On empty input, length mismatch or `k == 0`.
    pub fn fit(fingerprints: &[Fingerprint], y: &[f64], k: usize) -> Result<KnnRegressor> {
        if k == 0 {
            return Err(ChemoinformaticsError::InvalidParameter(
                "k must be >= 1".into(),
            ));
        }
        if fingerprints.is_empty() || fingerprints.len() != y.len() {
            return Err(model_err(
                "fingerprints and y must be non-empty and equal length",
            ));
        }
        Ok(KnnRegressor {
            k: k.min(y.len()),
            fingerprints: fingerprints.to_vec(),
            y: y.to_vec(),
        })
    }

    /// Predict a label; returns `(prediction, max similarity to the training set)`.
    pub fn predict(&self, fp: &Fingerprint) -> (f64, f64) {
        let mut sims: Vec<(f64, f64)> = self
            .fingerprints
            .iter()
            .zip(&self.y)
            .map(|(t, &y)| (fp.tanimoto(t), y))
            .collect();
        sims.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top = &sims[..self.k];
        let wsum: f64 = top.iter().map(|(s, _)| s).sum();
        let pred = if wsum > 0.0 {
            top.iter().map(|(s, y)| s * y).sum::<f64>() / wsum
        } else {
            top.iter().map(|(_, y)| y).sum::<f64>() / top.len() as f64
        };
        (pred, sims[0].0)
    }
}

/// Generic cross-validation. `folds` of 0 or `>= n` means leave-one-out.
/// `fit_predict(train_indices, test_indices)` returns predictions for the test
/// indices; the pooled out-of-fold predictions are scored against `y`.
///
/// Folds are assigned by a seeded Fisher–Yates shuffle, so results are
/// deterministic for a given `seed`.
///
/// # Errors
/// If fewer than two samples are given, or `fit_predict` fails.
pub fn cross_validate<F>(y: &[f64], folds: usize, seed: u64, mut fit_predict: F) -> Result<Metrics>
where
    F: FnMut(&[usize], &[usize]) -> Result<Vec<f64>>,
{
    let n = y.len();
    if n < 2 {
        return Err(model_err("cross-validation needs at least 2 samples"));
    }
    let k = if folds == 0 || folds >= n { n } else { folds };
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::seed_from_u64(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut observed = Vec::with_capacity(n);
    let mut predicted = Vec::with_capacity(n);
    for f in 0..k {
        let test: Vec<usize> = order.iter().copied().skip(f).step_by(k).collect();
        let train: Vec<usize> = order
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k != f)
            .map(|(_, &v)| v)
            .collect();
        let preds = fit_predict(&train, &test)?;
        if preds.len() != test.len() {
            return Err(model_err(
                "fit_predict returned the wrong number of predictions",
            ));
        }
        for (&t, p) in test.iter().zip(preds) {
            observed.push(y[t]);
            predicted.push(p);
        }
    }
    Metrics::compute(&observed, &predicted)
}
