//! Data assimilation for ocean state estimation.
//!
//! Two schemes are provided:
//!
//! * [`nudge`] — a simple, robust **nudging** (relaxation) toward observations,
//!   ideal for continuous / weak-constraint assimilation of sparse data.
//! * [`EnsembleKalmanFilter`] — a stochastic **ensemble Kalman filter** (EnKF)
//!   for nonlinear/linear observation operators, and [`Var3D`] — a
//!   **3D-Var-lite** analysis (optimal linear update with a background-error
//!   covariance), both validated against a toy linear observation operator.
//!
//! All operators are implemented from scratch (small dense linear algebra) so
//! the crate keeps no extra external dependency.

use crate::OceanError;

/// A single sparse observation: a measured `value` at flat `index`, with an
/// optional `weight` (default 1) scaling how strongly it nudges the state.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Flat index into the state / field vector.
    pub index: usize,
    /// Observed value.
    pub value: f64,
    /// Relaxation weight (multiplies `coeff·dt`); 1.0 by default.
    pub weight: f64,
}

/// Relax `state` toward the observations: for each [`Observation`],
/// `state[index] ⟵ state[index] + coeff·dt·weight·(value − state[index])`.
///
/// Unobserved entries are left unchanged. With `coeff·dt·weight = 1` the observed
/// entries are replaced by their values in a single call; smaller coefficients
/// give a gentle, continuous relaxation (the standard nudging limit).
#[must_use]
pub fn nudge(state: &[f64], obs: &[Observation], coeff: f64, dt: f64) -> Vec<f64> {
    let mut out = state.to_vec();
    for o in obs {
        if o.index < out.len() {
            let w = if o.weight.is_finite() { o.weight } else { 1.0 };
            out[o.index] += coeff * dt * w * (o.value - out[o.index]);
        }
    }
    out
}

/// A small deterministic xorshift32 generator (seeded) used only to make the
/// EnKF observation perturbations reproducible in tests and examples.
struct Rng(u32);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f64) / ((u32::MAX as f64) + 1.0)
    }

    /// Standard-normal sample via the Box–Muller transform.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// `a · b` for `a` (`m×k`) and `b` (`k×n`) returning an `m×n` matrix.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = b.len();
    let n = b[0].len();
    let mut out = vec![vec![0.0; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            if aip == 0.0 {
                continue;
            }
            let brow = &b[p];
            let orow = &mut out[i];
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
    out
}

/// Matrix–vector product `a · x` for `a` (`m×n`) and `x` (length `n`).
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// Transpose of `a`.
fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Lower Cholesky factor `L` of a symmetric positive-definite `a` (`L·Lᵀ = a`).
///
/// # Errors
///
/// Returns [`OceanError::LinAlg`] if `a` is not positive definite.
fn cholesky_lower(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OceanError> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (lik, ljk) in l[i].iter().take(j).zip(&l[j][..j]) {
                sum -= lik * ljk;
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(OceanError::LinAlg("matrix not positive definite".into()));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Inverse of a lower-triangular matrix `l`.
fn lower_inverse(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = l.len();
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        inv[j][j] = 1.0 / l[j][j];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[i][k] * inv[k][j];
            }
            inv[i][j] = -s / l[i][i];
        }
    }
    inv
}

/// Inverse of a symmetric positive-definite matrix via Cholesky factorisation.
///
/// # Errors
///
/// Returns [`OceanError::LinAlg`] if the input is not positive definite.
fn cholesky_inverse(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OceanError> {
    let l = cholesky_lower(a)?;
    let linv = lower_inverse(&l);
    let lt = mat_transpose(&linv);
    Ok(mat_mul(&linv, &lt))
}

/// A stochastic ensemble Kalman filter (EnKF).
///
/// Holds a background ensemble of `m` state vectors (each length `n`) and a
/// diagonal observation-error covariance `R`. [`EnsembleKalmanFilter::analyze`]
/// ingests observations `y` and a linear observation operator `H` (`p×n`) and
/// returns the analysis ensemble (and its mean) via the perturbed-observation
/// formulation.
#[derive(Debug, Clone)]
pub struct EnsembleKalmanFilter {
    /// State dimension `n`.
    pub n: usize,
    /// Ensemble size `m`.
    pub m: usize,
    /// Background ensemble: `m` members, each a length-`n` state vector.
    pub ensemble: Vec<Vec<f64>>,
    /// Diagonal of the observation-error covariance `R` (length `p`).
    pub r_diag: Vec<f64>,
}

impl EnsembleKalmanFilter {
    /// Build a filter from a background ensemble and diagonal `R`.
    ///
    /// # Errors
    ///
    /// Returns [`OceanError::DimensionMismatch`] if the ensemble is empty, has
    /// fewer than two members, or contains members of inconsistent length, or if
    /// `r_diag` is empty.
    pub fn new(ensemble: Vec<Vec<f64>>, r_diag: Vec<f64>) -> Result<Self, OceanError> {
        if ensemble.len() < 2 {
            return Err(OceanError::DimensionMismatch(
                "ensemble must have >= 2 members".into(),
            ));
        }
        let n = ensemble[0].len();
        if n == 0 {
            return Err(OceanError::DimensionMismatch(
                "state dimension must be > 0".into(),
            ));
        }
        for member in &ensemble {
            if member.len() != n {
                return Err(OceanError::DimensionMismatch(
                    "ensemble members differ in length".into(),
                ));
            }
        }
        if r_diag.is_empty() {
            return Err(OceanError::DimensionMismatch(
                "observation covariance is empty".into(),
            ));
        }
        Ok(Self {
            n,
            m: ensemble.len(),
            ensemble,
            r_diag,
        })
    }

    /// Analysis step for a linear observation operator `h` (`p×n`) and
    /// observation vector `y` (length `p`).
    ///
    /// # Errors
    ///
    /// Returns [`OceanError::DimensionMismatch`] if the operator dimensions do
    /// not match the state / observation sizes, or [`OceanError::LinAlg`] if the
    /// ensemble covariance matrix is not positive definite.
    pub fn analyze(&self, y: &[f64], h: &[Vec<f64>]) -> Result<EnkfResult, OceanError> {
        let m = self.m;
        let n = self.n;
        let p = y.len();
        if h.len() != p || h[0].len() != n || self.r_diag.len() != p {
            return Err(OceanError::DimensionMismatch(
                "observation operator / vector dimensions inconsistent".into(),
            ));
        }
        if p == 0 {
            return Err(OceanError::DimensionMismatch("zero observations".into()));
        }

        // Background mean and perturbation matrix A (n×m).
        let mut xb_mean = vec![0.0; n];
        for member in &self.ensemble {
            for (i, &v) in member.iter().enumerate() {
                xb_mean[i] += v;
            }
        }
        for v in &mut xb_mean {
            *v /= m as f64;
        }
        let mut a = vec![vec![0.0; m]; n];
        for (k, member) in self.ensemble.iter().enumerate() {
            for i in 0..n {
                a[i][k] = member[i] - xb_mean[i];
            }
        }

        // HA = H·A (p×m), C = (1/(m-1))·A·HAᵀ (n×p).
        let ha = mat_mul(h, &a);
        let ha_t = mat_transpose(&ha);
        let mut c = mat_mul(&a, &ha_t);
        let inv = 1.0 / (m as f64 - 1.0);
        for row in &mut c {
            for v in row {
                *v *= inv;
            }
        }

        // M = (1/(m-1))·HA·HAᵀ + R (p×p), then invert.
        let mut mmat = mat_mul(&ha, &ha_t);
        for row in &mut mmat {
            for v in row {
                *v *= inv;
            }
        }
        for (i, row) in mmat.iter_mut().enumerate().take(p) {
            row[i] += self.r_diag[i];
        }
        let minv = cholesky_inverse(&mmat)?;

        // Kalman gain K = C·M⁻¹ (n×p), then update each perturbed member.
        let k = mat_mul(&c, &minv);
        let mut rng = Rng(0x1234_5678);
        let mut analysis = Vec::with_capacity(m);
        for member in &self.ensemble {
            let hxb = mat_vec(h, member);
            let yp: Vec<f64> = (0..p)
                .map(|i| y[i] + self.r_diag[i].sqrt() * rng.normal())
                .collect();
            let d: Vec<f64> = (0..p).map(|i| yp[i] - hxb[i]).collect();
            let update = mat_vec(&k, &d);
            let xa: Vec<f64> = member.iter().zip(&update).map(|(x, u)| x + u).collect();
            analysis.push(xa);
        }

        let mut mean = vec![0.0; n];
        for member in &analysis {
            for (i, &v) in member.iter().enumerate() {
                mean[i] += v;
            }
        }
        for v in &mut mean {
            *v /= m as f64;
        }
        Ok(EnkfResult {
            mean,
            ensemble: analysis,
        })
    }
}

/// Result of an [`EnsembleKalmanFilter::analyze`] step.
#[derive(Debug, Clone)]
pub struct EnkfResult {
    /// Analysis ensemble mean (length `n`).
    pub mean: Vec<f64>,
    /// Analysis ensemble (`m` members, each length `n`).
    pub ensemble: Vec<Vec<f64>>,
}

/// A 3D-Var-lite analysis with a diagonal background-error covariance.
///
/// Minimises `J(x) = ½(x−xb)ᵀB⁻¹(x−xb) + ½(y−Hx)ᵀR⁻¹(y−Hx)`. For a linear `H`
/// and diagonal `B`, `R` this has the closed-form optimum
/// `xa = xb + B·Hᵀ(H·B·Hᵀ + R)⁻¹(y − H·xb)`.
#[derive(Debug, Clone)]
pub struct Var3D {
    /// Background state `xb` (length `n`).
    pub xb: Vec<f64>,
    /// Diagonal background-error covariance `B` (length `n`).
    pub b_diag: Vec<f64>,
    /// Diagonal observation-error covariance `R` (length `p`).
    pub r_diag: Vec<f64>,
}

impl Var3D {
    /// Build a 3D-Var analysis with diagonal `B` and `R`.
    ///
    /// # Errors
    ///
    /// Returns [`OceanError::DimensionMismatch`] if `xb` is empty or the
    /// covariance vectors are empty.
    pub fn new(xb: Vec<f64>, b_diag: Vec<f64>, r_diag: Vec<f64>) -> Result<Self, OceanError> {
        if xb.is_empty() || b_diag.len() != xb.len() || r_diag.is_empty() {
            return Err(OceanError::DimensionMismatch(
                "invalid 3D-Var dimensions".into(),
            ));
        }
        Ok(Self { xb, b_diag, r_diag })
    }

    /// Compute the analysis `xa` for observations `y` and linear operator `h`.
    ///
    /// # Errors
    ///
    /// Returns [`OceanError::DimensionMismatch`] if `h`'s dimensions do not
    /// match, or [`OceanError::LinAlg`] if `H·B·Hᵀ + R` is not positive definite.
    pub fn analyze(&self, y: &[f64], h: &[Vec<f64>]) -> Result<Vec<f64>, OceanError> {
        let n = self.xb.len();
        let p = y.len();
        if h.len() != p || h[0].len() != n || self.r_diag.len() != p {
            return Err(OceanError::DimensionMismatch(
                "observation operator / vector dimensions inconsistent".into(),
            ));
        }
        if p == 0 {
            return Err(OceanError::DimensionMismatch("zero observations".into()));
        }
        // C = B·Hᵀ (n×p): B is diagonal, so C[i][j] = b_diag[i]·h[j][i].
        let mut c = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                c[i][j] = self.b_diag[i] * h[j][i];
            }
        }
        // M = H·B·Hᵀ + R (p×p).
        let mut mmat = mat_mul(h, &c);
        for (i, row) in mmat.iter_mut().enumerate().take(p) {
            row[i] += self.r_diag[i];
        }
        let minv = cholesky_inverse(&mmat)?;
        // Gain K = C·M⁻¹ (n×p), innovation d = y − H·xb.
        let k = mat_mul(&c, &minv);
        let hxb = mat_vec(h, &self.xb);
        let d: Vec<f64> = (0..p).map(|i| y[i] - hxb[i]).collect();
        let update = mat_vec(&k, &d);
        let xa: Vec<f64> = self.xb.iter().zip(&update).map(|(x, u)| x + u).collect();
        Ok(xa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn nudge_relaxes_toward_observations() {
        // Start one unit off the truth at index 0 and observe truth there.
        let truth = [1.0, 2.0, 3.0, 4.0];
        let state = [2.0, 2.0, 3.0, 4.0];
        let obs = [Observation {
            index: 0,
            value: truth[0],
            weight: 1.0,
        }];
        let err0 = (state[0] - truth[0]).abs();
        let out = nudge(&state, &obs, 0.3, 1.0);
        let err1 = (out[0] - truth[0]).abs();
        assert!(
            err1 < err0,
            "nudging should move the state toward the observation"
        );
        // Unobserved entries are untouched.
        assert_abs_diff_eq!(out[1], state[1], epsilon = 1e-12);
        // Repeated nudging converges to the observed value.
        let mut s = state.to_vec();
        for _ in 0..200 {
            s = nudge(&s, &obs, 0.3, 1.0);
        }
        assert_abs_diff_eq!(s[0], truth[0], epsilon = 1e-6);
    }

    #[test]
    fn enkf_reduces_analysis_error() {
        let n = 6;
        let p = 6;
        let truth: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        // Linear identity observation operator.
        let h: Vec<Vec<f64>> = (0..p)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        // Background ensemble: mean biased from truth, small spread.
        let bias = 0.5;
        let mut rng = Rng(0xDEAD_BEEF);
        let m = 40;
        let mut ensemble = Vec::with_capacity(m);
        for _ in 0..m {
            let member: Vec<f64> = (0..n)
                .map(|i| truth[i] + bias + 0.3 * rng.normal())
                .collect();
            ensemble.push(member);
        }
        // Background mean error.
        let bg_mean: Vec<f64> = (0..n)
            .map(|i| ensemble.iter().map(|e| e[i]).sum::<f64>() / m as f64)
            .collect();
        let bg_err = bg_mean
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        // Observations = truth (tiny noise via R).
        let y = truth.clone();
        let r_diag = vec![1e-6; p];
        let enkf = EnsembleKalmanFilter::new(ensemble, r_diag).unwrap();
        let result = enkf.analyze(&y, &h).unwrap();
        let an_err = result
            .mean
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            an_err < bg_err,
            "EnKF analysis mean should beat the background (bg={bg_err}, an={an_err})"
        );
    }

    #[test]
    fn var3d_reduces_analysis_error() {
        let n = 6;
        let p = 6;
        let truth: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let h: Vec<Vec<f64>> = (0..p)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let xb: Vec<f64> = truth.iter().map(|v| v + 0.5).collect();
        let bg_err = xb
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let var = Var3D::new(xb, vec![0.2; n], vec![1e-6; p]).unwrap();
        let xa = var.analyze(&truth, &h).unwrap();
        let an_err = xa
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            an_err < bg_err,
            "3D-Var analysis should beat the background (bg={bg_err}, an={an_err})"
        );
    }
}
