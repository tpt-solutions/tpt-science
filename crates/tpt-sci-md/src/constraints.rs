//! Holonomic bond constraints: SHAKE position projection and RATTLE velocity projection.

use super::*;

/// A pairwise bond-length constraint for SHAKE/RATTLE.
#[derive(Debug, Clone, Copy)]
pub struct Bond {
    /// Index of the first particle.
    pub i: usize,
    /// Index of the second particle.
    pub j: usize,
    /// Target (fixed) bond length.
    pub r0: f64,
}

/// SHAKE (position) / RATTLE-style (velocity) holonomic bond-length
/// constraints for use with [`Integrator::velocity_verlet_constrained`].
///
/// Both solvers use the standard linearized SHAKE correction, applied
/// iteratively (Gauss-Seidel over all bonds) until every bond's squared-length
/// error is below `tol` (relative, for positions) or `max_iter` is exceeded.
#[derive(Debug, Clone)]
pub struct Shake {
    /// The constrained bonds.
    pub bonds: Vec<Bond>,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum Gauss-Seidel sweeps before giving up.
    pub max_iter: usize,
}

impl Shake {
    /// Construct a validated set of SHAKE/RATTLE constraints.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if `bonds` is empty, any bond has
    /// `i == j` or `r0 <= 0`, `tol <= 0`, or `max_iter == 0`.
    pub fn new(bonds: Vec<Bond>, tol: f64, max_iter: usize) -> Result<Self, MdError> {
        if bonds.is_empty() {
            return Err(MdError::ShakeError("at least one bond is required".into()));
        }
        for b in &bonds {
            if b.i == b.j {
                return Err(MdError::ShakeError(format!(
                    "bond {}-{} must connect two distinct particles",
                    b.i, b.j
                )));
            }
            if b.r0 <= 0.0 {
                return Err(MdError::ShakeError(format!(
                    "bond {}-{}: r0 must be > 0, got {}",
                    b.i, b.j, b.r0
                )));
            }
        }
        if tol <= 0.0 {
            return Err(MdError::ShakeError(format!("tol must be > 0, got {tol}")));
        }
        if max_iter == 0 {
            return Err(MdError::ShakeError("max_iter must be > 0".into()));
        }
        Ok(Self {
            bonds,
            tol,
            max_iter,
        })
    }

    /// Iteratively correct `positions` (the unconstrained new positions) so
    /// every bond returns to its target length, using `old_positions` (which
    /// must already satisfy the constraints) to fix the correction direction —
    /// the standard linearized SHAKE algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if a bond references an out-of-range
    /// particle index, or the iteration does not converge within `max_iter`.
    pub fn constrain_positions(
        &self,
        old_positions: &[DVector<f64>],
        positions: &mut [DVector<f64>],
        masses: &[f64],
        box_len: f64,
    ) -> Result<(), MdError> {
        let n = positions.len();
        for b in &self.bonds {
            if b.i >= n || b.j >= n {
                return Err(MdError::ShakeError(format!(
                    "bond references out-of-range particle {}/{} (n = {n})",
                    b.i, b.j
                )));
            }
        }
        let wrap = |x: f64| -> f64 {
            if !box_len.is_finite() {
                return x;
            }
            let half = box_len * 0.5;
            let mut y = x - box_len * (x / box_len).floor();
            if y > half {
                y -= box_len;
            }
            y
        };
        for _ in 0..self.max_iter {
            let mut max_err = 0.0_f64;
            for b in &self.bonds {
                let (i, j, d0) = (b.i, b.j, b.r0);
                let dim = positions[i].len();
                let raw_old = old_positions[i].clone() - old_positions[j].clone();
                let r_old = DVector::from_fn(dim, |k| wrap(raw_old[k]));
                let raw_new = positions[i].clone() - positions[j].clone();
                let s = DVector::from_fn(dim, |k| wrap(raw_new[k]));
                let r2 = s.dot(&s);
                let diff = r2 - d0 * d0;
                max_err = max_err.max(diff.abs() / (d0 * d0));
                let denom = 2.0 * (1.0 / masses[i] + 1.0 / masses[j]) * r_old.dot(&s);
                if denom.abs() < 1e-14 {
                    continue;
                }
                let g = diff / denom;
                positions[i] = positions[i].clone() - r_old.clone() * (g / masses[i]);
                positions[j] = positions[j].clone() + r_old * (g / masses[j]);
            }
            if max_err < self.tol {
                return Ok(());
            }
        }
        Err(MdError::ShakeError(format!(
            "SHAKE did not converge within {} iterations",
            self.max_iter
        )))
    }

    /// Iteratively project `velocities` so every constrained bond satisfies
    /// `d/dt |r_ij| = 0`, i.e. `r_ij · v_ij = 0` (the RATTLE velocity
    /// constraint), given the already position-constrained `positions`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::ShakeError`] if a bond references an out-of-range
    /// particle index, or the iteration does not converge within `max_iter`.
    pub fn constrain_velocities(
        &self,
        positions: &[DVector<f64>],
        velocities: &mut [DVector<f64>],
        masses: &[f64],
    ) -> Result<(), MdError> {
        let n = velocities.len();
        for b in &self.bonds {
            if b.i >= n || b.j >= n {
                return Err(MdError::ShakeError(format!(
                    "bond references out-of-range particle {}/{} (n = {n})",
                    b.i, b.j
                )));
            }
        }
        for _ in 0..self.max_iter {
            let mut max_err = 0.0_f64;
            for b in &self.bonds {
                let (i, j, d0) = (b.i, b.j, b.r0);
                let r_ij = positions[i].clone() - positions[j].clone();
                let v_ij = velocities[i].clone() - velocities[j].clone();
                let num = r_ij.dot(&v_ij);
                max_err = max_err.max(num.abs());
                let denom = (1.0 / masses[i] + 1.0 / masses[j]) * d0 * d0;
                if denom.abs() < 1e-14 {
                    continue;
                }
                let k = num / denom;
                velocities[i] = velocities[i].clone() - r_ij.clone() * (k / masses[i]);
                velocities[j] = velocities[j].clone() + r_ij * (k / masses[j]);
            }
            if max_err < self.tol {
                return Ok(());
            }
        }
        Err(MdError::ShakeError(format!(
            "RATTLE velocity projection did not converge within {} iterations",
            self.max_iter
        )))
    }
}
