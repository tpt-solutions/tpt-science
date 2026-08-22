//! Local (norm-conserving-style) analytic pseudopotentials for 3-D real-space
//! Kohn–Sham.
//!
//! A full norm-conserving Troullier–Martins generation requires an all-electron
//! reference calculation, which is out of scope. Instead this crate provides a
//! self-contained, **softened local** pseudopotential `V_ps(r)` that is finite
//! at the origin (so the 3-D finite-difference solver never blows up) and decays
//! to the correct `-Z/r` Coulomb tail asymptotically. This is sufficient to make
//! multi-electron 3-D atoms tractable as a local-potential problem.

use tpt_sci_grid::UniformGrid3D;

/// A softened local pseudopotential `V_ps(r)`.
///
/// The potential is `V_ps(r) = -Z · erf(r/σ) / r`, where `erf` is the error
/// function. As `r → 0`, `erf(r/σ) ≈ 2r/(σ√π)` so `V_ps → -2Z/(σ√π)` (finite —
/// the Coulomb singularity is regularised). As `r → ∞`, `erf → 1` so
/// `V_ps → -Z/r` (the correct bare Coulomb tail of a nucleus of charge `Z`).
#[derive(Debug, Clone)]
pub struct Pseudopotential {
    /// Nuclear charge `Z`.
    z: f64,
    /// Softening length `σ` (regularisation radius).
    sigma: f64,
    /// Value at the origin `-2Z/(σ√π)`.
    origin_value: f64,
}

impl Pseudopotential {
    /// Construct a softened Coulomb pseudopotential `-Z·erf(r/σ)/r`.
    ///
    /// # Panics
    ///
    /// Panics if `z <= 0` or `sigma <= 0`.
    #[must_use]
    pub fn new(z: f64, sigma: f64) -> Self {
        assert!(z > 0.0, "nuclear charge must be positive");
        assert!(sigma > 0.0, "softening length must be positive");
        let origin_value = -2.0 * z / (sigma * std::f64::consts::PI.sqrt());
        Self {
            z,
            sigma,
            origin_value,
        }
    }

    /// Nuclear charge `Z`.
    #[must_use]
    pub fn charge(&self) -> f64 {
        self.z
    }

    /// Softening length `σ`.
    #[must_use]
    pub fn softening(&self) -> f64 {
        self.sigma
    }

    /// Value of the potential at the origin (the regularised, finite limit).
    #[must_use]
    pub fn origin_value(&self) -> f64 {
        self.origin_value
    }

    /// Evaluate the pseudopotential at distance `r >= 0`.
    ///
    /// At `r = 0` this returns the finite origin limit rather than dividing by
    /// zero.
    #[must_use]
    pub fn value(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return self.origin_value;
        }
        -self.z * Self::erf(r / self.sigma) / r
    }

    /// Error function via the Abramowitz & Stegun 7.1.26 rational approximation
    /// (absolute accuracy better than 1.5e-7 across the real line).
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let poly = (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592;
        let y = 1.0 - poly * t * (-ax * ax).exp();
        sign * y
    }

    /// Sample the pseudopotential on a 3-D grid, returning a vector of length
    /// `grid.len()` addressed by `grid.index(ix, iy, iz)`.
    ///
    /// The potential is measured relative to `center`; for a grid that does not
    /// contain the center node exactly the nearest node to the center is used.
    ///
    /// # Panics
    ///
    /// Panics if any coordinate lookup on `grid` fails (it will not, for a valid
    /// [`UniformGrid3D`]).
    #[must_use]
    pub fn on_grid(&self, grid: &UniformGrid3D, center: [f64; 3]) -> Vec<f64> {
        let xs = grid.x_coordinates();
        let ys = grid.y_coordinates();
        let zs = grid.z_coordinates();
        (0..grid.len())
            .map(|k| {
                let ix = k % grid.nx();
                let iy = (k / grid.nx()) % grid.ny();
                let iz = k / (grid.nx() * grid.ny());
                let dx = xs[ix] - center[0];
                let dy = ys[iy] - center[1];
                let dz = zs[iz] - center[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                self.value(r)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use tpt_sci_grid::UniformGrid3D;

    #[test]
    fn finite_and_regular_at_origin() {
        let ps = Pseudopotential::new(4.0, 0.7);
        let at_origin = ps.value(0.0);
        assert!(at_origin.is_finite());
        // Analytic regularised limit: -2Z/(σ√π).
        let expected = -2.0 * 4.0 / (0.7 * std::f64::consts::PI.sqrt());
        assert_abs_diff_eq!(at_origin, expected, epsilon = 1e-12);
        assert_abs_diff_eq!(at_origin, ps.origin_value(), epsilon = 1e-12);
    }

    #[test]
    fn decays_to_coulomb_tail() {
        let z = 3.0;
        let sigma = 0.5;
        let ps = Pseudopotential::new(z, sigma);
        // Far from the origin erf(r/σ) ≈ 1, so V → -Z/r.
        for &r in &[5.0_f64, 10.0, 20.0] {
            assert_abs_diff_eq!(ps.value(r), -z / r, epsilon = 1e-3);
        }
    }

    #[test]
    fn on_grid_is_finite_and_centered() {
        let ps = Pseudopotential::new(2.0, 0.8);
        let grid = UniformGrid3D::new(11, -5.0, 5.0, 9, -4.0, 4.0, 7, -3.0, 3.0).unwrap();
        let v = ps.on_grid(&grid, [0.0, 0.0, 0.0]);
        assert_eq!(v.len(), grid.len());
        assert!(v.iter().all(|&x| x.is_finite()));
        // Deepest (most negative) value sits at the center.
        let center = grid.index(grid.nx() / 2, grid.ny() / 2, grid.nz() / 2);
        let min_val = v.iter().cloned().fold(f64::INFINITY, f64::min);
        assert_abs_diff_eq!(v[center], min_val, epsilon = 1e-12);
    }
}
