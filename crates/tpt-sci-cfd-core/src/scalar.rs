//! Scalar transport (species or temperature) on the collocated grid.
//!
//! Advances a cell-centred scalar `c` by
//!
//! ```text
//! ∂c/∂t + u·∇c = D·∇²c + S(c)
//! ```
//!
//! with first-order upwind advection, a five-point diffusion Laplacian, and an
//! explicit source `S(c)` (a reaction rate, for example). Walls are zero-flux:
//! normal gradients vanish there, so the total amount of `c` is conserved in the
//! absence of sources.
//!
//! The velocity is taken from a flow solver ([`crate::Step`]'s `u` and `v`
//! fields), which is the coupling used by the reacting-cavity example.
//!
//! The scheme is explicit. [`ScalarTransport::max_stable_dt`] gives the largest
//! step that keeps it stable, and [`ScalarTransport::step`] refuses anything
//! larger.

use crate::{CfdError, CollocatedGrid};

/// Explicit scalar transport on a [`CollocatedGrid`].
#[derive(Debug, Clone)]
pub struct ScalarTransport {
    grid: CollocatedGrid,
    diffusivity: f64,
}

impl ScalarTransport {
    /// Build a transport solver with diffusivity `D ≥ 0`.
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::InvalidParameter`] if `D` is negative or not finite.
    pub fn new(grid: CollocatedGrid, diffusivity: f64) -> Result<Self, CfdError> {
        if !diffusivity.is_finite() || diffusivity < 0.0 {
            return Err(CfdError::InvalidParameter(format!(
                "diffusivity must be finite and >= 0, got {diffusivity}"
            )));
        }
        Ok(Self { grid, diffusivity })
    }

    /// The largest explicit time step that keeps the scheme stable for the given
    /// velocity field: the CFL limit for advection plus the diffusive limit.
    /// Returns `f64::INFINITY` for a static, non-diffusive field.
    #[must_use]
    pub fn max_stable_dt(&self, u: &[f64], v: &[f64]) -> f64 {
        let umax = u.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let vmax = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let advective = if umax / dx + vmax / dy > 0.0 {
            1.0 / (umax / dx + vmax / dy)
        } else {
            f64::INFINITY
        };
        let diffusive = if self.diffusivity > 0.0 {
            0.5 / (self.diffusivity * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
        } else {
            f64::INFINITY
        };
        advective.min(diffusive)
    }

    /// Advance `c` by one explicit step of length `dt`.
    ///
    /// `source(i, c_i)` gives `S(c)` for cell `i` (the flat index
    /// [`CollocatedGrid::idx`]). Pass `&|_, _| 0.0` for pure transport.
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::DimensionMismatch`] if any field has the wrong length,
    /// or [`CfdError::InvalidParameter`] if `dt` is not positive or exceeds
    /// [`ScalarTransport::max_stable_dt`].
    pub fn step(
        &self,
        c: &mut [f64],
        u: &[f64],
        v: &[f64],
        dt: f64,
        source: &dyn Fn(usize, f64) -> f64,
    ) -> Result<(), CfdError> {
        let n = self.grid.nx * self.grid.ny;
        for field in [c.len(), u.len(), v.len()] {
            if field != n {
                return Err(CfdError::DimensionMismatch {
                    expected: n,
                    actual: field,
                });
            }
        }
        if dt.is_nan() || dt <= 0.0 {
            return Err(CfdError::InvalidParameter(format!(
                "dt must be > 0, got {dt}"
            )));
        }
        let limit = self.max_stable_dt(u, v);
        if dt > limit {
            return Err(CfdError::InvalidParameter(format!(
                "dt = {dt} exceeds the stable limit {limit}"
            )));
        }

        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let d = self.diffusivity;
        let old = c.to_vec();
        for j in 0..ny {
            for i in 0..nx {
                let k = self.grid.idx(i, j);
                let here = old[k];
                // Zero-flux walls: a missing neighbour is replaced by the cell itself.
                let west = if i > 0 {
                    old[self.grid.idx(i - 1, j)]
                } else {
                    here
                };
                let east = if i + 1 < nx {
                    old[self.grid.idx(i + 1, j)]
                } else {
                    here
                };
                let south = if j > 0 {
                    old[self.grid.idx(i, j - 1)]
                } else {
                    here
                };
                let north = if j + 1 < ny {
                    old[self.grid.idx(i, j + 1)]
                } else {
                    here
                };

                // Upwind advection: take the difference on the side the flow comes from.
                let dcdx = (if u[k] >= 0.0 {
                    here - west
                } else {
                    east - here
                }) / dx;
                let dcdy = (if v[k] >= 0.0 {
                    here - south
                } else {
                    north - here
                }) / dy;
                let advection = u[k] * dcdx + v[k] * dcdy;
                let diffusion = d
                    * ((east - 2.0 * here + west) / (dx * dx)
                        + (north - 2.0 * here + south) / (dy * dy));

                c[k] = here + dt * (-advection + diffusion + source(k, here));
            }
        }
        Ok(())
    }

    /// Total amount of the scalar, `Σ c_i·Δx·Δy`.
    #[must_use]
    pub fn total(&self, c: &[f64]) -> f64 {
        c.iter().sum::<f64>() * self.grid.dx * self.grid.dy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> CollocatedGrid {
        CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap()
    }

    #[test]
    fn pure_diffusion_conserves_total() {
        let t = ScalarTransport::new(grid(), 0.01).unwrap();
        let n = 16 * 16;
        let zero = vec![0.0; n];
        let mut c: Vec<f64> = (0..n)
            .map(|k| if k % 17 == 0 { 1.0 } else { 0.0 })
            .collect();
        let before = t.total(&c);
        let dt = 0.5 * t.max_stable_dt(&zero, &zero);
        for _ in 0..200 {
            t.step(&mut c, &zero, &zero, dt, &|_, _| 0.0).unwrap();
        }
        assert!((t.total(&c) - before).abs() < 1e-12 * before.max(1.0));
    }

    #[test]
    fn uniform_flow_transports_a_blob_downstream() {
        let g = grid();
        let t = ScalarTransport::new(g.clone(), 0.0).unwrap();
        let n = 16 * 16;
        let u = vec![0.5; n];
        let v = vec![0.0; n];
        let mut c = vec![0.0; n];
        c[g.idx(3, 8)] = 1.0;
        let dt = 0.5 * t.max_stable_dt(&u, &v);
        let steps = 10;
        for _ in 0..steps {
            t.step(&mut c, &u, &v, dt, &|_, _| 0.0).unwrap();
        }
        let centroid: f64 = (0..16)
            .map(|i| i as f64 * (0..16).map(|j| c[g.idx(i, j)]).sum::<f64>())
            .sum::<f64>()
            / c.iter().sum::<f64>();
        assert!(
            centroid > 3.0,
            "the blob should move downstream, centroid {centroid}"
        );
    }

    #[test]
    fn source_term_consumes_the_scalar() {
        let t = ScalarTransport::new(grid(), 0.0).unwrap();
        let n = 16 * 16;
        let zero = vec![0.0; n];
        let mut c = vec![1.0; n];
        let k = 0.5;
        let dt = 0.01;
        for _ in 0..100 {
            t.step(&mut c, &zero, &zero, dt, &|_, ci| -k * ci).unwrap();
        }
        // Explicit Euler on dc/dt = −k c after t = 1.
        let expected = (1.0 - k * dt).powi(100);
        assert!((c[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn unstable_steps_are_refused() {
        let t = ScalarTransport::new(grid(), 0.1).unwrap();
        let n = 16 * 16;
        let zero = vec![0.0; n];
        let mut c = vec![0.0; n];
        let too_big = 10.0 * t.max_stable_dt(&zero, &zero);
        assert!(t.step(&mut c, &zero, &zero, too_big, &|_, _| 0.0).is_err());
        assert!(ScalarTransport::new(grid(), -1.0).is_err());
    }
}
