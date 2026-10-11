//! Particle–mesh Ewald (PME / PPPM) electrostatics.
//!
//! Splits the Coulomb energy as [`crate::Ewald`] does: a short-range real-space
//! sum of `erfc(αr)/r`, a self-energy correction, a neutralising background, and
//! a reciprocal-space term. The reciprocal term is evaluated on a mesh instead of
//! by summing over every wavevector:
//!
//! 1. charges are spread onto an `M³` grid with cardinal B-splines of order `p`;
//! 2. a 3-D FFT of the grid gives the structure factor on the mesh;
//! 3. the Ewald kernel is multiplied by the B-spline aliasing correction;
//! 4. an inverse FFT gives the potential on the grid, which is interpolated back
//!    to the particles with the B-spline derivatives to give forces.
//!
//! The cost is `O(N·p³ + M³ log M)` rather than the `O(N·K³)` of the direct
//! reciprocal sum. With the same `α` and cutoff, the energy and forces agree with
//! [`crate::Ewald`] to the mesh's discretisation error, which falls quickly as
//! `M` grows and `p` increases.

use num_complex::Complex;
use tpt_math_linalg::tpt_math_linalg_dense::DVector;
use tpt_math_signal_fft::Fft;

use crate::MdError;
use crate::ewald::erfc;
use crate::particle::Particle;

/// PME electrostatics on a cubic periodic box.
#[derive(Debug, Clone, PartialEq)]
pub struct Pppm {
    box_len: f64,
    alpha: f64,
    rcut: f64,
    mesh: usize,
    order: usize,
}

/// Cardinal B-spline `M_p(x)`, supported on `[0, p]`.
fn cardinal_bspline(p: usize, x: f64) -> f64 {
    if p == 1 {
        return if (0.0..1.0).contains(&x) { 1.0 } else { 0.0 };
    }
    let n = p as f64;
    (x / (n - 1.0)) * cardinal_bspline(p - 1, x)
        + ((n - x) / (n - 1.0)) * cardinal_bspline(p - 1, x - 1.0)
}

/// Derivative `dM_p/dx = M_{p−1}(x) − M_{p−1}(x−1)`.
fn cardinal_bspline_deriv(p: usize, x: f64) -> f64 {
    cardinal_bspline(p - 1, x) - cardinal_bspline(p - 1, x - 1.0)
}

/// Per-axis spreading stencil of one particle: grid indices, weights, and
/// weight derivatives with respect to the fractional grid coordinate.
struct Stencil {
    index: Vec<usize>,
    weight: Vec<f64>,
    dweight: Vec<f64>,
}

impl Pppm {
    /// Construct a validated PME solver.
    ///
    /// `mesh` is the number of grid points per axis; `order` is the B-spline
    /// order (4 is a good default, giving cubic splines).
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if the box, `alpha` or cutoff are invalid
    /// (as for [`crate::Ewald::new`]), if `mesh < 2`, or if `order` is outside
    /// `2..=mesh`.
    pub fn new(
        box_len: f64,
        alpha: f64,
        rcut: f64,
        mesh: usize,
        order: usize,
    ) -> Result<Self, MdError> {
        // Reuse the Ewald validation for the shared parameters.
        crate::Ewald::new(box_len, alpha, 1, rcut)?;
        if mesh < 2 {
            return Err(MdError::InvalidEwald(format!(
                "mesh must be >= 2, got {mesh}"
            )));
        }
        if !(2..=mesh).contains(&order) {
            return Err(MdError::InvalidEwald(format!(
                "order must be in 2..=mesh ({mesh}), got {order}"
            )));
        }
        Ok(Self {
            box_len,
            alpha,
            rcut,
            mesh,
            order,
        })
    }

    /// Stencil of one coordinate `r` along one axis.
    fn stencil(&self, r: f64) -> Stencil {
        let m = self.mesh as f64;
        let frac = (r / self.box_len).rem_euclid(1.0);
        let u = frac * m; // fractional grid coordinate in [0, M)
        let base = u.floor() as i64;
        let p = self.order;
        let mut index = Vec::with_capacity(p);
        let mut weight = Vec::with_capacity(p);
        let mut dweight = Vec::with_capacity(p);
        for k in 0..p as i64 {
            let g = base - k; // grid point receiving charge from this particle
            let x = u - g as f64; // in [k, k+1) ⊂ [0, p)
            index.push(g.rem_euclid(self.mesh as i64) as usize);
            weight.push(cardinal_bspline(p, x));
            dweight.push(cardinal_bspline_deriv(p, x));
        }
        Stencil {
            index,
            weight,
            dweight,
        }
    }

    /// Compute the total electrostatic energy and per-particle forces.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if `charges` and `particles` differ in
    /// length or the positions are not 3-dimensional.
    pub fn energy_forces(
        &self,
        particles: &[Particle],
        charges: &[f64],
    ) -> Result<(f64, Vec<DVector<f64>>), MdError> {
        let n = particles.len();
        if charges.len() != n {
            return Err(MdError::InvalidEwald(
                "charges length must match particles length".into(),
            ));
        }
        if particles.iter().any(|p| p.position.len() != 3) {
            return Err(MdError::InvalidEwald(
                "PME requires 3-dimensional positions".into(),
            ));
        }
        let mut forces = vec![DVector::zeros(3); n];
        let l = self.box_len;
        let alpha = self.alpha;
        let volume = l.powi(3);

        // --- Real space, as in the direct Ewald sum. ---
        let wrap = |x: f64| -> f64 {
            let half = l * 0.5;
            let mut y = x - l * (x / l).floor();
            if y > half {
                y -= l;
            }
            y
        };
        let rcut2 = self.rcut * self.rcut;
        let mut e_real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let raw = particles[j].position.clone() - particles[i].position.clone();
                let dr = DVector::from_fn(3, |k| wrap(raw[k]));
                let r2 = dr.dot(&dr);
                if r2 >= rcut2 || r2 <= 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let qq = charges[i] * charges[j];
                let erfc_ar = erfc(alpha * r);
                e_real += qq * erfc_ar / r;
                let dudr = qq
                    * (-2.0 * alpha / std::f64::consts::PI.sqrt() * (-(alpha * r).powi(2)).exp()
                        / r
                        - erfc_ar / r2);
                let fvec = dr * (dudr / r);
                forces[i] = forces[i].clone() + fvec.clone();
                forces[j] = forces[j].clone() - fvec;
            }
        }

        // --- Self energy and neutralising background (position independent). ---
        let q2sum: f64 = charges.iter().map(|q| q * q).sum();
        let e_self = alpha / std::f64::consts::PI.sqrt() * q2sum;
        let qsum: f64 = charges.iter().sum();
        let e_background = -std::f64::consts::PI / (2.0 * alpha * alpha * volume) * qsum * qsum;

        // --- Reciprocal space on the mesh. ---
        let m = self.mesh;
        let stencils: Vec<[Stencil; 3]> = particles
            .iter()
            .map(|p| {
                [
                    self.stencil(p.position[0]),
                    self.stencil(p.position[1]),
                    self.stencil(p.position[2]),
                ]
            })
            .collect();
        let idx = |ix: usize, iy: usize, iz: usize| ix + m * (iy + m * iz);

        // Spread the charges.
        let mut grid = vec![Complex::new(0.0, 0.0); m * m * m];
        for (s, &q) in stencils.iter().zip(charges) {
            for (kz, &wz) in s[2].weight.iter().enumerate() {
                for (ky, &wy) in s[1].weight.iter().enumerate() {
                    for (kx, &wx) in s[0].weight.iter().enumerate() {
                        let g = idx(s[0].index[kx], s[1].index[ky], s[2].index[kz]);
                        grid[g].re += q * wx * wy * wz;
                    }
                }
            }
        }

        // Forward FFT: F(Q) = FFT3(Q).
        let mut fft = Fft::new();
        let spectrum = fft3(&mut fft, &grid, m, false);

        // Kernel H(n) = (4π/V)·exp(−k²/4α²)/k² / Π_d |D_d(n_d)|², with the B-spline
        // aliasing denominators D_d(n) = Σ_j M_p(j+1)·exp(2πi·n·j/M).
        let p = self.order;
        let aliasing: Vec<f64> = (0..m).map(|nd| aliasing_norm(p, nd, m)).collect();
        let two_pi_over_l = 2.0 * std::f64::consts::PI / l;
        let mut h = vec![0.0; m * m * m];
        for iz in 0..m {
            for iy in 0..m {
                for ix in 0..m {
                    if ix == 0 && iy == 0 && iz == 0 {
                        continue;
                    }
                    let signed = |v: usize| -> f64 {
                        if v <= m / 2 {
                            v as f64
                        } else {
                            v as f64 - m as f64
                        }
                    };
                    let (kx, ky, kz) = (
                        two_pi_over_l * signed(ix),
                        two_pi_over_l * signed(iy),
                        two_pi_over_l * signed(iz),
                    );
                    let k2 = kx * kx + ky * ky + kz * kz;
                    let ewald = (-k2 / (4.0 * alpha * alpha)).exp() / k2;
                    h[idx(ix, iy, iz)] = 4.0 * std::f64::consts::PI / volume * ewald
                        / (aliasing[ix] * aliasing[iy] * aliasing[iz]);
                }
            }
        }

        // Potential on the grid: φ = IFFT(H · F(Q)); energy E = ½ Σ_g Q·φ.
        let hf: Vec<Complex<f64>> = spectrum.iter().zip(&h).map(|(f, &hk)| f * hk).collect();
        let phi: Vec<f64> = fft3(&mut fft, &hf, m, true).iter().map(|c| c.re).collect();
        let e_recip: f64 = 0.5 * grid.iter().zip(&phi).map(|(q, &v)| q.re * v).sum::<f64>();

        // Forces: F_i = −q_i · Σ_g ∇_{r_i}(W_g) φ_g, with ∇ = (M/L)·∂/∂u.
        let scale = m as f64 / l;
        for (i, s) in stencils.iter().enumerate() {
            let q = charges[i];
            let mut f = [0.0; 3];
            for (kz, &wz) in s[2].weight.iter().enumerate() {
                for (ky, &wy) in s[1].weight.iter().enumerate() {
                    for (kx, &wx) in s[0].weight.iter().enumerate() {
                        let dwx = s[0].dweight[kx];
                        let dwy = s[1].dweight[ky];
                        let dwz = s[2].dweight[kz];
                        let g = idx(s[0].index[kx], s[1].index[ky], s[2].index[kz]);
                        let phig = phi[g];
                        f[0] += dwx * wy * wz * phig;
                        f[1] += wx * dwy * wz * phig;
                        f[2] += wx * wy * dwz * phig;
                    }
                }
            }
            let delta =
                DVector::from_row_slice(&[-q * scale * f[0], -q * scale * f[1], -q * scale * f[2]]);
            forces[i] = forces[i].clone() + delta;
        }

        let total = e_real + e_recip + e_background - e_self;
        Ok((total, forces))
    }
}

/// Squared modulus of the B-spline aliasing denominator along one axis,
/// `|Σ_j M_p(j+1)·exp(2πi·n·j/M)|²`, for mesh index `n`.
fn aliasing_norm(p: usize, n: usize, m: usize) -> f64 {
    let mut re = 0.0;
    let mut im = 0.0;
    for j in 0..p - 1 {
        let w = cardinal_bspline(p, (j + 1) as f64);
        let phase = 2.0 * std::f64::consts::PI * (n * j) as f64 / m as f64;
        re += w * phase.cos();
        im += w * phase.sin();
    }
    (re * re + im * im).max(f64::MIN_POSITIVE)
}

/// Separable 3-D FFT of an `m³` grid stored as `ix + m·(iy + m·iz)`. The
/// inverse is unnormalised.
fn fft3(fft: &mut Fft, data: &[Complex<f64>], m: usize, inverse: bool) -> Vec<Complex<f64>> {
    let mut out = data.to_vec();
    let mut line = vec![Complex::new(0.0, 0.0); m];
    for axis in 0..3 {
        let stride = match axis {
            0 => 1,
            1 => m,
            _ => m * m,
        };
        for base in 0..m * m * m {
            // Visit each line once: the line's first element has the axis index 0.
            let axis_index = (base / stride) % m;
            if axis_index != 0 {
                continue;
            }
            for (k, v) in line.iter_mut().enumerate() {
                *v = out[base + k * stride];
            }
            let transformed = if inverse {
                fft.inverse(&line)
            } else {
                fft.forward_cplx(&line)
            };
            for (k, v) in transformed.into_iter().enumerate() {
                out[base + k * stride] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bspline_is_a_partition_of_unity() {
        for p in 2..=6 {
            for step in 0..20 {
                let u = 0.05 * f64::from(step);
                let sum: f64 = (0..p).map(|k| cardinal_bspline(p, u + k as f64)).sum();
                assert!((sum - 1.0).abs() < 1e-12, "p = {p}, u = {u}: sum {sum}");
            }
        }
    }

    #[test]
    fn bspline_derivative_matches_finite_difference() {
        let p = 4;
        for &x in &[0.3, 1.2, 2.7] {
            let h = 1e-6;
            let fd = (cardinal_bspline(p, x + h) - cardinal_bspline(p, x - h)) / (2.0 * h);
            assert!((fd - cardinal_bspline_deriv(p, x)).abs() < 1e-7);
        }
    }

    #[test]
    fn fft3_round_trip_scales_by_volume() {
        let m = 4;
        let mut fft = Fft::new();
        let data: Vec<Complex<f64>> = (0..m * m * m)
            .map(|i| Complex::new((i as f64 * 0.37).sin(), 0.0))
            .collect();
        let spectrum = fft3(&mut fft, &data, m, false);
        let back = fft3(&mut fft, &spectrum, m, true);
        let scale = (m * m * m) as f64;
        for (a, b) in data.iter().zip(&back) {
            assert!((a.re * scale - b.re).abs() < 1e-9);
        }
    }
}
