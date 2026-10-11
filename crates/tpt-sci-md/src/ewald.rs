//! Ewald summation for periodic long-range electrostatics (direct reciprocal sum).

use super::*;

/// Complementary error function via the Abramowitz & Stegun 7.1.26 rational
/// approximation (max absolute error ~1.5e-7) — enough precision for MD forces
/// and energies without pulling in a special-functions dependency.
pub(crate) fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    poly * (-x * x).exp()
}

/// Real-space part of an Ewald sum shared by [`Ewald`] and [`crate::Pppm`]:
/// the short-range `erfc(α·r)/r` pair energy over minimum-image pairs inside
/// `rcut`. Forces are accumulated into the flat `forces` buffer (`3` entries per
/// particle) and the pair energy is returned.
pub(crate) fn real_space(
    particles: &[Particle],
    charges: &[f64],
    box_len: f64,
    alpha: f64,
    rcut: f64,
    forces: &mut [f64],
) -> f64 {
    let n = particles.len();
    let wrap = |x: f64| -> f64 {
        let half = box_len * 0.5;
        let mut y = x - box_len * (x / box_len).floor();
        if y > half {
            y -= box_len;
        }
        y
    };
    let pos: Vec<[f64; 3]> = particles
        .iter()
        .map(|p| [p.position[0], p.position[1], p.position[2]])
        .collect();
    let rcut2 = rcut * rcut;
    let mut e_real = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dr = [
                wrap(pos[j][0] - pos[i][0]),
                wrap(pos[j][1] - pos[i][1]),
                wrap(pos[j][2] - pos[i][2]),
            ];
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if r2 >= rcut2 || r2 <= 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let qq = charges[i] * charges[j];
            let erfc_ar = erfc(alpha * r);
            e_real += qq * erfc_ar / r;
            let dudr = qq
                * (-2.0 * alpha / std::f64::consts::PI.sqrt() * (-(alpha * r).powi(2)).exp() / r
                    - erfc_ar / r2);
            let f_mag = dudr / r;
            for k in 0..3 {
                let f = dr[k] * f_mag;
                forces[3 * i + k] += f;
                forces[3 * j + k] -= f;
            }
        }
    }
    e_real
}

/// Ewald summation for periodic Coulomb electrostatics in a cubic box.
///
/// Splits `1/r` into a rapidly-converging real-space short-range part
/// (`erfc(α·r)/r`, truncated at [`Ewald::rcut`]) and a reciprocal-space part
/// evaluated as a **direct discrete Fourier sum** over `k`-vectors
/// `k = (2π/L)·(nx, ny, nz)` with `|n| <= kmax` in each dimension, plus the
/// standard self-energy and net-charge background corrections.
///
/// **Implementation note:** this crate has no FFT available (per the
/// dependency policy — hand-rolled code only, ADR 0007), so the reciprocal
/// sum is the direct `O(N·kmax³)` double loop rather than an FFT-based PPPM
/// mesh solve. This is exact (not an approximation of the reciprocal sum
/// itself) but scales worse than PPPM for large systems or high accuracy
/// (large `kmax`); it is intended for the small/teaching-scale systems this
/// crate targets.
#[derive(Debug, Clone)]
pub struct Ewald {
    /// Cubic periodic-box side length.
    pub box_len: f64,
    /// Ewald splitting parameter `α` (inverse length).
    pub alpha: f64,
    /// Maximum reciprocal-lattice index per dimension.
    pub kmax: i32,
    /// Real-space cut-off (must be `<= box_len/2` for minimum-image validity).
    pub rcut: f64,
}

impl Ewald {
    /// Construct a validated Ewald summation.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if `box_len` is not finite and `> 0`,
    /// `alpha <= 0`, `kmax < 1`, or `rcut` is not in `(0, box_len/2]`.
    pub fn new(box_len: f64, alpha: f64, kmax: i32, rcut: f64) -> Result<Self, MdError> {
        if !box_len.is_finite() || box_len <= 0.0 {
            return Err(MdError::InvalidEwald(format!(
                "box_len must be finite and > 0, got {box_len}"
            )));
        }
        if alpha <= 0.0 {
            return Err(MdError::InvalidEwald(format!(
                "alpha must be > 0, got {alpha}"
            )));
        }
        if kmax < 1 {
            return Err(MdError::InvalidEwald(format!(
                "kmax must be >= 1, got {kmax}"
            )));
        }
        if rcut <= 0.0 || rcut > box_len * 0.5 {
            return Err(MdError::InvalidEwald(format!(
                "rcut must be in (0, box_len/2], got {rcut}"
            )));
        }
        Ok(Self {
            box_len,
            alpha,
            kmax,
            rcut,
        })
    }

    /// Compute the total electrostatic energy and per-particle forces.
    ///
    /// `charges[i]` is the point charge of `particles[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidEwald`] if `charges.len() != particles.len()`
    /// or particle positions are not 3-dimensional.
    #[allow(clippy::similar_names)]
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
                "Ewald summation requires 3-dimensional positions".into(),
            ));
        }
        let mut forces = vec![0.0_f64; 3 * n];
        let alpha = self.alpha;
        let box_len = self.box_len;
        let volume = box_len.powi(3);

        // --- Real space: short-range erfc(alpha*r)/r, minimum-image, cut off at rcut. ---
        let e_real = real_space(particles, charges, box_len, alpha, self.rcut, &mut forces);

        // --- Self-energy correction (position-independent, no force contribution). ---
        let q2sum: f64 = charges.iter().map(|q| q * q).sum();
        let e_self = alpha / std::f64::consts::PI.sqrt() * q2sum;

        // --- Net-charge background correction (position-independent). ---
        let qsum: f64 = charges.iter().sum();
        let e_background = -std::f64::consts::PI / (2.0 * alpha * alpha * volume) * qsum * qsum;

        // --- Reciprocal space: direct sum over k = (2*pi/L)*(nx,ny,nz), k != 0. ---
        let two_pi = 2.0 * std::f64::consts::PI;
        let kunit = two_pi / box_len;
        let pos: Vec<[f64; 3]> = particles
            .iter()
            .map(|p| [p.position[0], p.position[1], p.position[2]])
            .collect();
        let mut e_recip = 0.0;
        let mut krs = vec![0.0_f64; n];
        for nx in -self.kmax..=self.kmax {
            for ny in -self.kmax..=self.kmax {
                for nz in -self.kmax..=self.kmax {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }
                    let kx = kunit * f64::from(nx);
                    let ky = kunit * f64::from(ny);
                    let kz = kunit * f64::from(nz);
                    let k2 = kx * kx + ky * ky + kz * kz;
                    let a_k = (-k2 / (4.0 * alpha * alpha)).exp() / k2;

                    let mut c = 0.0_f64;
                    let mut s = 0.0_f64;
                    for idx in 0..n {
                        let kr = kx * pos[idx][0] + ky * pos[idx][1] + kz * pos[idx][2];
                        let (sin_kr, cos_kr) = kr.sin_cos();
                        c += charges[idx] * cos_kr;
                        s += charges[idx] * sin_kr;
                        krs[idx] = kr;
                    }
                    e_recip += a_k * (c * c + s * s);

                    // Force_i = (4*pi/V) * q_i * A(k) * k * [C*sin(k.r_i) - S*cos(k.r_i)]
                    let prefac = 4.0 * std::f64::consts::PI / volume * a_k;
                    for idx in 0..n {
                        let (sin_kr, cos_kr) = krs[idx].sin_cos();
                        let coeff = prefac * charges[idx] * (c * sin_kr - s * cos_kr);
                        forces[3 * idx] += coeff * kx;
                        forces[3 * idx + 1] += coeff * ky;
                        forces[3 * idx + 2] += coeff * kz;
                    }
                }
            }
        }
        e_recip *= two_pi / volume;

        let total = e_real + e_recip + e_background - e_self;
        let forces = forces
            .chunks_exact(3)
            .map(DVector::from_row_slice)
            .collect();
        Ok((total, forces))
    }
}
