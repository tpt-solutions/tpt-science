//! Exact (complex-Bessel) **Womersley** solution for pulsatile flow in a rigid
//! tube.
//!
//! The Womersley solution gives the axial velocity `u(r,t)` driven by an
//! oscillatory pressure gradient `∂p/∂z = -G·e^{iωt}` as
//!
//! ```text
//! u(r,t) = Re{ (G/(i·ω·ρ)) · [1 − J0(i^{3/2}·α·r/R) / J0(i^{3/2}·α)] · e^{i·ω·t} }
//! ```
//!
//! with Womersley number `α = R·√(ω/ν)` and `i^{3/2} = (−1 + i)/√2`. This module
//! evaluates `J0`/`J1` of a complex argument self-contained (converging power
//! series, accurate to ~1e-6 for the Womersley numbers found in physiology,
//! `α ≲ 20`), replacing the approximate parabolic-shape profile in the crate
//! root with the exact analytic form.

use num_complex::Complex;

/// `i^{3/2} = e^{i·3π/4} = (−1 + i)/√2`, the complex factor in the Womersley
/// argument `i^{3/2}·α`.
#[must_use]
pub fn womersley_factor() -> Complex<f64> {
    Complex::new(
        -std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    )
}

/// Womersley number `α = R·√(ω/ν) = R·√(ω·ρ/μ)` for radius `r0`, angular
/// frequency `omega`, and kinematic viscosity `nu`.
#[must_use]
pub fn womersley_number(r0: f64, omega: f64, nu: f64) -> f64 {
    r0 * (omega / nu).sqrt()
}

/// Bessel function of the first kind `J0(z)` for a complex argument, evaluated
/// by a converging power series (valid for all `z`; accurate to ~1e-6 across the
/// Womersley-number range used here, `|z| ≲ 30`).
#[must_use]
pub fn bessel_j0(z: Complex<f64>) -> Complex<f64> {
    bessel_j_series(z, 0)
}

/// Bessel function of the first kind `J1(z)` for a complex argument.
#[must_use]
pub fn bessel_j1(z: Complex<f64>) -> Complex<f64> {
    bessel_j_series(z, 1)
}

fn bessel_j_series(z: Complex<f64>, n: i32) -> Complex<f64> {
    let z_half = z / 2.0;
    let mut fact = 1.0_f64;
    for k in 1..=n {
        fact *= k as f64;
    }
    let mut term: Complex<f64> = z_half.powi(n) / fact;
    let mut sum = term;
    let z2 = -(z_half * z_half);
    let mut k: i32 = 1;
    loop {
        term *= z2 / ((k as f64) * ((k + n) as f64));
        sum += term;
        if term.norm() < 1e-18 {
            break;
        }
        k += 1;
        if k > 400 {
            break;
        }
    }
    sum
}

/// Complex axial-velocity amplitude `U(r)` for a unit pressure-gradient
/// amplitude `G = 1` (`∂p/∂z = -e^{i·ω·t}`), so the physical velocity is
/// `u(r,t) = Re{ U(r)·e^{i·ω·t} }`:
///
/// `U(r) = (1/(i·ω·ρ))·[1 − J0(i^{3/2}·α·r/R) / J0(i^{3/2}·α)]`.
///
/// The result scales linearly with the imposed pressure-gradient amplitude.
#[must_use]
pub fn womersley_complex_velocity(
    r: f64,
    r0: f64,
    alpha: f64,
    omega: f64,
    rho: f64,
) -> Complex<f64> {
    let beta = womersley_factor() * alpha;
    let denom = bessel_j0(beta);
    let num = bessel_j0(beta * (r / r0));
    let f = Complex::new(1.0, 0.0) - num / denom;
    f / (Complex::new(0.0, 1.0) * omega * rho)
}

/// Physical axial velocity `u(r,t)` (real) of the exact Womersley solution for
/// a unit pressure-gradient amplitude, given radius `r`, vessel radius `r0`,
/// Womersley number `alpha`, angular frequency `omega`, density `rho`, and time
/// `t`. Scales linearly with the imposed pressure-gradient amplitude.
#[must_use]
pub fn womersley_velocity_profile(
    r: f64,
    r0: f64,
    alpha: f64,
    omega: f64,
    rho: f64,
    t: f64,
) -> f64 {
    let u = womersley_complex_velocity(r, r0, alpha, omega, rho);
    let phase = Complex::new((omega * t).cos(), (omega * t).sin());
    (u * phase).re
}

/// Analytic Womersley volumetric flow-rate amplitude
/// `Q̃ = (π·R²/(i·ω·ρ))·[1 − 2·J1(i^{3/2}·α)/(i^{3/2}·α·J0(i^{3/2}·α))]`
/// for a unit pressure-gradient amplitude.
#[must_use]
pub fn womersley_flow_rate_analytic(alpha: f64, omega: f64, rho: f64, r0: f64) -> Complex<f64> {
    let beta = womersley_factor() * alpha;
    let ratio = 2.0 * bessel_j1(beta) / (beta * bessel_j0(beta));
    let g = Complex::new(1.0 - ratio.re, -ratio.im);
    (std::f64::consts::PI * r0 * r0 / (Complex::new(0.0, 1.0) * omega * rho)) * g
}

/// Volumetric flow-rate amplitude obtained by numerically integrating the
/// complex velocity profile `U(r)` over the cross-section
/// (`2π·∫ U(r)·r·dr`), using `n` midpoint samples. Used to validate
/// [`womersley_flow_rate_analytic`].
#[must_use]
pub fn womersley_flow_rate_numeric(
    alpha: f64,
    omega: f64,
    rho: f64,
    r0: f64,
    n: usize,
) -> Complex<f64> {
    let dr = r0 / (n as f64);
    let mut sum = Complex::new(0.0, 0.0);
    for i in 0..n {
        let r = (i as f64 + 0.5) * dr;
        let u = womersley_complex_velocity(r, r0, alpha, omega, rho);
        sum += u * (2.0 * std::f64::consts::PI * r * dr);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn bessel_j0_known_real_values() {
        assert_abs_diff_eq!(
            bessel_j0(Complex::new(1.0, 0.0)).re,
            0.7651976866,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            bessel_j0(Complex::new(2.0, 0.0)).re,
            0.2238907791,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            bessel_j0(Complex::new(5.0, 0.0)).re,
            -0.1775967713,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bessel_j1_known_real_values() {
        assert_abs_diff_eq!(
            bessel_j1(Complex::new(1.0, 0.0)).re,
            0.4400505857,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            bessel_j1(Complex::new(2.0, 0.0)).re,
            0.5767248078,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            bessel_j1(Complex::new(5.0, 0.0)).re,
            -0.3275791375,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bessel_j0_matches_series_expansion() {
        // J0(z) = 1 - z^2/4 + z^4/64 - z^6/2304 + ... (small z).
        let z = Complex::new(0.2, 0.1);
        let j0 = bessel_j0(z);
        let series =
            Complex::new(1.0, 0.0) - z * z / 4.0 + z * z * z * z / 64.0 - z.powi(6) / 2304.0;
        assert_abs_diff_eq!(j0.re, series.re, epsilon = 1e-10);
        assert_abs_diff_eq!(j0.im, series.im, epsilon = 1e-10);
    }

    #[test]
    fn bessel_j1_matches_series_expansion() {
        // J1(z) = z/2 - z^3/16 + z^5/384 - z^7/18432 + ... (small z).
        let z = Complex::new(0.2, 0.1);
        let j1 = bessel_j1(z);
        let series = z / 2.0 - z * z * z / 16.0 + z.powi(5) / 384.0 - z.powi(7) / 18432.0;
        assert_abs_diff_eq!(j1.re, series.re, epsilon = 1e-10);
        assert_abs_diff_eq!(j1.im, series.im, epsilon = 1e-10);
    }

    #[test]
    fn low_alpha_profile_is_parabolic() {
        // At low α and t = 0 the profile reduces to the parabolic Poiseuille
        // shape: u(r,0)/u(0,0) = 1 - (r/R)^2.
        let r0 = 1.0;
        let omega = 2.0;
        let nu = 0.04;
        let rho = 1.06;
        let alpha = womersley_number(r0, omega, nu) * 1e-4;
        let u0 = womersley_velocity_profile(0.0, r0, alpha, omega, rho, 0.0);
        let ur = womersley_velocity_profile(0.5 * r0, r0, alpha, omega, rho, 0.0);
        assert_abs_diff_eq!(ur / u0, 1.0 - 0.25, epsilon = 1e-4);
    }

    #[test]
    fn high_alpha_core_flattens() {
        // The velocity-amplitude profile across the core becomes flatter as α
        // grows (the centre approaches the near-wall value → plug flow).
        fn flatness(alpha: f64) -> f64 {
            let r0 = 1.0;
            let omega = 2.0;
            let rho = 1.06;
            let u0 = womersley_complex_velocity(0.0, r0, alpha, omega, rho).norm();
            let uhalf = womersley_complex_velocity(0.5 * r0, r0, alpha, omega, rho).norm();
            if u0 == 0.0 {
                return 0.0;
            }
            1.0 - (u0 - uhalf).abs() / u0
        }
        let low = flatness(0.5);
        let high = flatness(15.0);
        assert!(
            high > low,
            "high-α profile must be flatter than low-α (low={low}, high={high})"
        );
    }

    #[test]
    fn flow_rate_matches_analytic_formula() {
        // Numerically integrating U(r) must equal the closed-form Womersley Q̃.
        for &alpha in &[1.0_f64, 3.0, 8.0, 12.0] {
            let r0 = 1.0;
            let omega = 2.0;
            let rho = 1.06;
            let num = womersley_flow_rate_numeric(alpha, omega, rho, r0, 4000);
            let ana = womersley_flow_rate_analytic(alpha, omega, rho, r0);
            assert_abs_diff_eq!(num.re, ana.re, epsilon = 1e-4);
            assert_abs_diff_eq!(num.im, ana.im, epsilon = 1e-4);
        }
    }
}
