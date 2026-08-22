//! Exchange-correlation (XC) functionals for Kohn–Sham DFT.
//!
//! The crate ships two functionals behind a common [`XcFunctional`] trait:
//!
//! * [`Lda`] — the local-density approximation, wrapping the 1-D solver's
//!   [`crate::lda_xc`] Slater-exchange + Perdew–Zunger-style correlation form.
//! * [`Pbe`] — the Perdew–Burke–Ernzerhof (1996) **generalized-gradient
//!   approximation** (GGA), depending on both the density `ρ` and its gradient
//!   magnitude `|∇ρ|`.
//!
//! Every functional exposes the per-electron exchange-correlation energy density
//! `ε_xc(ρ, |∇ρ|)` together with its partial derivatives with respect to `ρ` and
//! `|∇ρ|`. Those partials are exactly what the Kohn–Sham solver needs to build
//! the XC potential `v_xc = ∂(ρ ε)/∂ρ − ∇·(∂(ρ ε)/∂∇ρ)`.

use std::f64::consts::PI;

/// A Kohn–Sham exchange-correlation functional.
///
/// All methods take the spin-summed density `rho` (in electrons per unit volume)
/// and the gradient magnitude `gr = |∇ρ|` (per unit length), and return the
/// **per-electron** exchange-correlation energy density `ε_xc` and its partial
/// derivatives. The total XC energy is `∫ ρ(r)·ε_xc(r) dr`.
pub trait XcFunctional {
    /// Per-electron exchange-correlation energy density `ε_xc(ρ, |∇ρ|)`.
    fn energy_density(&self, rho: f64, gr: f64) -> f64;

    /// Partial derivative `∂ε_xc/∂ρ` at `(rho, gr)`.
    fn deriv_rho(&self, rho: f64, gr: f64) -> f64;

    /// Partial derivative `∂ε_xc/∂|∇ρ|` at `(rho, gr)`.
    fn deriv_gr(&self, rho: f64, gr: f64) -> f64;

    /// Total exchange-correlation energy `∫ ρ·ε_xc dr` over a uniform grid cell of
    /// volume `dv`.
    fn total_energy(&self, rho: &[f64], gr: &[f64], dv: f64) -> f64 {
        rho.iter()
            .zip(gr)
            .map(|(&r, &g)| r * self.energy_density(r, g))
            .sum::<f64>()
            * dv
    }
}

/// Local-density-approximation functional wrapping [`crate::lda_xc`].
///
/// The gradient magnitude is ignored (this is a pure local functional), so
/// `deriv_gr` is always zero and `v_xc = d(ρ ε)/dρ`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lda;

impl XcFunctional for Lda {
    fn energy_density(&self, rho: f64, _gr: f64) -> f64 {
        crate::lda_xc(rho)
    }

    fn deriv_rho(&self, rho: f64, _gr: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        // ε = -C ρ^(1/3) + ec,  ec = -a ρ / (1 + b ρ)
        let c = (3.0 / 4.0) * (3.0 / PI).powf(1.0 / 3.0);
        let dex = -(1.0 / 3.0) * c * rho.powf(-2.0 / 3.0);
        let a = 0.056;
        let b = 11.4;
        let den = 1.0 + b * rho;
        let dec = -a / den + a * b * rho / (den * den);
        dex + dec
    }

    fn deriv_gr(&self, _rho: f64, _gr: f64) -> f64 {
        0.0
    }
}

/// Perdew–Burke–Ernzerhof (PBE) generalized-gradient-approximation functional
/// (unpolarized, `ζ = 0`).
///
/// Implements the standard PBE exchange enhancement `F_x(s)` and the
/// Perdew–Wang correlation plus gradient-correction `H(r_s, t)`, with the
/// canonical PBE constants. In the zero-gradient (`|∇ρ| → 0`) limit it reduces
/// exactly to the Perdew–Wang local spin-density approximation, which is the
/// local reference used to verify the GGA reduction.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pbe;

// PBE constants (Perdew, Burke, Ernzerhof 1996; atomic units).
const PBE_KAPPA: f64 = 0.804;
const PBE_MU: f64 = 0.2195149727645171;
const PBE_BETA: f64 = 0.06672455060314922;
const PBE_GAMMA: f64 = 0.03109069021011212; // (1 - ln 2)/π²
const PBE_CX: f64 = 0.738558766; // (3/4)(3/π)^{1/3}

/// Per-electron LDA (Perdew–Wang) exchange energy density `-C_x ρ^(1/3)`.
fn pw_exchange(rho: f64) -> f64 {
    -PBE_CX * rho.powf(1.0 / 3.0)
}

/// Per-electron Perdew–Wang (1992) correlation energy density `ε_c^unif(r_s)`.
fn pw_correlation(rho: f64) -> f64 {
    let a0 = 0.0310907;
    let al = 0.21370;
    let b1 = 7.5957;
    let b2 = 3.5876;
    let b3 = 1.6382;
    let b4 = 0.49294;
    let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
    let rsh = rs.sqrt();
    let p = b1 * rsh + b2 * rs + b3 * rs * rsh + b4 * rs * rs;
    let inner = 1.0 + 1.0 / (2.0 * a0 * p);
    -2.0 * a0 * (1.0 + al * rs) * inner.ln()
}

/// Derivative `∂ε_c^unif/∂ρ` of the Perdew–Wang correlation.
fn pw_correlation_deriv_rho(rho: f64) -> f64 {
    let a0 = 0.0310907;
    let al = 0.21370;
    let b1 = 7.5957;
    let b2 = 3.5876;
    let b3 = 1.6382;
    let b4 = 0.49294;
    let rs = (3.0 / (4.0 * PI * rho)).powf(1.0 / 3.0);
    let rsh = rs.sqrt();
    let p = b1 * rsh + b2 * rs + b3 * rs * rsh + b4 * rs * rs;
    let inner = 1.0 + 1.0 / (2.0 * a0 * p);
    let dp_drs = b1 / (2.0 * rsh) + b2 + 1.5 * b3 * rsh + 2.0 * b4 * rs;
    let d_inner_drs = -dp_drs / (2.0 * a0 * p * p);
    let drs_drho = -(1.0 / 3.0) * rs / rho;
    -2.0 * a0 * drs_drho * (al * inner.ln() + (1.0 + al * rs) * d_inner_drs / inner)
}

impl XcFunctional for Pbe {
    fn energy_density(&self, rho: f64, gr: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        let c = (3.0 * PI).powf(1.0 / 3.0);
        let s = gr / (2.0 * c * rho.powf(4.0 / 3.0));
        // Exchange enhancement F_x(s).
        let fx = 1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s * s / PBE_KAPPA);
        let ex = pw_exchange(rho) * fx;

        // Correlation: ε_c = ε_c^unif + H.
        let ec = pw_correlation(rho);
        let t = s; // t == s for the unpolarized PBE denominator (φ = 1).
        let a = (PBE_BETA / PBE_GAMMA) / ((-ec / PBE_GAMMA).exp() - 1.0);
        let num = t * t + a * t.powi(4);
        let den = 1.0 + (PBE_BETA / PBE_GAMMA) * t * t;
        let hh = PBE_GAMMA * (1.0 + (PBE_BETA / PBE_GAMMA) * num / den).ln();
        ex + ec + hh
    }

    fn deriv_rho(&self, rho: f64, gr: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        let c = (3.0 * PI).powf(1.0 / 3.0);
        let d = 2.0 * c * rho.powf(4.0 / 3.0);
        let s = gr / d;
        // Exchange: ε_x = ε_x^unif · F_x(s).
        let ex_unif = pw_exchange(rho);
        let dex_unif = ex_unif / (3.0 * rho);
        let dfx_ds = 2.0 * PBE_MU * s / (1.0 + PBE_MU * s * s / PBE_KAPPA).powi(2);
        let ds_drho = -(4.0 / 3.0) * s / rho;
        let dex_drho = dex_unif
            * (1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s * s / PBE_KAPPA))
            + ex_unif * dfx_ds * ds_drho;

        // Correlation: ec + H, with H depending on ρ via r_s and A.
        let ec = pw_correlation(rho);
        let dec_drho = pw_correlation_deriv_rho(rho);
        let t = s;
        let dt_drho = -(4.0 / 3.0) * t / rho;
        let u = -ec / PBE_GAMMA;
        let e_u = u.exp();
        let a = (PBE_BETA / PBE_GAMMA) / (e_u - 1.0);
        // a = alpha / (exp(u) - 1), u = -ec/gamma  =>
        // da/drho = alpha * exp(u) * (dec_drho / gamma) / (exp(u) - 1)^2
        // (du/drho = -dec_drho/gamma cancels the minus from differentiating
        // 1/(exp(u)-1)).
        let da_drho = (PBE_BETA / PBE_GAMMA) * e_u * (dec_drho / PBE_GAMMA) / (e_u - 1.0).powi(2);

        let alpha = PBE_BETA / PBE_GAMMA;
        let num = t * t + a * t.powi(4);
        let dnum_drho = 2.0 * t * dt_drho + da_drho * t.powi(4) + a * 4.0 * t.powi(3) * dt_drho;
        let den = 1.0 + alpha * t * t;
        let dden_drho = 2.0 * alpha * t * dt_drho;
        let q = alpha * num / den;
        let dq_drho = alpha * (dnum_drho * den - num * dden_drho) / (den * den);
        let dh_drho = PBE_GAMMA * dq_drho / (1.0 + q);

        dex_drho + dec_drho + dh_drho
    }

    fn deriv_gr(&self, rho: f64, gr: f64) -> f64 {
        if rho <= 0.0 || gr <= 0.0 {
            return 0.0;
        }
        let c = (3.0 * PI).powf(1.0 / 3.0);
        let d = 2.0 * c * rho.powf(4.0 / 3.0);
        let s = gr / d;
        // Exchange contribution: ε_x = ε_x^unif · F_x(s), depends on gr via s.
        let ex_unif = pw_exchange(rho);
        let dfx_ds = 2.0 * PBE_MU * s / (1.0 + PBE_MU * s * s / PBE_KAPPA).powi(2);
        let ds_dgr = 1.0 / d;
        let dex_dgr = ex_unif * dfx_ds * ds_dgr;

        // Correlation contribution: only H depends on gr (via t = s).
        let ec = pw_correlation(rho);
        let t = s;
        let dt_dgr = 1.0 / d;
        let a = (PBE_BETA / PBE_GAMMA) / ((-ec / PBE_GAMMA).exp() - 1.0);
        let alpha = PBE_BETA / PBE_GAMMA;
        let num = t * t + a * t.powi(4);
        let dnum_dgr = 2.0 * t * dt_dgr + a * 4.0 * t.powi(3) * dt_dgr;
        let den = 1.0 + alpha * t * t;
        let dden_dgr = 2.0 * alpha * t * dt_dgr;
        let q = alpha * num / den;
        let dq_dgr = alpha * (dnum_dgr * den - num * dden_dgr) / (den * den);
        let dh_dgr = PBE_GAMMA * dq_dgr / (1.0 + q);

        dex_dgr + dh_dgr
    }
}

/// Per-electron Perdew–Wang local-density (LDA, `ζ = 0`) exchange-correlation
/// energy density `ε_xc^LSDA(ρ)`.
///
/// This is the local functional that the PBE GGA reduces to in the
/// zero-gradient limit, and is exposed for testing/verification of the GGA
/// reduction (it is intentionally distinct from the ad-hoc [`crate::lda_xc`]
/// form used by the 1-D solver).
#[must_use]
pub fn pw_lda_xc(rho: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }
    pw_exchange(rho) + pw_correlation(rho)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::lda_xc;

    /// Central finite-difference derivative of a two-argument closure w.r.t. the
    /// first argument (rho), holding gr fixed.
    fn num_drho(f: &dyn Fn(f64, f64) -> f64, rho: f64, gr: f64) -> f64 {
        let h = 1e-7;
        (f(rho + h, gr) - f(rho - h, gr)) / (2.0 * h)
    }

    /// Central finite-difference derivative w.r.t. the second argument (gr).
    fn num_dgr(f: &dyn Fn(f64, f64) -> f64, rho: f64, gr: f64) -> f64 {
        let h = 1e-7;
        (f(rho, gr + h) - f(rho, gr - h)) / (2.0 * h)
    }

    #[test]
    fn pbe_reduces_to_lda_at_zero_gradient() {
        let pbe = Pbe;
        for &rho in &[0.01_f64, 0.1, 0.5, 1.0, 2.0] {
            assert_abs_diff_eq!(
                pbe.energy_density(rho, 0.0),
                pw_lda_xc(rho),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn pbe_finite_and_different_from_lda() {
        let pbe = Pbe;
        let rho = 1.0;
        let gr = 0.5;
        let e_pbe = pbe.energy_density(rho, gr);
        assert!(e_pbe.is_finite());
        // A non-uniform density must give a GGA energy distinct from the LDA one.
        assert!((e_pbe - lda_xc(rho)).abs() > 1e-4);
        // And distinct from the zero-gradient (LDA-limit) PBE value.
        assert!((e_pbe - pbe.energy_density(rho, 0.0)).abs() > 1e-4);
    }

    #[test]
    fn pbe_derivatives_match_numeric() {
        let pbe = Pbe;
        let f = |rho: f64, gr: f64| pbe.energy_density(rho, gr);
        for &(rho, gr) in &[
            (0.2_f64, 0.1),
            (0.5, 0.3),
            (1.0, 0.5),
            (1.5, 0.0),
            (2.0, 0.8),
        ] {
            let dr = pbe.deriv_rho(rho, gr);
            let dg = pbe.deriv_gr(rho, gr);
            assert_abs_diff_eq!(dr, num_drho(&f, rho, gr), epsilon = 1e-6);
            if gr > 0.0 {
                assert_abs_diff_eq!(dg, num_dgr(&f, rho, gr), epsilon = 1e-6);
            }
        }
        // Derivative w.r.t. |∇ρ| vanishes at zero gradient.
        assert_abs_diff_eq!(pbe.deriv_gr(1.0, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn lda_derivative_matches_numeric() {
        let lda = Lda;
        let f = |rho: f64, _gr: f64| lda_xc(rho);
        for &rho in &[0.2_f64, 0.7, 1.3, 2.0] {
            let dr = lda.deriv_rho(rho, 0.0);
            assert_abs_diff_eq!(dr, num_drho(&f, rho, 0.0), epsilon = 1e-7);
        }
    }

    #[test]
    fn lda_gradient_derivative_is_zero() {
        let lda = Lda;
        assert_abs_diff_eq!(lda.deriv_gr(1.0, 0.5), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lda.deriv_gr(1.0, 0.0), 0.0, epsilon = 1e-12);
    }
}
