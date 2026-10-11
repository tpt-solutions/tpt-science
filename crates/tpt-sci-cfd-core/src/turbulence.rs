//! Lightweight algebraic turbulence modelling for `tpt-sci-cfd-core`.
//!
//! This is an explicit, algebraic eddy-viscosity estimate (mixing-length /
//! Smagorinsky-style), *not* a coupled `k`-`ε` two-equation solve. It is enough
//! to let downstream solvers perturb the effective viscosity for high-Reynolds
//! number flows without pulling in a full turbulence model.
use crate::CfdError;

/// Smagorinsky-style eddy viscosity `ν_t = (C_s·Δ)²·|S|`, where `|S|` is the
/// strain-rate magnitude and `Δ` the filter width. Returns the effective
/// (molecular + turbulent) viscosity.
///
/// # Errors
///
/// Returns [`CfdError::InvalidParameter`] if `nu < 0`, `cs < 0`, `delta <= 0`,
/// `strain < 0`, or any argument is NaN.
pub fn eddy_viscosity(nu: f64, cs: f64, delta: f64, strain: f64) -> Result<f64, CfdError> {
    // `!(x >= 0)`-style checks also reject NaN, which compares false.
    let checks = [
        (nu >= 0.0, "nu must be >= 0"),
        (cs >= 0.0, "cs must be >= 0"),
        (delta > 0.0, "delta must be > 0"),
        (strain >= 0.0, "strain must be >= 0"),
    ];
    if let Some((_, msg)) = checks.iter().find(|(ok, _)| !ok) {
        return Err(CfdError::InvalidParameter((*msg).into()));
    }
    Ok(nu + (cs * delta).powi(2) * strain)
}

/// Strain-rate magnitude `|S| = sqrt(2·S:S)` from velocity gradients.
///
/// # Panics
///
/// Panics if any input is non-finite.
#[must_use]
pub fn strain_magnitude(dudx: f64, dudy: f64, dvdx: f64, dvdy: f64) -> f64 {
    assert!(
        [dudx, dudy, dvdx, dvdy].iter().all(|x| x.is_finite()),
        "gradients must be finite"
    );
    let sxx = dudx;
    let syy = dvdy;
    let sxy = 0.5 * (dudy + dvdx);
    (2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn eddy_viscosity_dominates_at_high_strain() {
        let nu = 1e-3;
        let nt = eddy_viscosity(nu, 0.1, 1.0, 100.0).unwrap();
        assert!(nt > nu);
    }

    #[test]
    fn strain_magnitude_zero_for_uniform() {
        assert_abs_diff_eq!(strain_magnitude(0.0, 0.0, 0.0, 0.0), 0.0, epsilon = 1e-12);
    }
}
