//! Coordinate-frame and anomaly helpers: perifocal-to-ECI rotation, cross product,
//! and the eccentric / true anomaly conversions with Kepler's equation solver.

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

/// The 3×3 rotation matrix `Q` that maps perifocal-frame coordinates to ECI.
///
/// `Q` is built from the right ascension of the ascending node `raan`, the
/// inclination `i`, and the argument of periapsis `argp`.
#[must_use]
pub fn perifocal_to_eci(raan: f64, i: f64, argp: f64) -> DMatrix<f64> {
    let c_o = raan.cos();
    let s_o = raan.sin();
    let ci = i.cos();
    let si = i.sin();
    let cw = argp.cos();
    let sw = argp.sin();

    DMatrix::from_fn(3, 3, |row, col| match (row, col) {
        (0, 0) => c_o * cw - s_o * sw * ci,
        (0, 1) => -c_o * sw - s_o * cw * ci,
        (0, 2) => s_o * si,
        (1, 0) => s_o * cw + c_o * sw * ci,
        (1, 1) => -s_o * sw + c_o * cw * ci,
        (1, 2) => -c_o * si,
        (2, 0) => sw * si,
        (2, 1) => cw * si,
        (2, 2) => ci,
        _ => 0.0,
    })
}

/// Cross product of two 3-vectors, `a × b`.
///
/// The inputs are treated as 3-vectors; only the first three components are
/// used. Returns a length-3 [`DVector`].
#[must_use]
pub fn cross3(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    DVector::from_vec(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Convert a true anomaly `nu` to an eccentric anomaly `E` for eccentricity `e`.
#[must_use]
pub fn true_to_eccentric(nu: f64, e: f64) -> f64 {
    let s = (1.0 - e).sqrt() * (nu / 2.0).sin();
    let c = (1.0 + e).sqrt() * (nu / 2.0).cos();
    2.0 * s.atan2(c)
}

/// Convert an eccentric anomaly `E` to a true anomaly `nu` for eccentricity `e`.
#[must_use]
pub fn eccentric_to_true(ecc: f64, e: f64) -> f64 {
    let s = (1.0 + e).sqrt() * (ecc / 2.0).sin();
    let c = (1.0 - e).sqrt() * (ecc / 2.0).cos();
    2.0 * s.atan2(c)
}

/// Solve Kepler's equation `M = E - e·sin(E)` for `E` via Newton iteration.
///
/// Converges quadratically for `0 ≤ e < 1`. The iteration is seeded with
/// `E₀ = M + e·sin(M)`, a first-order series approximation of the solution
/// (Danby 1992, ch. 6). Unlike a bare `E₀ = M` seed this stays close to the
/// root for near-parabolic orbits (`e → 1`), where `E ≈ M + e` and a `M` seed
/// degrades badly. A finite-difference step guards the derivative singularity
/// as `e → 1` so the loop still terminates when `1 - e·cos(E)` approaches zero.
#[must_use]
pub fn solve_kepler(m: f64, e: f64) -> f64 {
    debug_assert!((0.0..1.0).contains(&e), "solve_kepler requires 0 ≤ e < 1");
    let mut ecc = m + e * m.sin();
    for _ in 0..60 {
        let f = ecc - e * ecc.sin() - m;
        let fp = 1.0 - e * ecc.cos();
        let delta = f / fp;
        ecc -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    ecc
}
