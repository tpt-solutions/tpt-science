//! Environment models: exponential atmosphere, cylindrical eclipse shadow, and
//! cannonball solar radiation pressure.

use tpt_math_linalg::tpt_math_linalg_dense::DVector;

use crate::constants::{
    ASTRONOMICAL_UNIT_KM, EARTH_ATM_H0_KM, EARTH_ATM_RHO0_KG_M3, EARTH_ATM_SCALE_HEIGHT_KM,
    SOLAR_PRESSURE_1AU,
};

/// Exponential atmospheric density model, `ρ(h) = ρ0 · exp(-(h - h0) / H)`.
///
/// A single-band exponential fit to Earth's atmosphere (see
/// [`EARTH_ATM_RHO0_KG_M3`] for the reference and its limitations), adequate
/// for order-of-magnitude drag estimates in the ~300-500 km LEO band.
/// `altitude_km` is the geodetic altitude above the Earth's surface in km.
/// Returns density in kg/m³.
///
/// # Examples
///
/// ```
/// use tpt_sci_astro::{atmospheric_density, EARTH_ATM_H0_KM, EARTH_ATM_RHO0_KG_M3};
/// let rho = atmospheric_density(EARTH_ATM_H0_KM);
/// assert!((rho - EARTH_ATM_RHO0_KG_M3).abs() < 1e-20);
/// assert!(atmospheric_density(500.0) < rho); // density falls off with altitude
/// ```
#[must_use]
pub fn atmospheric_density(altitude_km: f64) -> f64 {
    EARTH_ATM_RHO0_KG_M3 * (-(altitude_km - EARTH_ATM_H0_KM) / EARTH_ATM_SCALE_HEIGHT_KM).exp()
}

/// Cylindrical Earth-shadow (eclipse) test.
///
/// Returns `true` when `pos` (satellite position relative to the central
/// body, km) lies within the infinite cylinder of radius `earth_radius`
/// extending anti-sunward from the Earth, i.e. the satellite is
/// geometrically eclipsed under the standard cylindrical-shadow
/// approximation (ignoring the Sun's finite angular size / penumbra, which
/// is adequate for cannonball SRP on/off modeling).
///
/// `sun_pos` is the Sun's position relative to the central body (same frame
/// and units as `pos`, e.g. ECI km).
#[must_use]
pub fn in_earth_shadow(pos: &DVector<f64>, sun_pos: &DVector<f64>, earth_radius: f64) -> bool {
    let sun_dist = sun_pos.norm();
    if sun_dist <= 0.0 {
        return false;
    }
    let sun_dir = sun_pos.clone() * (1.0 / sun_dist);
    let along = pos.dot(&sun_dir);
    if along >= 0.0 {
        // Satellite is on the sunward side (or in the terminator plane): lit.
        return false;
    }
    let perp = pos.clone() - sun_dir * along;
    perp.norm() < earth_radius
}

/// Cannonball-model solar radiation pressure acceleration magnitude, in
/// km/s².
///
/// `cr` is the dimensionless radiation-pressure coefficient (`1` for a
/// perfectly absorbing surface, up to `2` for perfectly reflecting),
/// `area_to_mass_m2_per_kg` is `A/m` in m²/kg, and `sun_distance_km` is the
/// satellite-to-Sun distance. `P_srp` at 1 AU is [`SOLAR_PRESSURE_1AU`],
/// scaled by the inverse-square law to the actual Sun distance:
/// `F = P_srp · Cr · A/m`.
#[must_use]
pub fn srp_acceleration(cr: f64, area_to_mass_m2_per_kg: f64, sun_distance_km: f64) -> f64 {
    let p_srp = SOLAR_PRESSURE_1AU * (ASTRONOMICAL_UNIT_KM / sun_distance_km).powi(2);
    let accel_m_s2 = p_srp * cr * area_to_mass_m2_per_kg;
    accel_m_s2 / 1000.0
}
