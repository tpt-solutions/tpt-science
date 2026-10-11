//! Physical constants for the astrodynamics model (km, s, and radians unless
//! stated otherwise).

/// Gravitational parameter of the Earth, μ = GM, in km³·s⁻².
///
/// Useful as a default when working in Earth-Centered Inertial (ECI) frames
/// with kilometre and second units.
pub const EARTH_MU: f64 = 398_600.441_8;

/// Earth's second zonal harmonic `J₂` (dimensionless), the leading oblateness
/// term that drives nodal regression and apsidal precession.
pub const EARTH_J2: f64 = 1.082_626_68e-3;

/// Earth's equatorial radius `Rₑ` in km, the reference length for the `J₂`
/// perturbation (the perturbation scales as `(Rₑ/p)²`).
pub const EARTH_RADIUS_EQ: f64 = 6378.137;

/// Earth's fourth zonal harmonic `J₄` (dimensionless), the next zonal
/// oblateness term after `J₂`.
pub const EARTH_J4: f64 = -1.620_836_15e-6;

/// Reference atmospheric density (kg/m³) for the single-band exponential
/// Earth atmosphere model, evaluated at [`EARTH_ATM_H0_KM`].
///
/// Coefficients are consistent with the ~400 km altitude band of the
/// exponential atmospheric density model tabulated in Vallado,
/// *Fundamentals of Astrodynamics and Applications*, 4th ed., Table 8-4
/// ("Exponential Atmospheric Density Model"), which piecewise-fits Earth's
/// atmosphere with `ρ(h) = ρ0 · exp(-(h - h0) / H)` in each altitude band.
/// Adequate for order-of-magnitude drag work in the ~300-500 km LEO band;
/// not a substitute for a full reference atmosphere (e.g. NRLMSISE-00) for
/// precision work.
pub const EARTH_ATM_RHO0_KG_M3: f64 = 5.428e-13;

/// Reference altitude (km) for [`EARTH_ATM_RHO0_KG_M3`] /
/// [`EARTH_ATM_SCALE_HEIGHT_KM`].
pub const EARTH_ATM_H0_KM: f64 = 400.0;

/// Atmospheric scale height (km) for the ~400 km exponential density band
/// (see [`EARTH_ATM_RHO0_KG_M3`]).
pub const EARTH_ATM_SCALE_HEIGHT_KM: f64 = 58.515;

/// Solar radiation pressure at 1 AU, in N/m² (solar constant / speed of
/// light, ≈ 1361 W/m² / 2.998e8 m/s).
pub const SOLAR_PRESSURE_1AU: f64 = 4.56e-6;

/// One astronomical unit, in km.
pub const ASTRONOMICAL_UNIT_KM: f64 = 1.495_978_707e8;

/// Gravitational parameter of the Sun, `μ☉ = GM☉`, in km³·s⁻².
pub const SUN_MU: f64 = 1.327_124_400_18e11;

/// Gravitational parameter of the Moon, `μ_Moon = GM_Moon`, in km³·s⁻².
pub const MOON_MU: f64 = 4_902.800_66;

/// Mean Earth-Moon distance, in km.
pub const MOON_DISTANCE_KM: f64 = 384_400.0;
