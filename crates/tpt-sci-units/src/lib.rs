//! Typed SI quantities and the physical constants shared across tpt-science.
//!
//! The kinetics, climate and astrodynamics crates all need the same handful of
//! constants (the gas constant, the Stefan–Boltzmann constant, Earth's
//! gravitational parameter). Defining them once here, with units in the type,
//! removes the duplicated literals and lets the compiler reject a temperature
//! passed where an energy is expected.
//!
//! The quantity types are thin newtypes around `f64`. They carry no runtime
//! cost and implement only the operations that make dimensional sense. Use
//! [`Kelvin::value`] and friends to get the raw number.
//!
//! ```
//! use tpt_sci_units::{JoulePerMole, Kelvin, arrhenius_factor};
//!
//! // exp(−Eₐ / (R·T)) for Eₐ = 80 kJ/mol at 800 K.
//! let f = arrhenius_factor(JoulePerMole::new(80_000.0), Kelvin::new(800.0));
//! assert!((f - (-80_000.0f64 / (8.314_462_618 * 800.0)).exp()).abs() < 1e-15);
//! ```

use std::fmt;

macro_rules! quantity {
    ($(#[$meta:meta])* $name:ident, $unit:expr) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            /// Wrap a raw value expressed in this quantity's unit.
            #[must_use]
            pub const fn new(value: f64) -> Self {
                Self(value)
            }

            /// The raw value in this quantity's unit.
            #[must_use]
            pub const fn value(self) -> f64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $unit)
            }
        }
    };
}

quantity!(
    /// Absolute temperature, in kelvin.
    Kelvin,
    "K"
);
quantity!(
    /// Molar energy (activation energy, enthalpy), in joules per mole.
    JoulePerMole,
    "J/mol"
);
quantity!(
    /// Molar gas constant units, joules per mole per kelvin.
    JoulePerMoleKelvin,
    "J/(mol·K)"
);
quantity!(
    /// Irradiance or flux density, in watts per square metre.
    WattPerSquareMetre,
    "W/m²"
);
quantity!(
    /// Stefan–Boltzmann constant units, W·m⁻²·K⁻⁴.
    WattPerSquareMetreKelvin4,
    "W/(m²·K⁴)"
);
quantity!(
    /// Length in kilometres (orbital radii and altitudes).
    Kilometre,
    "km"
);
quantity!(
    /// Gravitational parameter units, km³·s⁻².
    KilometreCubedPerSecondSquared,
    "km³/s²"
);

/// Molar gas constant, `R = 8.314 462 618 J·mol⁻¹·K⁻¹` (exact SI value of `N_A·k_B`).
pub const R_GAS: JoulePerMoleKelvin = JoulePerMoleKelvin::new(8.314_462_618);

/// Stefan–Boltzmann constant, `σ = 5.670 374 419 × 10⁻⁸ W·m⁻²·K⁻⁴`.
pub const STEFAN_BOLTZMANN: WattPerSquareMetreKelvin4 =
    WattPerSquareMetreKelvin4::new(5.670_374_419e-8);

/// Earth's gravitational parameter `μ = GM`, in km³·s⁻².
pub const EARTH_MU: KilometreCubedPerSecondSquared =
    KilometreCubedPerSecondSquared::new(398_600.441_8);

/// Earth's equatorial radius, in km.
pub const EARTH_RADIUS_EQ: Kilometre = Kilometre::new(6378.137);

/// The Arrhenius Boltzmann factor `exp(−Eₐ / (R·T))`.
///
/// Multiplying a pre-exponential factor by this gives the rate constant.
#[must_use]
pub fn arrhenius_factor(ea: JoulePerMole, t: Kelvin) -> f64 {
    (-ea.value() / (R_GAS.value() * t.value())).exp()
}

/// Blackbody emission `σ·ε·T⁴` for an emissivity `ε` at temperature `T`.
#[must_use]
pub fn radiative_exitance(emissivity: f64, t: Kelvin) -> WattPerSquareMetre {
    WattPerSquareMetre::new(emissivity * STEFAN_BOLTZMANN.value() * t.value().powi(4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrhenius_factor_matches_closed_form() {
        let f = arrhenius_factor(JoulePerMole::new(50_000.0), Kelvin::new(900.0));
        let expected = (-50_000.0_f64 / (8.314_462_618 * 900.0)).exp();
        assert!((f - expected).abs() < 1e-15);
    }

    #[test]
    fn radiative_exitance_is_stefan_boltzmann() {
        let w = radiative_exitance(1.0, Kelvin::new(288.0));
        let expected = 5.670_374_419e-8 * 288.0f64.powi(4);
        assert!((w.value() - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn display_includes_unit() {
        assert_eq!(Kelvin::new(300.0).to_string(), "300 K");
    }
}
