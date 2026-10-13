//! Minimal periodic-table data needed for organic / botanical chemistry.

/// Static data for one element.
#[derive(Debug, PartialEq)]
pub struct ElementData {
    /// Atomic number.
    pub z: u8,
    /// Element symbol (capitalised).
    pub symbol: &'static str,
    /// Standard atomic weight (g/mol).
    pub mass: f64,
    /// Allowed neutral valences, ascending (used for implicit hydrogens).
    pub valences: &'static [u8],
    /// Pauling electronegativity (0 when undefined).
    pub electronegativity: f64,
}

const fn e(
    z: u8,
    symbol: &'static str,
    mass: f64,
    valences: &'static [u8],
    electronegativity: f64,
) -> ElementData {
    ElementData {
        z,
        symbol,
        mass,
        valences,
        electronegativity,
    }
}

static TABLE: &[ElementData] = &[
    e(1, "H", 1.008, &[1], 2.20),
    e(3, "Li", 6.94, &[1], 0.98),
    e(5, "B", 10.81, &[3], 2.04),
    e(6, "C", 12.011, &[4], 2.55),
    e(7, "N", 14.007, &[3, 5], 3.04),
    e(8, "O", 15.999, &[2], 3.44),
    e(9, "F", 18.998, &[1], 3.98),
    e(11, "Na", 22.990, &[1], 0.93),
    e(12, "Mg", 24.305, &[2], 1.31),
    e(14, "Si", 28.085, &[4], 1.90),
    e(15, "P", 30.974, &[3, 5], 2.19),
    e(16, "S", 32.06, &[2, 4, 6], 2.58),
    e(17, "Cl", 35.45, &[1], 3.16),
    e(19, "K", 39.098, &[1], 0.82),
    e(20, "Ca", 40.078, &[2], 1.00),
    e(26, "Fe", 55.845, &[2, 3], 1.83),
    e(30, "Zn", 65.38, &[2], 1.65),
    e(34, "Se", 78.971, &[2, 4, 6], 2.55),
    e(35, "Br", 79.904, &[1], 2.96),
    e(53, "I", 126.904, &[1], 2.66),
];

/// Look up an element by its (case-sensitive, capitalised) symbol.
pub fn by_symbol(symbol: &str) -> Option<&'static ElementData> {
    TABLE.iter().find(|d| d.symbol == symbol)
}

/// Look up an element by atomic number.
pub fn by_z(z: u8) -> Option<&'static ElementData> {
    TABLE.iter().find(|d| d.z == z)
}

/// Atomic mass of the hydrogen atom used for implicit hydrogens.
pub const H_MASS: f64 = 1.008;
