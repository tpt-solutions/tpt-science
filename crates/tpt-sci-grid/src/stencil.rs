/// Finite-difference stencil definitions.
///
/// Coefficients are given for unit spacing (`h = 1`); when applying a stencil,
/// scale by `1 / h^order`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stencil {
    /// First derivative, 2nd-order central: `(-1, 0, +1) / (2h)`.
    CentralFirstDerivative,
    /// Second derivative, 2nd-order central: `(+1, -2, +1) / h^2`.
    CentralSecondDerivative,
}

impl Stencil {
    /// Returns `(offsets, coefficients)` for the stencil, normalised to unit
    /// spacing. `offsets` are relative node indices (e.g. `[-1, 0, 1]`).
    pub fn coefficients(&self) -> (Vec<isize>, Vec<f64>) {
        match self {
            Stencil::CentralFirstDerivative => (vec![-1, 0, 1], vec![-0.5, 0.0, 0.5]),
            Stencil::CentralSecondDerivative => (vec![-1, 0, 1], vec![1.0, -2.0, 1.0]),
        }
    }
}
