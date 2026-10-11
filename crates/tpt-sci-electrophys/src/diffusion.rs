//! Anisotropic (tensor) diffusion coefficients and 2-D tensor diffusion operator.

/// A 2×2 symmetric positive-definite diffusion (or conductivity) tensor
/// `[Dxx, Dxy; Dxy, Dyy]` (row-major `[dxx, dxy, dyy]`).
///
/// A diagonal tensor `[d, 0, d]` is isotropic (scalar diffusion `d`); a
/// diagonal `[dxx, 0, dyy]` with `dxx ≠ dyy` gives anisotropic (fibre-oriented)
/// conduction, faster along the stronger axis. Full symmetric tensors encode an
/// arbitrary (rotated) preferred direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffusionTensor {
    /// `xx` component.
    pub xx: f64,
    /// Off-diagonal (symmetric) component `xy = yx`.
    pub xy: f64,
    /// `yy` component.
    pub yy: f64,
}

impl DiffusionTensor {
    /// Build a tensor from its three independent entries.
    #[must_use]
    pub fn new(xx: f64, xy: f64, yy: f64) -> Self {
        Self { xx, xy, yy }
    }

    /// Isotropic tensor `diag(d, d)`.
    #[must_use]
    pub fn isotropic(d: f64) -> Self {
        Self::new(d, 0.0, d)
    }

    /// Diagonal (axis-aligned) tensor `diag(dxx, dyy)`.
    #[must_use]
    pub fn diagonal(dxx: f64, dyy: f64) -> Self {
        Self::new(dxx, 0.0, dyy)
    }

    /// Build from a row-major `[dxx, dxy, dyy]` triple.
    #[must_use]
    pub fn from_array(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    /// Row-major `[dxx, dxy, dyy]` representation.
    #[must_use]
    pub fn as_array(&self) -> [f64; 3] {
        [self.xx, self.xy, self.yy]
    }

    /// Whether the tensor is isotropic (equal diagonal, zero off-diagonal).
    #[must_use]
    pub fn is_isotropic(&self) -> bool {
        (self.xy.abs() < 1e-12) && (self.xx - self.yy).abs() < 1e-12
    }
}

impl Default for DiffusionTensor {
    fn default() -> Self {
        Self::isotropic(1.0)
    }
}

/// Conductivity tensor used for the intracellular/extracellular conductivities
/// `σ_i` / `σ_e` in the bidomain equations. It is exactly [`DiffusionTensor`]
/// — the same 2×2 SPD object — just named for its physical role.
pub type ConductivityTensor = DiffusionTensor;

/// Tensor-weighted discrete Laplacian (anisotropic diffusion operator)
/// `∇·(D·∇φ)` on an `nx × ny` grid with Neumann (clamped) boundaries.
///
/// `tensors` must have length `nx·ny`; each entry is a 2×2 symmetric SPD
/// [`DiffusionTensor`]. For a node `c`, using unit grid spacing:
///
/// ```text
/// vxx = φ[i+1] − 2φ[c] + φ[i−1]
/// vyy = φ[j+1] − 2φ[c] + φ[j−1]
/// vxy = φ[i+1,j+1] − φ[i−1,j+1] − φ[i+1,j−1] + φ[i−1,j−1]
/// (∇·(D∇φ))_c = Dxx·vxx + 2·Dxy·vxy + Dyy·vyy
/// ```
///
/// For an isotropic tensor this reduces to the scalar 5-point Laplacian.
#[must_use]
pub fn tensor_diffusion_2d(
    phi: &[f64],
    nx: usize,
    ny: usize,
    tensors: &[DiffusionTensor],
) -> Vec<f64> {
    let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
    let c_at = |i: usize, j: usize| phi[clamp(i as isize, nx) + clamp(j as isize, ny) * nx];
    let mut out = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            let d = tensors[c];
            let im = c_at(i.wrapping_sub(1), j);
            let ip = c_at(i + 1, j);
            let jm = c_at(i, j.wrapping_sub(1));
            let jp = c_at(i, j + 1);
            let vxx = ip - 2.0 * phi[c] + im;
            let vyy = jp - 2.0 * phi[c] + jm;
            let vxy = c_at(i + 1, j + 1)
                - c_at(i.wrapping_sub(1), j + 1)
                - c_at(i + 1, j.wrapping_sub(1))
                + c_at(i.wrapping_sub(1), j.wrapping_sub(1));
            out[c] = d.xx * vxx + 2.0 * d.xy * vxy + d.yy * vyy;
        }
    }
    out
}
