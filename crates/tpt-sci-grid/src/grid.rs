use tpt_math_numeric::Scalar;

use crate::error::GridError;

/// Evenly spaced coordinates in `[start, end]` with `n` points
/// (`start` and `end` inclusive). Generic over the scalar type so callers can
/// build grids in any `tpt-math` scalar vocabulary.
///
/// # Panics
///
/// Panics if the integer point count cannot be represented as `T` (e.g. a
/// fixed-width scalar vocabulary whose `T::from(f64)` returns `None`). With the
/// default `f64` scalar this never happens.
pub fn linspace<T: Scalar>(start: T, end: T, n: usize) -> Vec<T> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        n => {
            let step = (end - start) / T::from((n - 1) as f64).unwrap();
            (0..n)
                .map(|i| start + step * T::from(i as f64).unwrap())
                .collect()
        }
    }
}

/// Boundary-condition treatment applied when assembling a discrete operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// Homogeneous Dirichlet (`u = 0` on the boundary). The corresponding
    /// matrix rows become identity rows so the linear system enforces the
    /// boundary value directly.
    #[default]
    Dirichlet,
    /// Homogeneous Neumann (`du/dn = 0`, zero flux) via one-sided stencils at
    /// the boundary.
    Neumann,
}

/// A uniform 1-D grid over `[x0, x1]` with `n` nodes.
#[derive(Debug, Clone)]
pub struct UniformGrid1D {
    n: usize,
    x0: f64,
    x1: f64,
}

impl UniformGrid1D {
    /// Create a grid with `n` uniformly spaced nodes in `[x0, x1]`.
    /// `n >= 2` and `x1 > x0` are required.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::TooFewPoints`] if `n < 2` or
    /// [`GridError::InvalidDomain`] if `x1 <= x0`.
    pub fn new(n: usize, x0: f64, x1: f64) -> Result<Self, GridError> {
        if n < 2 {
            return Err(GridError::TooFewPoints(2, n));
        }
        if x1 <= x0 {
            return Err(GridError::InvalidDomain(x0, x1));
        }
        Ok(Self { n, x0, x1 })
    }

    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Left endpoint.
    pub fn x0(&self) -> f64 {
        self.x0
    }

    /// Right endpoint.
    pub fn x1(&self) -> f64 {
        self.x1
    }

    /// Uniform node spacing.
    pub fn dx(&self) -> f64 {
        (self.x1 - self.x0) / (self.n as f64 - 1.0)
    }

    /// The node coordinates, in increasing order.
    pub fn coordinates(&self) -> Vec<f64> {
        linspace(self.x0, self.x1, self.n)
    }
}

/// A tensor-product uniform 2-D grid: `nx` nodes in `x` over `[x0, x1]` and
/// `ny` nodes in `y` over `[y0, y1]`. Nodes are addressed `index = ix + iy * nx`.
#[derive(Debug, Clone)]
pub struct UniformGrid2D {
    nx: usize,
    x0: f64,
    x1: f64,
    ny: usize,
    y0: f64,
    y1: f64,
}

impl UniformGrid2D {
    /// Create a tensor-product grid. Requires `nx, ny >= 2` and strictly
    /// ordered domains.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::TooFewPoints`] if `nx < 2` or `ny < 2`, or
    /// [`GridError::InvalidDomain`] if either axis domain is not strictly
    /// ordered (`x1 <= x0` or `y1 <= y0`).
    pub fn new(
        nx: usize,
        x0: f64,
        x1: f64,
        ny: usize,
        y0: f64,
        y1: f64,
    ) -> Result<Self, GridError> {
        if nx < 2 {
            return Err(GridError::TooFewPoints(2, nx));
        }
        if ny < 2 {
            return Err(GridError::TooFewPoints(2, ny));
        }
        if x1 <= x0 {
            return Err(GridError::InvalidDomain(x0, x1));
        }
        if y1 <= y0 {
            return Err(GridError::InvalidDomain(y0, y1));
        }
        Ok(Self {
            nx,
            x0,
            x1,
            ny,
            y0,
            y1,
        })
    }

    /// Nodes along the `x` axis.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Nodes along the `y` axis.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Total node count (`nx * ny`).
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Whether the grid has zero nodes (`nx * ny == 0`). Always `false` for a
    /// valid grid (construction requires `nx, ny >= 2`), provided for the
    /// `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Spacing along `x`.
    pub fn dx(&self) -> f64 {
        (self.x1 - self.x0) / (self.nx as f64 - 1.0)
    }

    /// Spacing along `y`.
    pub fn dy(&self) -> f64 {
        (self.y1 - self.y0) / (self.ny as f64 - 1.0)
    }

    /// Left endpoint of the `x` axis.
    pub fn x0(&self) -> f64 {
        self.x0
    }

    /// Right endpoint of the `x` axis.
    pub fn x1(&self) -> f64 {
        self.x1
    }

    /// Lower endpoint of the `y` axis.
    pub fn y0(&self) -> f64 {
        self.y0
    }

    /// Upper endpoint of the `y` axis.
    pub fn y1(&self) -> f64 {
        self.y1
    }

    /// `x` node coordinates.
    pub fn x_coordinates(&self) -> Vec<f64> {
        linspace(self.x0, self.x1, self.nx)
    }

    /// `y` node coordinates.
    pub fn y_coordinates(&self) -> Vec<f64> {
        linspace(self.y0, self.y1, self.ny)
    }
}

/// A tensor-product uniform 3-D grid: `nx` nodes in `x` over `[x0, x1]`,
/// `ny` nodes in `y` over `[y0, y1]`, and `nz` nodes in `z` over `[z0, z1]`.
///
/// Nodes are addressed with the lexicographic ordering
/// `index = ix + iy·nx + iz·nx·ny` (so `x` is the fastest-varying axis and `z`
/// the slowest). This mirrors [`UniformGrid2D`] (whose own ordering is
/// `ix + iy·nx`) and is the convention used by [`crate::laplacian_3d`] and the
/// feature-gated [`crate::laplacian_3d_sparse`].
#[derive(Debug, Clone)]
pub struct UniformGrid3D {
    nx: usize,
    x0: f64,
    x1: f64,
    ny: usize,
    y0: f64,
    y1: f64,
    nz: usize,
    z0: f64,
    z1: f64,
}

impl UniformGrid3D {
    /// Create a tensor-product 3-D grid. Requires `nx, ny, nz >= 2` and strictly
    /// ordered domains on every axis.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::TooFewPoints`] if any axis has fewer than 2 nodes, or
    /// [`GridError::InvalidDomain`] if any axis domain is not strictly ordered.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        x0: f64,
        x1: f64,
        ny: usize,
        y0: f64,
        y1: f64,
        nz: usize,
        z0: f64,
        z1: f64,
    ) -> Result<Self, GridError> {
        if nx < 2 {
            return Err(GridError::TooFewPoints(2, nx));
        }
        if ny < 2 {
            return Err(GridError::TooFewPoints(2, ny));
        }
        if nz < 2 {
            return Err(GridError::TooFewPoints(2, nz));
        }
        if x1 <= x0 {
            return Err(GridError::InvalidDomain(x0, x1));
        }
        if y1 <= y0 {
            return Err(GridError::InvalidDomain(y0, y1));
        }
        if z1 <= z0 {
            return Err(GridError::InvalidDomain(z0, z1));
        }
        Ok(Self {
            nx,
            x0,
            x1,
            ny,
            y0,
            y1,
            nz,
            z0,
            z1,
        })
    }

    /// Nodes along the `x` axis.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Nodes along the `y` axis.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Nodes along the `z` axis.
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Total node count (`nx * ny * nz`).
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Whether the grid has zero nodes (always `false` for a valid grid).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Spacing along `x`.
    pub fn dx(&self) -> f64 {
        (self.x1 - self.x0) / (self.nx as f64 - 1.0)
    }

    /// Spacing along `y`.
    pub fn dy(&self) -> f64 {
        (self.y1 - self.y0) / (self.ny as f64 - 1.0)
    }

    /// Spacing along `z`.
    pub fn dz(&self) -> f64 {
        (self.z1 - self.z0) / (self.nz as f64 - 1.0)
    }

    /// Left endpoint of the `x` axis.
    pub fn x0(&self) -> f64 {
        self.x0
    }

    /// Right endpoint of the `x` axis.
    pub fn x1(&self) -> f64 {
        self.x1
    }

    /// Lower endpoint of the `y` axis.
    pub fn y0(&self) -> f64 {
        self.y0
    }

    /// Upper endpoint of the `y` axis.
    pub fn y1(&self) -> f64 {
        self.y1
    }

    /// Lower endpoint of the `z` axis.
    pub fn z0(&self) -> f64 {
        self.z0
    }

    /// Upper endpoint of the `z` axis.
    pub fn z1(&self) -> f64 {
        self.z1
    }

    /// `x` node coordinates.
    pub fn x_coordinates(&self) -> Vec<f64> {
        linspace(self.x0, self.x1, self.nx)
    }

    /// `y` node coordinates.
    pub fn y_coordinates(&self) -> Vec<f64> {
        linspace(self.y0, self.y1, self.ny)
    }

    /// `z` node coordinates.
    pub fn z_coordinates(&self) -> Vec<f64> {
        linspace(self.z0, self.z1, self.nz)
    }

    /// Lexicographic node index `ix + iy·nx + iz·nx·ny`.
    #[must_use]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.nx + iz * self.nx * self.ny
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid3d_rejects_invalid() {
        assert!(UniformGrid3D::new(1, 0.0, 1.0, 2, 0.0, 1.0, 2, 0.0, 1.0).is_err());
        assert!(UniformGrid3D::new(2, 1.0, 0.0, 2, 0.0, 1.0, 2, 0.0, 1.0).is_err());
        assert!(UniformGrid3D::new(2, 0.0, 1.0, 2, 0.0, 1.0, 2, 1.0, 0.0).is_err());
    }

    #[test]
    fn grid3d_index_is_lexicographic() {
        let g = UniformGrid3D::new(3, 0.0, 1.0, 4, 0.0, 1.0, 5, 0.0, 1.0).unwrap();
        assert_eq!(g.len(), 60);
        assert_eq!(g.index(0, 0, 0), 0);
        assert_eq!(g.index(1, 0, 0), 1);
        assert_eq!(g.index(0, 1, 0), 3);
        assert_eq!(g.index(0, 0, 1), 12);
        // All indices are distinct and in range.
        let mut seen = vec![false; g.len()];
        for iz in 0..g.nz() {
            for iy in 0..g.ny() {
                for ix in 0..g.nx() {
                    let k = g.index(ix, iy, iz);
                    assert!(!seen[k]);
                    seen[k] = true;
                }
            }
        }
    }
}
