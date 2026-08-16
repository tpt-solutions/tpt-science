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
