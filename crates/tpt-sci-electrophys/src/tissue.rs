//! Tissue: a 2-D sheet of cells coupled by diffusion, with monodomain and bidomain steps.

use super::*;

/// Diffusivity of a [`Tissue`] node: isotropic scalar `D` or a per-node 2×2
/// [`DiffusionTensor`] (enabling anisotropic, fibre-oriented conduction).
#[derive(Debug, Clone)]
pub enum Diffusivity {
    /// Isotropic scalar `D` (cm²/s, scaled).
    Scalar(f64),
    /// Per-node tensor; length must equal `nx·ny`.
    Tensor(Vec<DiffusionTensor>),
}

/// A 2-D tissue sheet coupling an [`IonicModel`] to a diffusion operator
/// (monodomain, or full bidomain when enabled).
#[derive(Debug, Clone)]
pub struct Tissue {
    /// `nx·ny` transmembrane potentials `Vm` (mV).
    pub vm: Vec<f64>,
    /// `nx·ny` extracellular potentials `Ve` (mV); all zero in monodomain mode.
    pub ve: Vec<f64>,
    /// Intracellular diffusion coefficient `D_i` (scalar isotropic default).
    pub diff: f64,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    /// Per-node diffusivity (scalar or tensor).
    diffusivity: Diffusivity,
    /// Per-node membrane models.
    cells: Vec<Box<dyn IonicModel>>,
    /// Whether the full bidomain (extracellular) machinery is active.
    bidomain: bool,
    /// Extracellular diffusion coefficient `D_e` (used only in bidomain mode).
    pub(crate) sigma_e: f64,
}

impl Tissue {
    /// Construct an `nx × ny` sheet of resting Hodgkin–Huxley cells (monodomain,
    /// isotropic diffusion `diff`).
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `nx == 0` or `ny == 0`.
    pub fn new(nx: usize, ny: usize, diff: f64) -> Result<Self, ElectrophysError> {
        Self::with_model(nx, ny, Diffusivity::Scalar(diff), HodgkinHuxley::resting())
    }

    /// Construct an `nx × ny` sheet from a chosen [`IonicModel`] (cloned into
    /// every node) and a diffusivity (scalar or tensor).
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `nx == 0`, `ny == 0`, or a
    /// tensor diffusivity's length disagrees with `nx·ny`.
    pub fn with_model(
        nx: usize,
        ny: usize,
        diffusivity: Diffusivity,
        model: impl IonicModel + 'static,
    ) -> Result<Self, ElectrophysError> {
        if nx == 0 || ny == 0 {
            return Err(ElectrophysError::InvalidTissue("dims must be > 0".into()));
        }
        let n = nx * ny;
        if let Diffusivity::Tensor(t) = &diffusivity {
            if t.len() != n {
                return Err(ElectrophysError::InvalidTissue(
                    "tensor diffusivity length must equal nx*ny".into(),
                ));
            }
        }
        let diff = match &diffusivity {
            Diffusivity::Scalar(d) => *d,
            Diffusivity::Tensor(_) => 1.0,
        };
        let cells: Vec<Box<dyn IonicModel>> = (0..n).map(|_| model.clone_box()).collect();
        // Start every node at its cell's resting potential so the membrane state
        // is consistent with the gating variables.
        let vm = cells.iter().map(|c| c.voltage()).collect();
        Ok(Self {
            vm,
            ve: vec![0.0; n],
            diff,
            nx,
            ny,
            diffusivity,
            cells,
            bidomain: false,
            sigma_e: diff,
        })
    }

    /// Switch the solver into bidomain mode, coupling the intracellular field to
    /// an extracellular potential `Ve` solved from `(D_i + D_e)·∇²Ve = −D_i·∇²Vm`.
    pub fn enable_bidomain(&mut self, extracellular_diff: f64) {
        self.bidomain = true;
        self.sigma_e = extracellular_diff;
    }

    /// Override the diffusivity (e.g. to install a per-node tensor for
    /// anisotropy). The `diff` field is ignored once a tensor is set.
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if `tensor.len() != nx·ny`.
    pub fn set_diffusivity(&mut self, diffusivity: Diffusivity) -> Result<(), ElectrophysError> {
        let n = self.nx * self.ny;
        if let Diffusivity::Tensor(t) = &diffusivity {
            if t.len() != n {
                return Err(ElectrophysError::InvalidTissue(
                    "tensor diffusivity length must equal nx*ny".into(),
                ));
            }
        }
        if let Diffusivity::Scalar(d) = &diffusivity {
            self.diff = *d;
        }
        self.diffusivity = diffusivity;
        Ok(())
    }

    /// Index of node `(i, j)`.
    #[must_use]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Depolarize a single node (e.g. to trigger an action potential).
    pub fn stimulate(&mut self, i: usize, j: usize, v: f64) {
        let c = self.idx(i, j);
        self.vm[c] = v;
        // Keep the cell's internal voltage in sync: the ionic current is read
        // from the cell, so a zero-length reaction step sets `v` without moving
        // the gating variables.
        self.cells[c].advance_reaction(v, 0.0);
    }

    /// Solve the extracellular elliptic equation `(σ_i + σ_e)·L·Ve = −σ_i·L·Vm`
    /// for the current transmembrane field, returning `Ve`. The conductivity-
    /// weighted Laplacian `L` is assembled as a [`tpt_sci_grid::sparse::CsrMatrix`]
    /// and the SPD system is solved each call with
    /// [`tpt_sci_grid::sparse::conjugate_gradient`] (Neumann boundaries give a
    /// singular-but-consistent system whose unique zero-mean solution CG
    /// recovers). In monodomain mode this returns a zero vector.
    ///
    /// # Panics
    ///
    /// Panics only if the internal grid cannot be constructed, which cannot
    /// happen for a [`Tissue`] built through [`Tissue::new`] or
    /// [`Tissue::with_model`] (those guarantee `nx, ny > 0`).
    #[must_use]
    pub fn extracellular_potential(&self) -> Vec<f64> {
        if !self.bidomain {
            return vec![0.0; self.nx * self.ny];
        }
        let grid = UniformGrid2D::new(self.nx, 0.0, 1.0, self.ny, 0.0, 1.0).expect("grid");
        let lap = laplacian_2d_sparse(&grid, Boundary::Neumann);
        let di = self.diff;
        let de = self.sigma_e;
        // b = −σ_i · L · Vm  (use the unscaled Laplacian first).
        let lvm = lap.mul_vec(&self.vm);
        let b: Vec<f64> = lvm.iter().map(|x| -di * x).collect();
        // A = (σ_i + σ_e) · L  (scale the Laplacian values in place for the solve).
        let mut a = lap;
        for v in &mut a.values {
            *v *= di + de;
        }
        conjugate_gradient(&a, &b, None, 1e-12, 5000)
    }

    /// Diffusion term `∇·(D∇φ)` (per node) for the field `phi`, using unit grid
    /// spacing and clamped (Neumann) boundaries. For a scalar `D` this is the
    /// 5-point Laplacian `D·(φ[i±1]+φ[j±1]−4φ)`; for a tensor diffusivity it
    /// delegates to [`tensor_diffusion_2d`] (the anisotropic
    /// `Dxx·φ_xx + 2·Dxy·φ_xy + Dyy·φ_yy` operator).
    pub(crate) fn diffusion_term(&self, phi: &[f64]) -> Vec<f64> {
        match &self.diffusivity {
            Diffusivity::Scalar(d) => {
                let nx = self.nx;
                let ny = self.ny;
                let clamp = |i: isize, m: usize| -> usize { i.clamp(0, m as isize - 1) as usize };
                let c_at =
                    |i: usize, j: usize| phi[clamp(i as isize, nx) + clamp(j as isize, ny) * nx];
                let mut out = vec![0.0; nx * ny];
                for j in 0..ny {
                    for i in 0..nx {
                        let c = self.idx(i, j);
                        let lap = c_at(i.wrapping_sub(1), j)
                            + c_at(i + 1, j)
                            + c_at(i, j.wrapping_sub(1))
                            + c_at(i, j + 1)
                            - 4.0 * phi[c];
                        out[c] = d * lap;
                    }
                }
                out
            }
            Diffusivity::Tensor(t) => tensor_diffusion_2d(phi, self.nx, self.ny, t),
        }
    }

    /// Advance the tissue PDE by `dt` with explicit operator splitting: the
    /// reaction term `-I_ion/Cm` plus the (scalar or tensor) diffusion term, and
    /// — in bidomain mode — the extracellular elliptic solve coupling `Ve`.
    pub fn step(&mut self, dt: f64) {
        let vm0 = self.vm.clone();
        let phi: Vec<f64> = if self.bidomain {
            let ve = self.extracellular_potential();
            self.ve = ve.clone();
            vm0.iter().zip(&ve).map(|(a, b)| a + b).collect()
        } else {
            vm0.clone()
        };

        let diff_term = self.diffusion_term(&phi);
        let n = self.nx * self.ny;
        let next: Vec<f64> = (0..n)
            .map(|c| {
                let i_ion = self.cells[c].ionic_current();
                let cm = self.cells[c].capacitance();
                vm0[c] + dt * (-i_ion / cm + diff_term[c])
            })
            .collect();
        // Evolve the internal membrane state at the new transmembrane voltage.
        for (cell, &v) in self.cells.iter_mut().zip(next.iter()).take(n) {
            cell.advance_reaction(v, dt);
        }
        self.vm = next;
    }

    /// Advance one bidomain time step, solving the extracellular elliptic
    /// equation each step before coupling it into the intracellular update.
    ///
    /// This is a thin, explicitly-named wrapper around [`Tissue::step`] that
    /// errors (instead of silently degrading to monodomain) if bidomain mode has
    /// not been enabled via [`Tissue::enable_bidomain`].
    ///
    /// # Errors
    ///
    /// Returns [`ElectrophysError::InvalidTissue`] if [`Tissue::enable_bidomain`]
    /// has not been called.
    pub fn bidomain_step(&mut self, dt: f64) -> Result<(), ElectrophysError> {
        if !self.bidomain {
            return Err(ElectrophysError::InvalidTissue(
                "bidomain mode not enabled; call enable_bidomain first".into(),
            ));
        }
        self.step(dt);
        Ok(())
    }

    /// Maximum transmembrane potential in the sheet (post-stimulus spread
    /// indicator).
    #[must_use]
    pub fn max_voltage(&self) -> f64 {
        self.vm.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}
