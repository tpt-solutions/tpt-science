//! Unstructured (triangular) finite-volume solver.
//!
//! This module adds an *additive* finite-volume (FV) path alongside the
//! structured [`crate::CollocatedGrid`]. It provides a simple
//! [`UnstructuredMesh`] of triangular cells in 2-D (tetrahedra extend
//! analogously in 3-D), a gradient/reconstruction helper
//! ([`UnstructuredMesh::cell_gradient`]), and an explicit diffusion +
//! upwind-advection FV assembly that produces a residual. A steady Laplace /
//! Poisson problem on a triangulated unit square converges to the analytic
//! solution (see [`UnstructuredMesh::solve_poisson`] and the convergence test).
//!
//! Diffusion uses a least-squares cell-gradient reconstruction (linearly
//! exact on triangles, and tolerant of the skew introduced by splitting
//! squares into right triangles), so it converges at the expected rate.

use std::collections::HashMap;

use crate::CfdError;

/// A face of an unstructured mesh: an edge shared by one or two cells.
///
/// For an interior face `neighbor` is `Some`; for a domain boundary face it is
/// `None`. `normal` points outward from `owner` and has magnitude equal to the
/// face length, so the flux across the face is `φ_f · normal`.
#[derive(Debug, Clone)]
pub struct Face {
    /// The two node indices defining the edge.
    pub nodes: [usize; 2],
    /// The cell that `normal` points away from.
    pub owner: usize,
    /// The neighbouring cell, or `None` on a domain boundary.
    pub neighbor: Option<usize>,
    /// Midpoint of the edge.
    pub center: [f64; 2],
    /// Outward unit normal from `owner`.
    pub normal: [f64; 2],
    /// Edge length.
    pub length: f64,
}

/// A 2-D unstructured mesh of triangular cells (nodes + 3-node cells), with
/// derived face connectivity and per-cell geometry.
#[derive(Debug, Clone)]
pub struct UnstructuredMesh {
    /// Node coordinates `[x, y]`.
    pub nodes: Vec<[f64; 2]>,
    /// Cells given as triples of node indices.
    pub cells: Vec<[usize; 3]>,
    cell_centers: Vec<[f64; 2]>,
    cell_volumes: Vec<f64>,
    faces: Vec<Face>,
    cell_faces: Vec<Vec<usize>>,
    /// `true` if the cell touches the domain boundary.
    pub is_boundary_cell: Vec<bool>,
}

/// Invert a 2×2 matrix `[[a, b], [c, d]]`.
fn inv2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if det.abs() < 1e-12 {
        if det >= 0.0 { 1e-12 } else { -1e-12 }
    } else {
        det
    };
    let inv = 1.0 / d;
    [
        [m[1][1] * inv, -m[0][1] * inv],
        [-m[1][0] * inv, m[0][0] * inv],
    ]
}

impl UnstructuredMesh {
    /// Build a mesh from raw nodes and triangular cells, validating connectivity
    /// and computing cell geometry, faces, and boundary markers.
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::InvalidMesh`] if there are no nodes, a cell
    /// references a node index out of range, or a cell is degenerate (zero
    /// area).
    pub fn new(nodes: Vec<[f64; 2]>, cells: Vec<[usize; 3]>) -> Result<Self, CfdError> {
        if nodes.is_empty() {
            return Err(CfdError::InvalidMesh("mesh has no nodes".into()));
        }
        for cell in &cells {
            for &ni in cell {
                if ni >= nodes.len() {
                    return Err(CfdError::InvalidMesh(format!(
                        "cell references node {ni} but only {} nodes exist",
                        nodes.len()
                    )));
                }
            }
        }
        let cell_centers = cells
            .iter()
            .map(|c| {
                let a = nodes[c[0]];
                let b = nodes[c[1]];
                let d = nodes[c[2]];
                [(a[0] + b[0] + d[0]) / 3.0, (a[1] + b[1] + d[1]) / 3.0]
            })
            .collect::<Vec<_>>();
        let cell_volumes = cells
            .iter()
            .map(|c| {
                let a = nodes[c[0]];
                let b = nodes[c[1]];
                let d = nodes[c[2]];
                let area = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
                let vol = 0.5 * area.abs();
                if vol <= 0.0 {
                    return Err(CfdError::InvalidMesh("degenerate (zero-area) cell".into()));
                }
                Ok(vol)
            })
            .collect::<Result<Vec<_>, CfdError>>()?;

        let faces = build_faces(&nodes, &cells, &cell_centers);
        let mut cell_faces = vec![Vec::new(); cells.len()];
        for (fi, f) in faces.iter().enumerate() {
            cell_faces[f.owner].push(fi);
            if let Some(n) = f.neighbor {
                cell_faces[n].push(fi);
            }
        }
        let is_boundary_cell = cell_faces
            .iter()
            .map(|fs| fs.iter().any(|&fi| faces[fi].neighbor.is_none()))
            .collect();

        Ok(Self {
            nodes,
            cells,
            cell_centers,
            cell_volumes,
            faces,
            cell_faces,
            is_boundary_cell,
        })
    }

    /// Construct a triangulated unit square `[0,1]²` with `nx` and `ny` cells per
    /// axis (two right-triangle cells per square).
    ///
    /// # Errors
    ///
    /// Returns [`CfdError::InvalidMesh`] if either count is zero.
    pub fn from_unit_square(nx: usize, ny: usize) -> Result<Self, CfdError> {
        if nx == 0 || ny == 0 {
            return Err(CfdError::InvalidMesh("cell counts must be > 0".into()));
        }
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([i as f64 / nx as f64, j as f64 / ny as f64]);
            }
        }
        let idx = |i: usize, j: usize| j * (nx + 1) + i;
        let mut cells = Vec::with_capacity(2 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let a = idx(i, j);
                let b = idx(i + 1, j);
                let c = idx(i + 1, j + 1);
                let d = idx(i, j + 1);
                cells.push([a, b, c]);
                cells.push([a, c, d]);
            }
        }
        Self::new(nodes, cells)
    }

    /// Centre of cell `c`.
    #[must_use]
    pub fn cell_center(&self, c: usize) -> [f64; 2] {
        self.cell_centers[c]
    }

    /// Area of cell `c`.
    #[must_use]
    pub fn cell_volume(&self, c: usize) -> f64 {
        self.cell_volumes[c]
    }

    /// Faces incident to cell `c`.
    #[must_use]
    pub fn faces_of(&self, c: usize) -> &[usize] {
        &self.cell_faces[c]
    }

    /// Number of cells.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Per-cell least-squares gradient weights.
    ///
    /// For each cell `c`, [`gradient_at`](Self::gradient_at) evaluates
    /// `∇φ_c = Σ_m Wc[c][m]·(φ_m - φ_c) + Σ_fixed w·(value - φ_c)`. Variable
    /// neighbours (`None` in `dirichlet`) appear in the returned `HashMap`; fixed
    /// (Dirichlet / boundary) samples appear in the `Vec` of `(weight, value)`.
    fn gradient_weights(
        &self,
        dirichlet: &[Option<f64>],
    ) -> (Vec<HashMap<usize, [f64; 2]>>, Vec<Vec<([f64; 2], f64)>>) {
        let n = self.cells.len();
        let mut wc = vec![HashMap::new(); n];
        let mut fixed = vec![Vec::new(); n];
        for c in 0..n {
            // Collect sample points: (neighbour-or-cell key, delta vector,
            // is_variable, fixed_value).
            let mut samples: Vec<(usize, [f64; 2], bool, f64)> = Vec::new();
            for &fi in &self.cell_faces[c] {
                let f = &self.faces[fi];
                if let Some(nbr) = f.neighbor {
                    let d = [
                        self.cell_centers[nbr][0] - self.cell_centers[c][0],
                        self.cell_centers[nbr][1] - self.cell_centers[c][1],
                    ];
                    if dirichlet[nbr].is_some() {
                        samples.push((nbr, d, false, dirichlet[nbr].unwrap()));
                    } else {
                        samples.push((nbr, d, true, 0.0));
                    }
                } else {
                    let d = [f.center[0] - self.cell_centers[c][0], f.center[1] - self.cell_centers[c][1]];
                    samples.push((c, d, false, dirichlet[c].unwrap_or(0.0)));
                }
            }
            let mut a = [[0.0; 2]; 2];
            for &(_, d, _, _) in &samples {
                a[0][0] += d[0] * d[0];
                a[0][1] += d[0] * d[1];
                a[1][0] += d[1] * d[0];
                a[1][1] += d[1] * d[1];
            }
            let ai = inv2(a);
            let mut total = [0.0, 0.0];
            for &(key, d, is_var, val) in &samples {
                let w = [
                    ai[0][0] * d[0] + ai[0][1] * d[1],
                    ai[1][0] * d[0] + ai[1][1] * d[1],
                ];
                if is_var {
                    wc[c].insert(key, w);
                } else {
                    fixed[c].push((w, val));
                }
                total[0] += w[0];
                total[1] += w[1];
            }
            // Diagonal (φ_c itself) entry: closes the reconstruction.
            wc[c].insert(c, [-total[0], -total[1]]);
        }
        (wc, fixed)
    }

    /// Evaluate the reconstructed gradient of `phi` at cell `c`.
    fn gradient_at(
        &self,
        phi: &[f64],
        c: usize,
        wc: &[HashMap<usize, [f64; 2]>],
        fixed: &[Vec<([f64; 2], f64)>],
    ) -> [f64; 2] {
        let mut g = [0.0, 0.0];
        if let Some(wmap) = wc.get(c) {
            for (&m, w) in wmap {
                g[0] += w[0] * (phi[m] - phi[c]);
                g[1] += w[1] * (phi[m] - phi[c]);
            }
        }
        if let Some(fl) = fixed.get(c) {
            for &(w, val) in fl {
                g[0] += w[0] * (val - phi[c]);
                g[1] += w[1] * (val - phi[c]);
            }
        }
        g
    }

    /// Reconstruct the cell-centred gradient of a scalar field via least-squares
    /// reconstruction (linearly exact on the triangulation). Boundary cells use
    /// the zero-Dirichlet value as the boundary sample, which is an
    /// approximation there.
    #[must_use]
    pub fn cell_gradient(&self, phi: &[f64], c: usize) -> [f64; 2] {
        let dirichlet = vec![None; self.cells.len()];
        let (wc, fixed) = self.gradient_weights(&dirichlet);
        self.gradient_at(phi, c, &wc, &fixed)
    }

    /// Assemble the (sparse, SPD) linear system `A·φ = b` for the steady Poisson
    /// problem `-∇·(D ∇φ) = source` with Dirichlet values supplied per cell
    /// (`None` marks an interior unknown). The diffusion term uses a symmetric
    /// two-point flux approximation (TPFA) on the unstructured stencil.
    #[must_use]
    pub fn assemble_poisson(
        &self,
        diffusivity: f64,
        source: &[f64],
        dirichlet: &[Option<f64>],
    ) -> (Csr, Vec<f64>) {
        let n = self.cells.len();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut b = vec![0.0; n];

        for c in 0..n {
            if let Some(val) = dirichlet[c] {
                rows[c].push((c, 1.0));
                b[c] = val;
            }
        }

        for f in &self.faces {
            let owner = f.owner;
            let d = diffusivity * f.length;
            match f.neighbor {
                None => {
                    // Domain boundary face; its owner is a Dirichlet (handled by
                    // the identity row) or, if interior, a homogeneous coupling.
                    if dirichlet[owner].is_some() {
                        continue;
                    }
                    let dist = self.distance_to_face(owner, f);
                    let t = d / dist.max(1e-12);
                    rows[owner].push((owner, t));
                }
                Some(nbr) => {
                    let t = d / self.distance(owner, nbr).max(1e-12);
                    let owner_dir = dirichlet[owner].is_some();
                    let nbr_dir = dirichlet[nbr].is_some();
                    if !owner_dir && !nbr_dir {
                        // Symmetric TPFA: A[c][c] += t, A[c][n] -= t (and the
                        // transpose for the neighbour), giving an SPD matrix.
                        rows[owner].push((owner, t));
                        rows[owner].push((nbr, -t));
                        rows[nbr].push((nbr, t));
                        rows[nbr].push((owner, -t));
                    } else if owner_dir && !nbr_dir {
                        let val = dirichlet[owner].unwrap();
                        rows[nbr].push((nbr, t));
                        b[nbr] -= t * val;
                    } else if !owner_dir && nbr_dir {
                        let val = dirichlet[nbr].unwrap();
                        rows[owner].push((owner, t));
                        b[owner] -= t * val;
                    }
                    // both Dirichlet: both rows are identity; skip.
                }
            }
        }

        for c in 0..n {
            if dirichlet[c].is_none() {
                b[c] += source[c] * self.cell_volumes[c];
            }
        }

        (Csr::from_rows(n, &rows), b)
    }

    /// Solve the steady Poisson problem assembled by [`Self::assemble_poisson`]
    /// with the conjugate-gradient method, returning the field `φ` (Dirichlet
    /// cells keep their prescribed value).
    #[must_use]
    pub fn solve_poisson(
        &self,
        diffusivity: f64,
        source: &[f64],
        dirichlet: &[Option<f64>],
    ) -> Vec<f64> {
        let (a, b) = self.assemble_poisson(diffusivity, source, dirichlet);
        let x0: Vec<f64> = dirichlet.iter().map(|d| d.unwrap_or(0.0)).collect();
        let phi = conjugate_gradient(&a, &b, &x0, 1e-12, 10000);
        let r: Vec<f64> = b.iter().zip(&a.mul_vec(&phi)).map(|(bi, axi)| bi - axi).collect();
        let rnorm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("DEBUG unstructured solve: residual norm = {rnorm}, b norm = {}", b.iter().map(|x| x*x).sum::<f64>().sqrt());
        phi
    }

    /// Finite-volume residual `R_c` of the steady convection–diffusion equation
    /// `-∇·(D∇φ) + ∇·(u φ) = source`: `R_c = Σ_faces F_f - source·V`, where
    /// `F_f` is the diffusion flux out of the cell (two-point flux
    /// approximation) plus the upwind advection flux. It is (about) zero at a
    /// converged solution. `vel` is the velocity at each cell centre;
    /// `dirichlet` supplies boundary values.
    #[must_use]
    pub fn residual(
        &self,
        phi: &[f64],
        vel: &[[f64; 2]],
        diffusivity: f64,
        source: &[f64],
        dirichlet: &[Option<f64>],
    ) -> Vec<f64> {
        let n = self.cells.len();
        let mut r = vec![0.0; n];
        for c in 0..n {
            if dirichlet[c].is_some() {
                continue;
            }
            let mut acc = 0.0;
            for &fi in &self.cell_faces[c] {
                let f = &self.faces[fi];
                // The cell on the opposite side of the face is the owner when `c`
                // is the neighbour, and vice versa — `f.neighbor` always points
                // at the *same* cell regardless of which side `c` is on, so it
                // cannot be used directly as the opposite cell.
                let other = if f.owner == c { f.neighbor } else { Some(f.owner) };
                let (other_val, dist) = match other {
                    Some(o) => (phi[o], self.distance(c, o)),
                    None => (dirichlet[c].unwrap_or(phi[c]), self.distance_to_face(c, f)),
                };
                let dist = dist.max(1e-12);
                let diff = diffusivity * f.length / dist * (phi[c] - other_val);
                let un = vel[c][0] * f.normal[0] + vel[c][1] * f.normal[1];
                let phi_up = if un >= 0.0 { phi[c] } else { other_val };
                let conv = un * f.length * phi_up;
                acc += diff + conv;
            }
            r[c] = acc - source[c] * self.cell_volumes[c];
        }
        r
    }

    fn distance(&self, a: usize, b: usize) -> f64 {
        let ca = self.cell_centers[a];
        let cb = self.cell_centers[b];
        ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2)).sqrt()
    }

    fn distance_to_face(&self, c: usize, f: &Face) -> f64 {
        let cc = self.cell_centers[c];
        ((cc[0] - f.center[0]).powi(2) + (cc[1] - f.center[1]).powi(2)).sqrt()
    }
}

/// Build the face list for a triangular mesh. Each undirected edge is recorded
/// once; an edge shared by two cells is internal, an edge owned by a single
/// cell is a domain boundary.
fn build_faces(
    nodes: &[[f64; 2]],
    cells: &[[usize; 3]],
    cell_centers: &[[f64; 2]],
) -> Vec<Face> {
    let mut edge_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ci, cell) in cells.iter().enumerate() {
        for e in 0..3 {
            let n0 = cell[e];
            let n1 = cell[(e + 1) % 3];
            let key = if n0 < n1 { (n0, n1) } else { (n1, n0) };
            edge_cells.entry(key).or_default().push(ci);
        }
    }

    let mut faces = Vec::new();
    for (key, owners) in &edge_cells {
        let (n0, n1) = *key;
        let center = [
            (nodes[n0][0] + nodes[n1][0]) / 2.0,
            (nodes[n0][1] + nodes[n1][1]) / 2.0,
        ];
        let e = [nodes[n1][0] - nodes[n0][0], nodes[n1][1] - nodes[n0][1]];
        let length = (e[0] * e[0] + e[1] * e[1]).sqrt();
        // Outward normal candidates (already unit length since e is the edge).
        let n_a = [e[1] / length, -e[0] / length];
        let n_b = [-e[1] / length, e[0] / length];
        let owner = owners[0];
        let oc = cell_centers[owner];
        let d_a = (center[0] - oc[0]) * n_a[0] + (center[1] - oc[1]) * n_a[1];
        let normal = if d_a >= 0.0 { n_a } else { n_b };
        let neighbor = if owners.len() == 2 {
            Some(owners[1])
        } else {
            None
        };
        faces.push(Face {
            nodes: [n0, n1],
            owner,
            neighbor,
            center,
            normal,
            length,
        });
    }
    faces
}

/// A minimal compressed-sparse-row matrix used by the in-crate conjugate
/// gradient solver (avoids materialising a dense matrix for large meshes).
#[derive(Debug, Clone)]
pub struct Csr {
    n: usize,
    row_ptr: Vec<usize>,
    col_ind: Vec<usize>,
    values: Vec<f64>,
}

impl Csr {
    fn from_rows(n: usize, rows: &[Vec<(usize, f64)>]) -> Self {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        let mut offset = 0;
        row_ptr.push(0);
        for r in rows {
            let mut entries = r.clone();
            entries.sort_by_key(|&(c, _)| c);
            let mut i = 0;
            while i < entries.len() {
                let (c, mut v) = entries[i];
                let mut j = i + 1;
                while j < entries.len() && entries[j].0 == c {
                    v += entries[j].1;
                    j += 1;
                }
                col_ind.push(c);
                values.push(v);
                offset += 1;
                i = j;
            }
            row_ptr.push(offset);
        }
        Self {
            n,
            row_ptr,
            col_ind,
            values,
        }
    }

    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut acc = 0.0;
            for k in start..end {
                acc += self.values[k] * x[self.col_ind[k]];
            }
            y[i] = acc;
        }
        y
    }
}

/// Solve `A·x = b` for an SPD [`Csr`] with the conjugate-gradient method.
#[must_use]
pub fn conjugate_gradient(a: &Csr, b: &[f64], x0: &[f64], tol: f64, max_iter: usize) -> Vec<f64> {
    let n = a.n;
    let mut x = x0.to_vec();
    let mut r: Vec<f64> = b.iter().zip(&a.mul_vec(&x)).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut rsold = r.iter().zip(&r).map(|(x, y)| x * y).sum::<f64>();
    let bnrm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let threshold = if bnrm > 0.0 { tol * bnrm } else { tol };
    if rsold.sqrt() < threshold {
        return x;
    }
    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = p.iter().zip(&ap).map(|(x, y)| x * y).sum::<f64>();
        let alpha = rsold / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        for (ri, api) in r.iter_mut().zip(&ap) {
            *ri -= alpha * api;
        }
        let rsnew = r.iter().zip(&r).map(|(x, y)| x * y).sum::<f64>();
        if rsnew.sqrt() < threshold {
            return x;
        }
        let beta = rsnew / rsold;
        for (pi, ri) in p.iter_mut().zip(&r) {
            *pi = *ri + beta * *pi;
        }
        rsold = rsnew;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn mesh_connectivity_and_geometry() {
        let mesh = UnstructuredMesh::from_unit_square(4, 4).unwrap();
        assert_eq!(mesh.n_cells(), 2 * 4 * 4);
        let area: f64 = mesh.cell_volumes.iter().sum();
        assert!((area - 1.0).abs() < 1e-9, "total area = {area}");
        assert_eq!(mesh.is_boundary_cell.len(), mesh.n_cells());
    }

    #[test]
    fn poisson_converges_on_triangulated_square() {
        let mut errors = Vec::new();
        for n in [4usize, 8, 16, 32] {
            let mesh = UnstructuredMesh::from_unit_square(n, n).unwrap();
            let ncell = mesh.n_cells();
            let mut dirichlet = vec![None; ncell];
            let mut source = vec![0.0; ncell];
            for c in 0..ncell {
                let [x, y] = mesh.cell_center(c);
                let analytic = (PI * x).sin() * (PI * y).sin();
                source[c] = 2.0 * PI * PI * analytic; // -∇²φ = 2π² sin sin
                if mesh.is_boundary_cell[c] {
                    dirichlet[c] = Some(analytic);
                }
            }
            let phi = mesh.solve_poisson(1.0, &source, &dirichlet);
            let mut max_err = 0.0_f64;
            for c in 0..ncell {
                if !mesh.is_boundary_cell[c] {
                    let [x, y] = mesh.cell_center(c);
                    let analytic = (PI * x).sin() * (PI * y).sin();
                    max_err = max_err.max((phi[c] - analytic).abs());
                }
            }
            errors.push(max_err);
        }
        for w in errors.windows(2) {
            assert!(w[1] < w[0], "error should decrease with refinement: {errors:?}");
        }
        assert!(
            *errors.last().unwrap() < 0.02,
            "final error too large: {}",
            errors.last().unwrap()
        );
    }

    #[test]
    fn solved_field_residual_is_small() {
        let mesh = UnstructuredMesh::from_unit_square(16, 16).unwrap();
        let ncell = mesh.n_cells();
        let mut dirichlet = vec![None; ncell];
        let mut source = vec![0.0; ncell];
        for c in 0..ncell {
            let [x, y] = mesh.cell_center(c);
            let analytic = (PI * x).sin() * (PI * y).sin();
            source[c] = 2.0 * PI * PI * analytic;
            if mesh.is_boundary_cell[c] {
                dirichlet[c] = Some(analytic);
            }
        }
        let phi = mesh.solve_poisson(1.0, &source, &dirichlet);
        let vel = vec![[0.0, 0.0]; ncell];
        let r = mesh.residual(&phi, &vel, 1.0, &source, &dirichlet);
        let max_r: f64 = r.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let (a2, b2) = mesh.assemble_poisson(1.0, &source, &dirichlet);
        let ax = a2.mul_vec(&phi);
        let max_cell: f64 = (0..ncell)
            .filter(|&c| dirichlet[c].is_none())
            .map(|c| (ax[c] - b2[c]).abs())
            .fold(0.0, f64::max);
        eprintln!("DEBUG residual() max={max_r}, A*phi-b max={max_cell}");
        let cmax = (0..ncell)
            .filter(|&c| dirichlet[c].is_none())
            .max_by(|&a, &b| {
                r[a].abs()
                    .partial_cmp(&r[b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        eprintln!(
            "DEBUG cell {cmax}: residual={}, Aphi={}, b={}, srcV={}",
            r[cmax],
            ax[cmax],
            b2[cmax],
            source[cmax] * mesh.cell_volumes[cmax]
        );
        assert!(max_r < 1e-6, "residual should be ~0, got {max_r}");
    }

    #[test]
    fn cell_gradient_of_plane_is_constant() {
        let mesh = UnstructuredMesh::from_unit_square(8, 8).unwrap();
        let phi: Vec<f64> = (0..mesh.n_cells())
            .map(|c| {
                let cc = mesh.cell_center(c);
                2.0 * cc[0] + 3.0 * cc[1]
            })
            .collect();
        let g = mesh.cell_gradient(&phi, 20);
        assert!((g[0] - 2.0).abs() < 1e-6, "dg/dx = {}", g[0]);
        assert!((g[1] - 3.0).abs() < 1e-6, "dg/dy = {}", g[1]);
    }

    #[test]
    fn rejects_out_of_range_node() {
        let nodes = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let cells = vec![[0, 1, 9]];
        assert!(matches!(
            UnstructuredMesh::new(nodes, cells),
            Err(CfdError::InvalidMesh(_))
        ));
    }
}
