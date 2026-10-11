//! Cell (linked-cell) neighbor lists, with a brute-force reference for cross-checks.

use super::*;

/// A linked-cell (cell-list) neighbor structure for a cubic periodic box,
/// replacing the `O(n²)` pairwise scan used by [`lennard_jones`] with an
/// `O(n)`-ish build plus a 27-cell-stencil query, for a fixed cut-off radius.
#[derive(Debug, Clone)]
pub struct CellList {
    ncells: usize,
    box_len: f64,
    rcut: f64,
    cells: Vec<Vec<usize>>,
}

impl CellList {
    /// Build a cell list for `particles` in a cubic periodic box of side
    /// `box_len` with cut-off radius `rcut`.
    ///
    /// # Errors
    ///
    /// Returns [`MdError::InvalidCellList`] if `box_len` is not finite and
    /// `> 0`, `rcut <= 0`, `rcut > box_len/2`, or any particle is not
    /// 3-dimensional.
    pub fn build(particles: &[Particle], box_len: f64, rcut: f64) -> Result<Self, MdError> {
        if !box_len.is_finite() || box_len <= 0.0 {
            return Err(MdError::InvalidCellList(format!(
                "box_len must be finite and > 0, got {box_len}"
            )));
        }
        if rcut <= 0.0 {
            return Err(MdError::InvalidCellList(format!(
                "rcut must be > 0, got {rcut}"
            )));
        }
        if rcut > box_len * 0.5 {
            return Err(MdError::InvalidCellList(format!(
                "rcut ({rcut}) must be <= box_len/2 ({})",
                box_len * 0.5
            )));
        }
        if particles.iter().any(|p| p.position.len() != 3) {
            return Err(MdError::InvalidCellList(
                "cell lists require 3-dimensional positions".into(),
            ));
        }
        let ncells = ((box_len / rcut).floor() as usize).max(1);
        let cell_size = box_len / ncells as f64;
        let cell_index = |x: f64| -> usize {
            let xin = x - box_len * (x / box_len).floor();
            ((xin / cell_size) as usize).min(ncells - 1)
        };
        let mut cells = vec![Vec::new(); ncells * ncells * ncells];
        for (idx, p) in particles.iter().enumerate() {
            let cx = cell_index(p.position[0]);
            let cy = cell_index(p.position[1]);
            let cz = cell_index(p.position[2]);
            cells[(cx * ncells + cy) * ncells + cz].push(idx);
        }
        Ok(Self {
            ncells,
            box_len,
            rcut,
            cells,
        })
    }

    /// Return every particle-index pair `(i, j)` with `i < j` whose
    /// minimum-image distance is less than `rcut`, found via the 27-cell
    /// stencil around each occupied cell (wrapping periodically). Equivalent
    /// to, but generally much faster than, [`neighbor_pairs_brute_force`].
    #[must_use]
    pub fn neighbor_pairs(&self, particles: &[Particle]) -> Vec<(usize, usize)> {
        let n = self.ncells;
        let rcut2 = self.rcut * self.rcut;
        let box_len = self.box_len;
        let wrap = |x: f64| -> f64 {
            let half = box_len * 0.5;
            let mut y = x - box_len * (x / box_len).floor();
            if y > half {
                y -= box_len;
            }
            y
        };
        let pos: Vec<[f64; 3]> = particles
            .iter()
            .map(|p| [p.position[0], p.position[1], p.position[2]])
            .collect();
        let mut pairs = Vec::new();
        for cx in 0..n {
            for cy in 0..n {
                for cz in 0..n {
                    let cell = &self.cells[(cx * n + cy) * n + cz];
                    if cell.is_empty() {
                        continue;
                    }
                    for dx in -1i64..=1 {
                        for dy in -1i64..=1 {
                            for dz in -1i64..=1 {
                                let ox = (cx as i64 + dx).rem_euclid(n as i64) as usize;
                                let oy = (cy as i64 + dy).rem_euclid(n as i64) as usize;
                                let oz = (cz as i64 + dz).rem_euclid(n as i64) as usize;
                                let other = &self.cells[(ox * n + oy) * n + oz];
                                for &i in cell {
                                    for &j in other {
                                        if i >= j {
                                            continue;
                                        }
                                        let r2: f64 = (0..3)
                                            .map(|k| wrap(pos[j][k] - pos[i][k]).powi(2))
                                            .sum();
                                        if r2 < rcut2 && r2 > 0.0 {
                                            pairs.push((i, j));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // With fewer than 3 cells per side the 27-cell stencil revisits cells, so dedup.
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

/// Return every particle-index pair `(i, j)` with `i < j` whose minimum-image
/// distance in a cubic periodic box is less than `rcut`, via a direct `O(n²)`
/// scan. Used as the correctness reference for [`CellList::neighbor_pairs`].
#[must_use]
pub fn neighbor_pairs_brute_force(
    particles: &[Particle],
    box_len: f64,
    rcut: f64,
) -> Vec<(usize, usize)> {
    let n = particles.len();
    let rcut2 = rcut * rcut;
    let wrap = |x: f64| -> f64 {
        if !box_len.is_finite() {
            return x;
        }
        let half = box_len * 0.5;
        let mut y = x - box_len * (x / box_len).floor();
        if y > half {
            y -= box_len;
        }
        y
    };
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let dim = raw.len();
            let dr = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r2 = dr.dot(&dr);
            if r2 < rcut2 && r2 > 0.0 {
                pairs.push((i, j));
            }
        }
    }
    pairs
}
