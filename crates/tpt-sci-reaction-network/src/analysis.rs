//! Stoichiometric network analysis: linear conservation-law detection.
//!
//! A conservation law is a vector `c` such that `c · y` is invariant along
//! every trajectory of the network — deterministic (`dy/dt = S·r(y)`) or
//! stochastic (SSA / tau-leaping / CLE) — because `d/dt (c·y) = c·S·r(y) = 0`
//! for every rate vector `r(y)` exactly when `cᵀS = 0`, i.e. `c` lies in the
//! **left null space** of the stoichiometry matrix `S`. This module finds a
//! basis for that null space by row-reducing `Sᵀ` to reduced row-echelon
//! form (RREF) and reading off one basis vector per free column — a
//! textbook approach appropriate for the small, dense networks this crate
//! targets (for very large or ill-conditioned networks a rank-revealing SVD
//! would be the more robust choice).

use crate::model::ReactionSystem;

/// Numerical tolerance below which a pivot candidate is treated as zero.
const PIVOT_EPS: f64 = 1e-9;

impl ReactionSystem {
    /// Linear conservation laws of the network: a basis for the vectors `c`
    /// such that `c · y` is constant along every trajectory (`cᵀS = 0`,
    /// where `S` is [`Self::stoichiometry_matrix`]).
    ///
    /// Each returned vector has length [`Self::n_species`], in the same
    /// species order as [`Self::species_names`]. An empty result means the
    /// network has no non-trivial conservation law (e.g. an "open" system
    /// with birth/death reactions). A network with no reactions at all is
    /// considered to conserve every species independently, so this returns
    /// the standard basis in that case.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the internal `.expect()` used while
    /// searching for a pivot is on a row range that is checked non-empty
    /// immediately beforehand.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_reaction_network::ReactionNetwork;
    ///
    /// // A <-> B: total A + B is conserved.
    /// let sys = ReactionNetwork::from_dsl(
    ///     "kf, A --> B
    ///      kr, B --> A",
    /// )
    /// .unwrap();
    /// let laws = sys.conservation_laws();
    /// assert_eq!(laws.len(), 1);
    /// let c = &laws[0];
    /// // c is proportional to [1, 1] (up to sign/scale): A + B is conserved.
    /// assert!((c[0] - c[1]).abs() < 1e-9);
    ///
    /// // A pure birth-death system (0 --> A --> 0) has no conservation law.
    /// let open = ReactionNetwork::from_dsl(
    ///     "k1, 0 --> A
    ///      k2, A --> 0",
    /// )
    /// .unwrap();
    /// assert!(open.conservation_laws().is_empty());
    /// ```
    #[must_use]
    pub fn conservation_laws(&self) -> Vec<Vec<f64>> {
        let s = self.stoichiometry_matrix(); // n_species x n_reactions
        let n_species = s.len();
        if n_species == 0 {
            return Vec::new();
        }
        let n_reactions = s[0].len();
        if n_reactions == 0 {
            // No reactions at all: every species is trivially conserved on
            // its own, so report the standard basis.
            return (0..n_species)
                .map(|i| {
                    let mut v = vec![0.0; n_species];
                    v[i] = 1.0;
                    v
                })
                .collect();
        }

        // M = S^T, shape (n_reactions x n_species); we row-reduce M to RREF
        // and read the null space (Mv = 0, i.e. c = v satisfies c^T S = 0)
        // off the free columns.
        let mut m: Vec<Vec<f64>> = (0..n_reactions)
            .map(|j| (0..n_species).map(|i| s[i][j]).collect())
            .collect();

        let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; n_reactions];
        let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; n_species];
        let mut row = 0usize;
        for col in 0..n_species {
            if row >= n_reactions {
                break;
            }
            let (best_row, best_val) = (row..n_reactions)
                .map(|r| (r, m[r][col].abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("row range is non-empty since row < n_reactions");
            if best_val < PIVOT_EPS {
                continue; // no usable pivot in this column; it stays free
            }
            m.swap(row, best_row);
            let piv = m[row][col];
            for v in &mut m[row] {
                *v /= piv;
            }
            let pivot_row = m[row].clone();
            for (r, m_row) in m.iter_mut().enumerate() {
                if r == row {
                    continue;
                }
                let factor = m_row[col];
                if factor.abs() < PIVOT_EPS {
                    continue;
                }
                for (c, mv) in m_row.iter_mut().enumerate() {
                    *mv -= factor * pivot_row[c];
                }
            }
            pivot_col_of_row[row] = Some(col);
            pivot_row_of_col[col] = Some(row);
            row += 1;
        }

        (0..n_species)
            .filter(|c| pivot_row_of_col[*c].is_none())
            .map(|free_col| {
                let mut v = vec![0.0; n_species];
                v[free_col] = 1.0;
                for r in 0..n_reactions {
                    if let Some(pivot_col) = pivot_col_of_row[r] {
                        v[pivot_col] = -m[r][free_col];
                    }
                }
                v
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::ReactionNetwork;

    #[test]
    fn isomerization_conserves_total_mass() {
        let sys = ReactionNetwork::from_dsl(
            "kf, A --> B
             kr, B --> A",
        )
        .unwrap();
        let laws = sys.conservation_laws();
        assert_eq!(laws.len(), 1);
        let c = &laws[0];
        assert_eq!(c.len(), 2);
        assert!((c[0] - c[1]).abs() < 1e-9);
        assert!(c[0].abs() > 1e-9);
    }

    #[test]
    fn birth_death_has_no_conservation_law() {
        let sys = ReactionNetwork::from_dsl(
            "k1, 0 --> A
             k2, A --> 0",
        )
        .unwrap();
        assert!(sys.conservation_laws().is_empty());
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // `col` indexes reaction columns of `s`, not entries of `laws`
    fn michaelis_menten_has_two_independent_conservation_laws() {
        // S + E <-> SE -> P + E: enzyme (E + SE) and substrate (S + SE + P)
        // are each conserved, and they are linearly independent.
        let sys = ReactionNetwork::from_dsl(
            "kB, S + E --> SE
             kD, SE --> S + E
             kP, SE --> P + E",
        )
        .unwrap();
        let laws = sys.conservation_laws();
        assert_eq!(laws.len(), 2);
        // Every returned law must actually annihilate S (c^T S == 0).
        let s = sys.stoichiometry_matrix();
        for c in &laws {
            for col in 0..s[0].len() {
                let dot: f64 = (0..c.len()).map(|i| c[i] * s[i][col]).sum();
                assert!(
                    dot.abs() < 1e-9,
                    "law {c:?} does not annihilate column {col}"
                );
            }
        }
    }

    #[test]
    fn no_reactions_returns_standard_basis() {
        let mut net = ReactionNetwork::new();
        net.species("A");
        net.species("B");
        let sys = net.build().unwrap();
        let laws = sys.conservation_laws();
        assert_eq!(laws.len(), 2);
    }
}
