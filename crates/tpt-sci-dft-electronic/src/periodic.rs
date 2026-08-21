//! Periodic-boundary-condition Kohn–Sham band structure with k-point sampling.
//!
//! For a 1-D periodic potential `V(x) = V(x + a)` the Bloch / phase-twisted
//! boundary condition is handled exactly in reciprocal space: the wavefunctions
//! are expanded in plane waves `e^{i(k+G)x}` (with `G = 2πn/a` the reciprocal
//! lattice vectors), and the single-particle Hamiltonian is a small, dense, real
//! matrix in that basis. Diagonalizing it at a sequence of **k**-points and
//! averaging over a Monkhorst–Pack set yields the band energy and a basic
//! `E(k)` band structure.

use std::f64::consts::PI;

use crate::eigen::jacobi;

/// A 1-D periodic potential on a lattice of constant `a`, stored by its Fourier
/// coefficients `V_n` in `V(x) = Σ_n V_n·e^{i·2π·n·x/a}`.
///
/// Real potentials have conjugate-symmetric coefficients (`V_{-n} = V_n*`), and
/// this struct stores the real `V_n` for real `V(x)`.
#[derive(Debug, Clone)]
pub struct PeriodicPotential1D {
    /// Lattice constant `a`.
    a: f64,
    /// Fourier coefficients `V_n` (real), addressed by reciprocal index `n`.
    coeffs: Vec<(i32, f64)>,
}

impl PeriodicPotential1D {
    /// Construct from explicit Fourier coefficients on a lattice of constant `a`.
    ///
    /// # Panics
    ///
    /// Panics if `a <= 0`.
    #[must_use]
    pub fn from_coeffs(a: f64, coeffs: Vec<(i32, f64)>) -> Self {
        assert!(a > 0.0, "lattice constant must be positive");
        Self { a, coeffs }
    }

    /// A free-electron potential (`V(x) ≡ 0`): bands are exactly
    /// `E = ℏ²|k+G|²/2m` (atomic units, ℏ = m = 1).
    #[must_use]
    pub fn free(a: f64) -> Self {
        Self::from_coeffs(a, Vec::new())
    }

    /// A weak sinusoidal potential `V(x) = V0·cos(2πx/a)`, i.e.
    /// `V_±1 = V0/2` with all other coefficients zero.
    ///
    /// # Panics
    ///
    /// Panics if `a <= 0`.
    #[must_use]
    pub fn from_cosine(a: f64, v0: f64) -> Self {
        assert!(a > 0.0, "lattice constant must be positive");
        Self::from_coeffs(a, vec![(1, v0 / 2.0), (-1, v0 / 2.0)])
    }

    /// Fourier coefficient `V_n` (zero if not stored).
    #[must_use]
    pub fn coefficient(&self, n: i32) -> f64 {
        self.coeffs
            .iter()
            .find(|(idx, _)| *idx == n)
            .map(|&(_, v)| v)
            .unwrap_or(0.0)
    }

    /// Reciprocal-lattice spacing `G_0 = 2π/a`.
    #[must_use]
    pub fn reciprocal_spacing(&self) -> f64 {
        2.0 * PI / self.a
    }

    /// Lowest band energies `E_n(k)` at a given crystal momentum `k`, using
    /// plane waves `G = 2πn/a` for `n ∈ [−npw, npw]`. Returned ascending.
    ///
    /// The matrix elements are `H_{nm} = ½(k+G_n)²·δ_{nm} + V_{n−m}`; for the
    /// real potentials considered here these are real and symmetric.
    ///
    /// # Panics
    ///
    /// Panics if `npw == 0`.
    #[must_use]
    pub fn band_energies(&self, k: f64, npw: usize) -> Vec<f64> {
        assert!(npw > 0, "need at least one plane wave");
        let g0 = self.reciprocal_spacing();
        let gs: Vec<i32> = (-(npw as i32)..=npw as i32).collect();
        let n = gs.len();
        let mut h = vec![vec![0.0_f64; n]; n];
        for (i, &ni) in gs.iter().enumerate() {
            h[i][i] = 0.5 * (k + g0 * ni as f64).powi(2);
            for (j, &nj) in gs.iter().enumerate() {
                h[i][j] += self.coefficient(ni - nj);
            }
        }
        let (mut eig, _v) = jacobi(&h);
        eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eig
    }

    /// Lowest band gap `E_1(k) − E_0(k)` at crystal momentum `k`.
    ///
    /// # Panics
    ///
    /// Panics if `npw == 0`.
    #[must_use]
    pub fn band_gap(k: f64, npw: usize, pot: &PeriodicPotential1D) -> f64 {
        let e = pot.band_energies(k, npw);
        e[1] - e[0]
    }

    /// A Monkhorst–Pack set of `nk` **k**-points across the first Brillouin zone
    /// `[0, 2π/a)`: `k_i = (i + ½)/nk · 2π/a`.
    ///
    /// # Panics
    ///
    /// Panics if `nk == 0`.
    #[must_use]
    pub fn monkhorst_pack(&self, nk: usize) -> Vec<f64> {
        assert!(nk > 0, "need at least one k-point");
        let span = 2.0 * PI / self.a;
        (0..nk)
            .map(|i| (i as f64 + 0.5) / nk as f64 * span)
            .collect()
    }

    /// Average of the lowest band energy over a Monkhorst–Pack set of `nk`
    /// **k**-points (the k-point-sampled ground band energy).
    ///
    /// # Panics
    ///
    /// Panics if `nk == 0` or `npw == 0`.
    #[must_use]
    pub fn average_ground_band_energy(&self, nk: usize, npw: usize) -> f64 {
        assert!(nk > 0, "need at least one k-point");
        let ks = self.monkhorst_pack(nk);
        ks.iter()
            .map(|&k| self.band_energies(k, npw)[0])
            .sum::<f64>()
            / ks.len() as f64
    }

    /// Full band structure: `(k, E_n(k))` for each k-point in `kpoints`.
    ///
    /// # Panics
    ///
    /// Panics if `npw == 0`.
    #[must_use]
    pub fn band_structure(&self, kpoints: &[f64], npw: usize) -> Vec<(f64, Vec<f64>)> {
        kpoints
            .iter()
            .map(|&k| (k, self.band_energies(k, npw)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn free_electron_bands_match_formula() {
        let a = 2.0;
        let pot = PeriodicPotential1D::free(a);
        let g0 = pot.reciprocal_spacing();
        let k = 0.3;
        let npw = 5;
        let bands = pot.band_energies(k, npw);
        let mut ref_bands: Vec<f64> = (-(npw as i32)..=npw as i32)
            .map(|n| 0.5 * (k + g0 * n as f64).powi(2))
            .collect();
        ref_bands.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(bands.len(), ref_bands.len());
        for (got, want) in bands.iter().zip(&ref_bands) {
            assert_abs_diff_eq!(*got, *want, epsilon = 1e-9);
        }
    }

    #[test]
    fn free_electron_has_no_gap_at_zone_boundary() {
        let a = 2.0;
        let pot = PeriodicPotential1D::free(a);
        let k = std::f64::consts::PI / a; // Brillouin-zone boundary.
        let gap = PeriodicPotential1D::band_gap(k, 5, &pot);
        assert_abs_diff_eq!(gap, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn weak_periodic_potential_opens_gap() {
        let a = 2.0;
        let v0 = 0.3;
        let pot = PeriodicPotential1D::from_cosine(a, v0);
        let k = std::f64::consts::PI / a; // Zone boundary: degeneracy lifted.
        let bands = pot.band_energies(k, 5);
        eprintln!("DEBUG cosine bands: {:?}", bands);
        let gap = PeriodicPotential1D::band_gap(k, 5, &pot);
        eprintln!("DEBUG gap = {}", gap);
        // Coupling V_{±1} = v0/2 splits the degenerate pair by v0.
        assert!(gap > 0.0, "gap must open at the zone boundary");
        assert_abs_diff_eq!(gap, v0, epsilon = 1e-9);
    }

    #[test]
    fn monkhorst_pack_average_is_finite() {
        let a = 3.0;
        let pot = PeriodicPotential1D::from_cosine(a, 0.2);
        let avg = pot.average_ground_band_energy(8, 4);
        assert!(avg.is_finite());
    }

    #[test]
    fn band_structure_returns_energy_vs_k() {
        let a = 2.0;
        let pot = PeriodicPotential1D::free(a);
        let ks = pot.monkhorst_pack(6);
        let bs = pot.band_structure(&ks, 3);
        assert_eq!(bs.len(), 6);
        // Energy is a smooth even function of k; check finite and non-negative.
        for (_, bands) in &bs {
            assert!(bands[0].is_finite());
            assert!(bands[0] >= -1e-12);
        }
    }
}
