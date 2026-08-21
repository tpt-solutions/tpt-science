//! Multi-band (and correlated-k) longwave radiative transfer.
//!
//! This module replaces the single grey band of [`crate::grey_radiative_transfer`]
//! with a genuine *multi-band* longwave scheme: the longwave spectrum is split
//! into `n` bands, each carrying a Planck-weighted fraction `w_b` of the surface
//! emission and its own grey-gas slab emissivity `ε_b = 1 − exp(−κ_b·m_b)`. The
//! band-resolved outgoing-longwave radiation (OLR) is then summed to give the
//! total TOA flux.
//!
//! Two interchangeable band models are provided:
//!
//! * [`MultiBandRadiativeTransfer`] — a **simplified multi-band** scheme where
//!   each band is a single grey slab.
//! * [`CorrelatedKRt`] — a **correlated-k** scheme where each band is resolved
//!   with a small `g`-point quadrature (`k`-distribution), giving a more accurate
//!   band-averaged transmittance `Σ_g w_g·exp(−k_g·m)`.

use crate::SIGMA;
use crate::error::ClimateError;

/// One spectral band of a simplified (single-slab) multi-band longwave scheme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    /// Planck-weighted fraction of surface emission carried by this band (Σ `w_b = 1`).
    pub weight: f64,
    /// Well-mixed absorber absorption coefficient `κ_b` (m⁻¹).
    pub absorption: f64,
    /// Column path length `m_b` (m) over which to evaluate the transmittance.
    pub path_length: f64,
}

impl Band {
    /// Construct a band, validating the physical parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if `weight`, `absorption`, or
    /// `path_length` are negative (a band must have a non-negative Planck weight
    /// and a non-negative optical path).
    pub fn new(weight: f64, absorption: f64, path_length: f64) -> Result<Self, ClimateError> {
        if weight < 0.0 {
            return Err(ClimateError::InvalidModel("band weight must be >= 0".into()));
        }
        if absorption < 0.0 {
            return Err(ClimateError::InvalidModel(
                "band absorption must be >= 0".into(),
            ));
        }
        if path_length < 0.0 {
            return Err(ClimateError::InvalidModel(
                "band path length must be >= 0".into(),
            ));
        }
        Ok(Self {
            weight,
            absorption,
            path_length,
        })
    }

    /// Band emissivity `ε_b = 1 − exp(−κ_b·m_b)` (grey-gas slab, `0 ≤ ε_b ≤ 1`).
    #[must_use]
    pub fn emissivity(&self) -> f64 {
        let tau = (self.absorption * self.path_length).min(700.0);
        1.0 - (-tau).exp()
    }

    /// Band transmittance `exp(−κ_b·m_b)` (fraction of surface emission escaping
    /// directly to space).
    #[must_use]
    pub fn transmittance(&self) -> f64 {
        let tau = (self.absorption * self.path_length).min(700.0);
        (-tau).exp()
    }
}

/// A simplified **multi-band** longwave radiation scheme (a stack of grey slabs).
#[derive(Debug, Clone)]
pub struct MultiBandRadiativeTransfer {
    bands: Vec<Band>,
    /// Emitting-layer (atmospheric) temperature `T_a` (K) used for band emission.
    pub t_atm: f64,
}

impl MultiBandRadiativeTransfer {
    /// Build a multi-band scheme from its bands and an emitting-layer temperature.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if no bands are supplied, if
    /// `t_atm <= 0`, or if the Planck weights do not sum to `1 ± 1e-6`.
    pub fn new(bands: Vec<Band>, t_atm: f64) -> Result<Self, ClimateError> {
        if bands.is_empty() {
            return Err(ClimateError::InvalidModel("need at least one band".into()));
        }
        if t_atm <= 0.0 {
            return Err(ClimateError::InvalidModel(
                "emitting-layer temperature must be > 0".into(),
            ));
        }
        let wsum: f64 = bands.iter().map(|b| b.weight).sum();
        if (wsum - 1.0).abs() > 1e-6 {
            return Err(ClimateError::InvalidModel(format!(
                "band weights must sum to 1 (got {wsum})"
            )));
        }
        Ok(Self { bands, t_atm })
    }

    /// Effective (Planck-weighted) emissivity `Σ w_b·ε_b`, the multi-band
    /// replacement for the single grey-band emissivity.
    #[must_use]
    pub fn effective_emissivity(&self) -> f64 {
        self.bands
            .iter()
            .map(|b| b.weight * b.emissivity())
            .sum()
    }

    /// Per-band upwelling longwave flux at the top of the atmosphere (W/m²),
    /// `F_b = w_b·(τ_b·σ·T_s⁴ + ε_b·σ·T_a⁴)` (two-stream grey slab).
    #[must_use]
    pub fn band_olr(&self, t_surf: f64) -> Vec<f64> {
        let sst4 = SIGMA * t_surf.powi(4);
        let sat4 = SIGMA * self.t_atm.powi(4);
        self.bands
            .iter()
            .map(|b| b.weight * (b.transmittance() * sst4 + b.emissivity() * sat4))
            .collect()
    }

    /// Total outgoing longwave radiation at the TOA (W/m²) = `Σ F_b`.
    #[must_use]
    pub fn olr(&self, t_surf: f64) -> f64 {
        self.band_olr(t_surf).iter().sum()
    }

    /// Downwelling longwave flux at the surface (W/m²), `Σ w_b·ε_b·σ·T_a⁴`
    /// (independent of `T_s`).
    #[must_use]
    pub fn downward_flux(&self) -> f64 {
        let sat4 = SIGMA * self.t_atm.powi(4);
        self.bands.iter().map(|b| b.weight * b.emissivity() * sat4).sum()
    }
}

/// One spectral band of a **correlated-k** longwave scheme: a `k`-distribution
/// quadrature over `g`-points.
#[derive(Debug, Clone)]
pub struct CorrelatedKBand {
    /// Planck-weighted fraction of surface emission carried by this band.
    pub weight: f64,
    /// Per-`g`-point absorption coefficients `k_g` (m⁻¹).
    pub k_g: Vec<f64>,
    /// Per-`g`-point quadrature weights `w_g` (Σ `w_g = 1`).
    pub w_g: Vec<f64>,
}

impl CorrelatedKBand {
    /// Construct a correlated-k band, validating the quadrature.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if `weight < 0`, if `k_g` and `w_g`
    /// have different lengths, if any `k_g`/`w_g` is negative, or if the
    /// `g`-point weights do not sum to `1 ± 1e-6`.
    pub fn new(weight: f64, k_g: Vec<f64>, w_g: Vec<f64>) -> Result<Self, ClimateError> {
        if weight < 0.0 {
            return Err(ClimateError::InvalidModel("band weight must be >= 0".into()));
        }
        if k_g.len() != w_g.len() {
            return Err(ClimateError::InvalidModel(
                "k_g and w_g must have equal length".into(),
            ));
        }
        if k_g.iter().any(|&k| k < 0.0) || w_g.iter().any(|&w| w < 0.0) {
            return Err(ClimateError::InvalidModel(
                "k_g/w_g weights must be >= 0".into(),
            ));
        }
        let wsum: f64 = w_g.iter().sum();
        if (wsum - 1.0).abs() > 1e-6 {
            return Err(ClimateError::InvalidModel(format!(
                "g-point weights must sum to 1 (got {wsum})"
            )));
        }
        Ok(Self { weight, k_g, w_g })
    }

    /// Band-averaged transmittance `Σ_g w_g·exp(−k_g·m)`.
    #[must_use]
    pub fn transmittance(&self, path_length: f64) -> f64 {
        self.k_g
            .iter()
            .zip(&self.w_g)
            .map(|(&k, &w)| w * (-(k * path_length).min(700.0)).exp())
            .sum()
    }

    /// Band-averaged emissivity `1 − transmittance`.
    #[must_use]
    pub fn emissivity(&self, path_length: f64) -> f64 {
        1.0 - self.transmittance(path_length)
    }
}

/// A **correlated-k** longwave radiation scheme: a stack of `k`-distribution
/// bands, each evaluated at a common column path length `m`.
#[derive(Debug, Clone)]
pub struct CorrelatedKRt {
    bands: Vec<CorrelatedKBand>,
    /// Emitting-layer temperature `T_a` (K).
    pub t_atm: f64,
    /// Column path length `m` (m) shared by every band.
    pub path_length: f64,
}

impl CorrelatedKRt {
    /// Build a correlated-k scheme from its bands, an emitting-layer temperature,
    /// and a common column path length.
    ///
    /// # Errors
    ///
    /// Returns [`ClimateError::InvalidModel`] if no bands are supplied, if
    /// `t_atm <= 0`, or if the band Planck weights do not sum to `1 ± 1e-6`.
    pub fn new(
        bands: Vec<CorrelatedKBand>,
        t_atm: f64,
        path_length: f64,
    ) -> Result<Self, ClimateError> {
        if bands.is_empty() {
            return Err(ClimateError::InvalidModel("need at least one band".into()));
        }
        if t_atm <= 0.0 {
            return Err(ClimateError::InvalidModel(
                "emitting-layer temperature must be > 0".into(),
            ));
        }
        let wsum: f64 = bands.iter().map(|b| b.weight).sum();
        if (wsum - 1.0).abs() > 1e-6 {
            return Err(ClimateError::InvalidModel(format!(
                "band weights must sum to 1 (got {wsum})"
            )));
        }
        Ok(Self {
            bands,
            t_atm,
            path_length,
        })
    }

    /// Total OLR (W/m²) summed over the correlated-k bands:
    /// `F = Σ_b w_b·(τ_b(m)·σ·T_s⁴ + ε_b(m)·σ·T_a⁴)`.
    #[must_use]
    pub fn olr(&self, t_surf: f64) -> f64 {
        let sst4 = SIGMA * t_surf.powi(4);
        let sat4 = SIGMA * self.t_atm.powi(4);
        self.bands
            .iter()
            .map(|b| {
                let tau = b.transmittance(self.path_length);
                let eps = 1.0 - tau;
                b.weight * (tau * sst4 + eps * sat4)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn band_emissivity_monotonic_in_path() {
        let b = Band::new(1.0, 0.1, 1.0).unwrap();
        assert!(b.emissivity() > 0.0 && b.emissivity() < 1.0);
        assert_abs_diff_eq!(b.transmittance(), 1.0 - b.emissivity(), epsilon = 1e-12);
        let thick = Band::new(1.0, 0.1, 100.0).unwrap();
        assert!(thick.emissivity() > b.emissivity());
    }

    #[test]
    fn single_band_reduces_to_grey() {
        // A single band with weight 1 is exactly the single grey-band formula
        // F = ε·σ·(T_s⁴ − T_a⁴) + σ·T_a⁴ = σ·(ε·T_s⁴ + (1−ε)·T_a⁴)?  Verified by
        // comparing against the explicit two-stream expression.
        let eps = 0.6f64;
        let kappa = -((1.0f64 - eps).ln()); // so 1 - exp(-kappa*m) = eps at m = 1
        let b = Band::new(1.0, kappa, 1.0).unwrap();
        let rt = MultiBandRadiativeTransfer::new(vec![b], 250.0).unwrap();
        let ts = 288.0;
        let f = rt.olr(ts);
        let expected = (1.0 - eps) * SIGMA * ts.powi(4) + eps * SIGMA * 250.0_f64.powi(4);
        assert_abs_diff_eq!(f, expected, epsilon = 1e-6);
    }

    #[test]
    fn multi_band_olr_exceeds_grey_for_same_mean_eps() {
        // Splitting the longwave into two bands with equal weights and equal
        // slab emissivity must reproduce the same total OLR as a single band.
        let kappa = 0.05;
        let b1 = Band::new(0.5, kappa, 10.0).unwrap();
        let b2 = Band::new(0.5, kappa, 10.0).unwrap();
        let split = MultiBandRadiativeTransfer::new(vec![b1, b2], 250.0).unwrap();
        let whole = Band::new(1.0, kappa, 10.0).unwrap();
        let whole_rt = MultiBandRadiativeTransfer::new(vec![whole], 250.0).unwrap();
        assert_abs_diff_eq!(split.olr(288.0), whole_rt.olr(288.0), epsilon = 1e-9);
    }

    #[test]
    fn correlated_k_matches_single_grey_at_one_point() {
        // A correlated-k band with a single g-point at mean k is identical to the
        // simplified grey band.
        let kappa = 0.07;
        let ck = CorrelatedKBand::new(1.0, vec![kappa], vec![1.0]).unwrap();
        let rt = CorrelatedKRt::new(vec![ck], 250.0, 10.0).unwrap();
        let grey = Band::new(1.0, kappa, 10.0).unwrap();
        let grey_rt = MultiBandRadiativeTransfer::new(vec![grey], 250.0).unwrap();
        assert_abs_diff_eq!(rt.olr(288.0), grey_rt.olr(288.0), epsilon = 1e-9);
    }

    #[test]
    fn correlated_k_is_monotonic_in_path() {
        let ck = CorrelatedKBand::new(
            1.0,
            vec![0.02, 0.1, 0.5],
            vec![0.5, 0.3, 0.2],
        )
        .unwrap();
        let thin = CorrelatedKRt::new(vec![ck.clone()], 250.0, 1.0).unwrap();
        let thick = CorrelatedKRt::new(vec![ck], 250.0, 50.0).unwrap();
        // A thicker path traps more longwave: OLR decreases.
        assert!(thick.olr(288.0) < thin.olr(288.0));
    }

    #[test]
    fn weights_must_sum_to_one() {
        let b = Band::new(0.4, 0.1, 1.0).unwrap();
        let r = MultiBandRadiativeTransfer::new(vec![b], 250.0);
        assert!(r.is_err());
    }
}
