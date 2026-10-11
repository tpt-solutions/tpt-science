//! Langmuir–Hinshelwood surface coverages, single-site and multi-site.

use super::*;

/// Compute Langmuir–Hinshelwood fractional surface coverages for a single site
/// type given per-species adsorption equilibrium constants `K_i` and
/// (partial) pressures `p_i`.
///
/// The coverages satisfy `θ_i = K_i·p_i / (1 + Σ_j K_j·p_j)` and sum to ≤ 1
/// (the remaining fraction is bare surface). Multi-component equilibria with
/// shared sites are handled by the standard single-site isotherm.
///
/// # Errors
///
/// Returns [`KineticsError::CoverageError`] if any `K_i` or `p_i` is negative or
/// non-finite, or the denominator is non-positive.
pub fn langmuir_hinshelwood_coverages(
    ks: &[f64],
    pressures: &[f64],
) -> Result<Vec<f64>, KineticsError> {
    if ks.len() != pressures.len() {
        return Err(KineticsError::CoverageError(format!(
            "K/pressure length mismatch ({} vs {})",
            ks.len(),
            pressures.len()
        )));
    }
    let mut denom = 1.0;
    for (&k, &p) in ks.iter().zip(pressures.iter()) {
        if !k.is_finite() || k < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative adsorption constant {k}"
            )));
        }
        if !p.is_finite() || p < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative pressure {p}"
            )));
        }
        denom += k * p;
    }
    if denom <= 0.0 {
        return Err(KineticsError::CoverageError(
            "non-positive site denominator".into(),
        ));
    }
    Ok(ks
        .iter()
        .zip(pressures.iter())
        .map(|(&k, &p)| k * p / denom)
        .collect())
}

/// Compute Langmuir–Hinshelwood fractional surface coverages for **multiple
/// distinct site types**, e.g. species that adsorb on site type A vs species
/// that adsorb on an independent site type B (each with its own site-balance
/// normalization).
///
/// `ks` and `pressures` are per-species, as in [`langmuir_hinshelwood_coverages`].
/// `site_of` assigns each species (by index) to a site-type index; species
/// sharing a site-type index compete for the *same* pool of sites and their
/// coverages are normalized together, while species on different site types
/// are normalized independently. This reduces to
/// [`langmuir_hinshelwood_coverages`] when every species is assigned to the
/// same (single) site type.
///
/// The returned vector is aligned with `ks`/`pressures`/`site_of` (one
/// coverage per species), and within each site type the coverages sum to ≤ 1.
///
/// # Errors
///
/// Returns [`KineticsError::CoverageError`] if the input lengths mismatch, any
/// `K_i`/`p_i` is negative or non-finite, or any per-site-type denominator is
/// non-positive.
///
/// # Example
///
/// ```
/// use tpt_sci_kinetics::multi_site_langmuir_hinshelwood_coverages;
///
/// // Species 0, 1 adsorb on site type 0; species 2 adsorbs on site type 1.
/// let theta = multi_site_langmuir_hinshelwood_coverages(
///     &[2.0, 3.0, 5.0],
///     &[1.0, 1.0, 1.0],
///     &[0, 0, 1],
/// )
/// .unwrap();
/// // Site type 0 coverages sum with the bare fraction to 1, independent of site 1.
/// assert!((theta[0] + theta[1]) <= 1.0 + 1e-9);
/// assert!(theta[2] <= 1.0 + 1e-9);
/// ```
pub fn multi_site_langmuir_hinshelwood_coverages(
    ks: &[f64],
    pressures: &[f64],
    site_of: &[usize],
) -> Result<Vec<f64>, KineticsError> {
    if ks.len() != pressures.len() || ks.len() != site_of.len() {
        return Err(KineticsError::CoverageError(format!(
            "K/pressure/site-type length mismatch ({} vs {} vs {})",
            ks.len(),
            pressures.len(),
            site_of.len()
        )));
    }
    for (&k, &p) in ks.iter().zip(pressures.iter()) {
        if !k.is_finite() || k < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative adsorption constant {k}"
            )));
        }
        if !p.is_finite() || p < 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-finite/negative pressure {p}"
            )));
        }
    }
    let n_sites = site_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut denoms = vec![1.0_f64; n_sites];
    for ((&k, &p), &site) in ks.iter().zip(pressures.iter()).zip(site_of.iter()) {
        denoms[site] += k * p;
    }
    for (site, &denom) in denoms.iter().enumerate() {
        if denom <= 0.0 {
            return Err(KineticsError::CoverageError(format!(
                "non-positive site denominator for site type {site}"
            )));
        }
    }
    Ok(ks
        .iter()
        .zip(pressures.iter())
        .zip(site_of.iter())
        .map(|((&k, &p), &site)| k * p / denoms[site])
        .collect())
}
