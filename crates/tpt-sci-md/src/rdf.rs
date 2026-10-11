//! Radial distribution function for structural analysis.

use super::*;

/// Compute the radial distribution function `g(r)` of a configuration of
/// particles in a cubic periodic box.
///
/// `r_max` is the largest sampled radius (≤ `box_len/2`), `nbins` the histogram
/// resolution. Returns `(r_centers, g)` where `g` is normalized by the ideal-gas
/// shell volume (so `g → 1` in the dilute limit).
///
/// # Errors
///
/// Returns [`MdError::RdfError`] if fewer than two particles, `nbins == 0`,
/// `r_max <= 0`, or `r_max > box_len/2` (for a periodic box).
pub fn rdf(
    particles: &[Particle],
    box_len: f64,
    r_max: f64,
    nbins: usize,
) -> Result<(Vec<f64>, Vec<f64>), MdError> {
    let n = particles.len();
    if n < 2 {
        return Err(MdError::RdfError("need at least two particles".into()));
    }
    if nbins == 0 {
        return Err(MdError::RdfError("nbins must be > 0".into()));
    }
    if r_max <= 0.0 {
        return Err(MdError::RdfError("r_max must be > 0".into()));
    }
    if box_len.is_finite() && r_max > box_len * 0.5 {
        return Err(MdError::RdfError("r_max must be <= box_len/2".into()));
    }
    let dim = particles.first().map_or(3, |p| p.position.len());
    let dr = r_max / nbins as f64;
    let mut hist = vec![0_usize; nbins];

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

    for i in 0..n {
        for j in (i + 1)..n {
            let raw = particles[j].position.clone() - particles[i].position.clone();
            let d = DVector::from_fn(dim, |k| wrap(raw[k]));
            let r = d.norm();
            if r < r_max {
                let bin = (r / dr) as usize;
                if bin < nbins {
                    hist[bin] += 1;
                }
            }
        }
    }

    let rho = n as f64
        / if box_len.is_finite() {
            box_len.powi(dim as i32)
        } else {
            1.0
        };
    let volume = if box_len.is_finite() {
        box_len.powi(dim as i32)
    } else {
        1.0
    };
    let centers: Vec<f64> = (0..nbins).map(|b| (b as f64 + 0.5) * dr).collect();
    let g: Vec<f64> = (0..nbins)
        .map(|b| {
            let r = centers[b];
            // Shell volume in `dim` dimensions (spherical).
            let shell = if dim == 3 {
                4.0 * std::f64::consts::PI * r * r * dr
            } else if dim == 2 {
                2.0 * std::f64::consts::PI * r * dr
            } else {
                2.0 * r * dr
            };
            let expected = rho * shell * (n - 1) as f64 / volume * volume;
            if expected == 0.0 {
                0.0
            } else {
                hist[b] as f64 / expected * volume
            }
        })
        .collect();
    Ok((centers, g))
}
