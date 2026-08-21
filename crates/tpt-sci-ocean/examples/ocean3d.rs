//! Tour of the 3-D ocean core and data assimilation in `tpt-sci-ocean`.
//!
//! Demonstrates, on a small grid:
//!
//! 1. **3-D z-level core** — a warm surface anomaly drives a baroclinic flow.
//! 2. **Non-hydrostatic correction** — a divergent field is projected to be
//!    (interior) divergence-free via the 3-D pressure-Poisson solve.
//! 3. **Data assimilation** — nudging a perturbed state toward truth, and an
//!    ensemble Kalman filter reducing analysis error vs the background.
//!
//! Run with: `cargo run --example ocean3d -p tpt-sci-ocean`

use tpt_sci_ocean::{EnsembleKalmanFilter, Observation, Ocean3D, Var3D};

/// Build a small 3-D ocean with a Gaussian warm anomaly in the upper layers.
fn warm_core() -> Ocean3D {
    let mut o = Ocean3D::new(
        11, 3, 6, 100.0, 30.0, 50.0, 1025.0, 0.2, 0.8, 15.0, 35.0, 9.81, 0.0, 0.01,
    )
    .expect("valid ocean");
    let (nx, ny, nz) = (o.grid.nx(), o.grid.ny(), o.grid.nz());
    let cx = nx / 2;
    let cy = ny / 2;
    for iz in (nz - 2)..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let d2 = (ix as f64 - cx as f64).powi(2) + (iy as f64 - cy as f64).powi(2);
                let c = o.index(ix, iy, iz);
                o.t[c] += 4.0 * (-d2 / 4.0).exp();
            }
        }
    }
    o
}

fn main() {
    println!("tpt-sci-ocean: 3-D core + assimilation tour\n");

    // --- 1. Baroclinic flow from a warm anomaly -----------------------------
    let mut o = warm_core();
    let bottom = o.grid.nz() - 1;
    let cx = o.grid.nx() / 2;
    let cy = o.grid.ny() / 2;
    let u_before = o.u[o.index(cx - 2, cy, bottom)];
    for _ in 0..8 {
        o.step_3d(0.5);
    }
    let u_after = o.u[o.index(cx - 2, cy, bottom)];
    println!(
        "1. Warm anomaly: u (left of centre, bottom) {:.4e} -> {:.4e} m/s \
         (flow develops toward the anomaly).",
        u_before, u_after
    );
    assert!(u_after > 0.0, "expected convergence toward the warm core");

    // --- 2. Non-hydrostatic pressure correction -----------------------------
    let mut o2 = Ocean3D::new(
        9, 3, 5, 100.0, 30.0, 50.0, 1025.0, 0.2, 0.8, 15.0, 35.0, 9.81, 0.0, 0.01,
    )
    .expect("valid ocean");
    let (nx, ny, nz) = (o2.grid.nx(), o2.grid.ny(), o2.grid.nz());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let c = o2.index(ix, iy, iz);
                o2.u[c] = ix as f64 * 0.01; // du/dx = 0.01 everywhere (divergent)
            }
        }
    }
    let div_before = o2.max_divergence();
    o2.nonhydrostatic_correct(1.0, 1e-10);
    let div_after = o2.max_divergence();
    println!(
        "2. Non-hydrostatic projection: max interior divergence {:.3e} -> {:.3e}.",
        div_before, div_after
    );
    assert!(
        div_after < div_before * 0.05,
        "projection should remove divergence"
    );

    // --- 3a. Nudging toward sparse observations -----------------------------
    let truth = [1.0, 2.0, 3.0, 4.0, 5.0];
    let state = [2.0, 2.0, 3.0, 4.0, 5.0];
    let obs = [Observation {
        index: 0,
        value: truth[0],
        weight: 1.0,
    }];
    let corrected = tpt_sci_ocean::nudge(&state, &obs, 0.5, 1.0);
    println!(
        "3a. Nudging: index 0 {:.3} -> {:.3} toward truth {:.3}.",
        state[0], corrected[0], truth[0]
    );
    assert!((corrected[0] - truth[0]).abs() < (state[0] - truth[0]).abs());

    // --- 3b. Ensemble Kalman filter on a toy linear operator ----------------
    let n = 6;
    let p = 6;
    let truth_vec: Vec<f64> = (1..=n).map(|x| x as f64).collect();
    let h: Vec<Vec<f64>> = (0..p)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut rng = 0x9E3779B9u32;
    let next_f64 = |seed: &mut u32| -> f64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *seed = x;
        (x as f64) / ((u32::MAX as f64) + 1.0)
    };
    let mut ensemble = Vec::new();
    for _ in 0..40 {
        let member: Vec<f64> = (0..n)
            .map(|i| truth_vec[i] + 0.5 + 0.3 * (next_f64(&mut rng) - 0.5))
            .collect();
        ensemble.push(member);
    }
    let r_diag = vec![1e-6; p];
    let m = ensemble.len();
    let bg_mean: Vec<f64> = (0..n)
        .map(|i| ensemble.iter().map(|e| e[i]).sum::<f64>() / m as f64)
        .collect();
    let bg_err = bg_mean
        .iter()
        .zip(&truth_vec)
        .map(|(a, t)| (a - t).powi(2))
        .sum::<f64>()
        .sqrt();
    let enkf = EnsembleKalmanFilter::new(ensemble, r_diag).expect("valid ensemble");
    let result = enkf.analyze(&truth_vec, &h).expect("analysis succeeds");
    let an_err = result
        .mean
        .iter()
        .zip(&truth_vec)
        .map(|(a, t)| (a - t).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "3b. EnKF: background RMSE {:.3e} -> analysis RMSE {:.3e}.",
        bg_err, an_err
    );
    assert!(an_err < bg_err, "EnKF analysis should beat the background");

    // --- 3c. 3D-Var-lite -----------------------------------------------------
    let xb: Vec<f64> = truth_vec.iter().map(|v| v + 0.5).collect();
    let bg_err_v = xb
        .iter()
        .zip(&truth_vec)
        .map(|(a, t)| (a - t).powi(2))
        .sum::<f64>()
        .sqrt();
    let var = Var3D::new(xb, vec![0.2; n], vec![1e-6; p]).expect("valid 3D-Var");
    let xa = var.analyze(&truth_vec, &h).expect("analysis succeeds");
    let an_err_v = xa
        .iter()
        .zip(&truth_vec)
        .map(|(a, t)| (a - t).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "3c. 3D-Var-lite: background RMSE {:.3e} -> analysis RMSE {:.3e}.",
        bg_err_v, an_err_v
    );
    assert!(
        an_err_v < bg_err_v,
        "3D-Var analysis should beat the background"
    );

    println!("\nAll tour assertions passed.");
}
