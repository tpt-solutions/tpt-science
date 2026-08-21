//! # 1-D hydrogen-like atom (`tpt-sci-dft-electronic`)
//!
//! A second, complementary tour of the crate's 1-D Kohn–Sham LDA surface, this
//! time exercising **Coulomb-confined** model atoms rather than the harmonic
//! confining well of `harmonic_atom`.
//!
//! What this example demonstrates:
//!
//! * A **softened Coulomb (hydrogen-like) external potential**
//!   `V_ext(x) = −Z / sqrt(x² + a²)` — the 1-D analog of a nucleus of charge `Z`.
//!   Because the demo solver fills *spin-paired* orbitals (2 electrons per
//!   level), the simplest bound Coulomb system it can represent is the
//!   **isoelectronic series** (`Z = 1, 2, 3, 4`) in the two-electron (He-like)
//!   state: we show the ground-state energy becomes progressively more bound
//!   (more negative) as `Z` grows, the orbital contracts toward the nucleus, and
//!   the density stays normalized to the electron count.
//! * A **finite square well** `-V0 (|x| < L)` as a contrasting bound system with
//!   compact support, confirming the same solver yields a localized, nodeless
//!   ground orbital and a normalised density.
//! * Real assertions: occupied orbital is nodeless (ground state), density
//!   integrates to the 2 electrons, and the Coulomb ground-state energy is a
//!   decreasing function of nuclear charge. A small grid-resolution sweep
//!   (varying `n`) confirms density normalization and finite energy hold.
//!
//! Run with:
//! `cargo run --example hydrogen_atom -p tpt-sci-dft-electronic`
//!
//! Observe: the softened-Coulomb series binds more tightly with `Z`, the ground
//! orbital is nodeless and peaks at the nucleus, and the finite well produces a
//! confined bound state — distinct physical scenarios from the harmonic well.

use tpt_sci_dft_electronic::{DftError, Grid1D, KohnSham, KohnShamResult, lda_xc};

/// Softened 1-D Coulomb well `V(x) = -Z / sqrt(x² + a²)` — a hydrogen-like
/// nucleus of charge `Z` (softened so the `x = 0` singularity is regularised).
fn coulomb_well(z: f64, soft: f64, x: f64) -> f64 {
    -z / (x * x + soft * soft).sqrt()
}

/// Finite square well: `-v0` inside `|x| < half`, zero outside.
fn square_well(v0: f64, half: f64, x: f64) -> f64 {
    if x.abs() < half {
        -v0
    } else {
        0.0
    }
}

/// Solve the 1-D Kohn–Sham equations for an external potential `v_ext` on a grid
/// of `n` points, returning the grid and the raw [`KohnShamResult`].
fn solve(
    n: usize,
    xmin: f64,
    xmax: f64,
    v_ext: Vec<f64>,
    nelect: usize,
    max_iter: usize,
) -> (Grid1D, KohnShamResult) {
    let grid = Grid1D::new(n, xmin, xmax).unwrap();
    let mut ks = KohnSham::new(grid.clone(), v_ext, nelect).unwrap();
    let res = ks.solve(max_iter);
    (grid, res)
}

/// Count sign changes of an orbital coefficient vector (0 = nodeless ground
/// state).
fn sign_changes(orb: &[f64]) -> usize {
    orb.windows(2).filter(|w| w[0] * w[1] < 0.0).count()
}

/// Peak position (x) of the self-consistent density.
fn density_peak_x(grid: &Grid1D, density: &[f64]) -> f64 {
    let i = density
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    grid.xs[i]
}

fn main() {
    println!("== tpt-sci-dft-electronic: 1-D hydrogen-like atom ==");

    // --- 0. Grid / setup error handling (public DftError surface) ----------
    match Grid1D::new(1, 0.0, 1.0) {
        Err(DftError::InvalidGrid(_)) => println!("grid error path OK: InvalidGrid raised"),
        other => panic!("expected InvalidGrid, got {other:?}"),
    }
    let probe_grid = Grid1D::new(81, -8.0, 8.0).unwrap();
    let probe_v: Vec<f64> = probe_grid
        .x()
        .iter()
        .copied()
        .map(|x| coulomb_well(1.0, 0.5, x))
        .collect();
    match KohnSham::new(probe_grid.clone(), probe_v.clone(), 0) {
        Err(DftError::InvalidSetup(_)) => println!("setup error path OK: InvalidSetup (0 e⁻)"),
        other => panic!("expected InvalidSetup, got {other:?}"),
    }

    // LDA XC sanity: e_xc must stay attractive (<= 0) for the hydrogen density.
    assert!(lda_xc(0.5) <= 0.0, "LDA e_xc must be attractive");
    println!("LDA e_xc(0.5) = {:.4} Ha (attractive)\n", lda_xc(0.5));

    // --- 1. Coulomb-confined isoelectronic series (vary nuclear charge Z) ----
    println!("1. Softened-Coulomb isoelectronic series (2 e⁻, soft = 0.5)");
    println!("   Z    E_tot (Ha)    ∫ρ dx    peak-x    orbital nodes");
    let mut energies: Vec<f64> = Vec::new();
    for &z in &[1.0_f64, 2.0, 3.0, 4.0] {
        let n = 161;
        let grid = Grid1D::new(n, -8.0, 8.0).unwrap();
        let v: Vec<f64> = grid
            .x()
            .iter()
            .copied()
            .map(|x| coulomb_well(z, 0.5, x))
            .collect();
        let (g, res) = solve(n, -8.0, 8.0, v, 2, 80);

        assert!(res.total_energy.is_finite(), "total energy must be finite");
        assert_eq!(res.orbitals.len(), 1, "2 e⁻ occupy one spin-paired level");

        let ne: f64 = res.density.iter().sum::<f64>() * g.dx;
        let nodes = sign_changes(&res.orbitals[0]);
        let px = density_peak_x(&g, &res.density);
        println!(
            "   {z:>3.0}  {e:>10.4}   {ne:>6.3}   {px:>6.2}      {nodes}",
            e = res.total_energy
        );

        // Self-checks: normalized density, nodeless ground orbital, nucleus-localised.
        assert!((ne - 2.0).abs() < 0.3, "density integrates to 2 e⁻");
        assert_eq!(nodes, 0, "ground orbital must be nodeless");
        assert!(px.abs() < 1.5, "density localises near the nucleus (x≈0)");

        energies.push(res.total_energy);
    }
    // Stronger nuclear charge binds the electrons more tightly: the ground-state
    // energy must decrease (become more negative) as Z increases.
    println!(
        "   trend: E(Z=1) = {:.4}, E(Z=4) = {:.4} Ha",
        energies[0], energies[3]
    );
    assert!(
        energies[3] < energies[0],
        "ground-state energy must become more bound as Z increases"
    );

    // --- 2. Finite square well (compact-support bound state) ----------------
    println!("\n2. Finite square well (-V0 inside |x|<L)");
    let n = 161;
    let grid = Grid1D::new(n, -6.0, 6.0).unwrap();
    let v: Vec<f64> = grid
        .x()
        .iter()
        .copied()
        .map(|x| square_well(2.0, 2.0, x))
        .collect();
    let (g, res) = solve(n, -6.0, 6.0, v, 2, 80);
    let ne: f64 = res.density.iter().sum::<f64>() * g.dx;
    let nodes = sign_changes(&res.orbitals[0]);
    let px = density_peak_x(&g, &res.density);
    println!(
        "   E_tot = {:.4} Ha, ∫ρ dx = {ne:.3}, peak-x = {px:.2}, orbital nodes = {nodes}",
        res.total_energy
    );
    assert!(res.total_energy.is_finite());
    assert!((ne - 2.0).abs() < 0.3, "well density integrates to 2 e⁻");
    assert_eq!(nodes, 0, "ground orbital nodeless");
    assert!(px.abs() < 1.0, "well density peaks at centre");

    // --- 3. Grid-resolution sweep on the Coulomb Z=1 case -------------------
    println!("\n3. Resolution sweep (Coulomb Z=1):");
    for &n in &[81_usize, 161, 241] {
        let grid = Grid1D::new(n, -8.0, 8.0).unwrap();
        let v: Vec<f64> = grid
            .x()
            .iter()
            .copied()
            .map(|x| coulomb_well(1.0, 0.5, x))
            .collect();
        let (g, res) = solve(n, -8.0, 8.0, v, 2, 70);
        let ne: f64 = res.density.iter().sum::<f64>() * g.dx;
        println!(
            "   n={n:>3}: E_tot = {:>10.4} Ha, ∫ρ dx = {ne:.3}",
            res.total_energy
        );
        assert!(res.total_energy.is_finite());
        assert!((ne - 2.0).abs() < 0.3, "density stays normalized across grids");
    }

    println!("\n1-D hydrogen-like atom tour complete.");
}
