//! Electronic-structure demo: a 1-D Kohn–Sham solve of a two-electron system in
//! a harmonic confining well, reporting total energy and self-consistent density.
//!
//! Run with: `cargo run --example harmonic_atom -p tpt-sci-dft-electronic`

use tpt_sci_dft_electronic::{Grid1D, KohnSham};

fn main() {
    let grid = Grid1D::new(121, -6.0, 6.0).unwrap();
    let xs = grid.x().to_vec();
    let v_ext: Vec<f64> = xs.iter().map(|&x| 0.5 * x * x).collect();
    let mut ks = KohnSham::new(grid.clone(), v_ext, 2).unwrap();

    let res = ks.solve(80);
    println!("total energy      = {:.4} Ha", res.total_energy);
    println!("n points          = {}", res.density.len());

    // Peak density location (should sit near the well center x = 0).
    let peak = res
        .density
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    if let Some((i, _)) = peak {
        println!("peak density at index {i} (x ≈ {:.2})", xs[i]);
    }
    println!("1-D Kohn-Sham solve complete.");
}
