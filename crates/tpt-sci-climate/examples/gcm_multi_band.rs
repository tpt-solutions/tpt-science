//! Tour of the expanded climate capabilities: multi-band radiative transfer,
//! 3-D atmospheric tracer transport, and the primitive-equation GCM core
//! coupled to the energy-balance model.
//!
//! Run with: `cargo run --example gcm_multi_band -p tpt-sci-climate`

use tpt_sci_climate::{
    AtmosphereGcm, Band, ChemistryBox, EnergyBalanceModel, MultiBandRadiativeTransfer, Tracer3D,
};
use tpt_sci_grid::UniformGrid3D;

fn main() {
    println!("tpt-sci-climate: multi-band RT + 3-D tracer + GCM core\n");

    // --- 1. Multi-band longwave radiative transfer --------------------------
    // Two grey slabs (window band + opaque band) replacing the single grey band.
    let window = Band::new(0.25, 0.02, 8.0).expect("window band");
    let opaque = Band::new(0.75, 0.4, 8.0).expect("opaque band");
    let rt = MultiBandRadiativeTransfer::new(vec![window, opaque], 250.0).expect("RT");
    let olr = rt.olr(288.0);
    println!(
        "multi-band OLR = {olr:.1} W/m² (effective ε = {:.3})",
        rt.effective_emissivity()
    );
    println!("  surface downwelling LW = {:.1} W/m²", rt.downward_flux());

    // --- 2. 3-D tracer transport -------------------------------------------
    let grid = UniformGrid3D::new(11, 0.0, 1.0, 11, 0.0, 1.0, 9, 0.0, 1.0).unwrap();
    let n = grid.len();
    let mut conc = vec![0.0f64; n];
    conc[grid.index(5, 5, 4)] = 1.0; // point source
    let mut tr = Tracer3D::new(
        grid,
        conc,
        vec![0.5; n], // uniform zonal advection
        vec![0.0; n],
        vec![0.0; n],
        0.02,
        vec![0.0; n],
        0.1,
    )
    .expect("tracer");
    for _ in 0..20 {
        tr.step(0.05);
    }
    println!(
        "3-D tracer mean = {:.4} (total mass finite: {})",
        tr.mean_concentration(),
        tr.total_mass().is_finite()
    );

    // --- 3. GCM dynamical core coupled to the EBM ---------------------------
    let mut gcm =
        AtmosphereGcm::new(11, 5, 7, 100.0, 40.0, 60.0, 1.2, 1.0 / 300.0, 250.0, 9.81, 1e-4, 1.6e-11, 0.01)
            .expect("gcm");
    let mut ebm = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 280.0).unwrap();
    let teq = gcm.couple_to_ebm(&ebm);
    println!("GCM coupled to EBM: target T_eq = {teq:.1} K");
    for step in 0..10 {
        gcm.step(0.5);
        if step % 3 == 0 {
            println!(
                "  step {step}: mean T = {:.2} K, max wind = {:.3} m/s",
                gcm.mean_temperature(),
                gcm.max_wind()
            );
        }
    }
    // Doubling CO2 warms the EBM target; the GCM relaxes toward it.
    ebm.co2 = 560.0;
    gcm.couple_to_ebm(&ebm);
    println!("after CO2 doubling: EBM-driven target T_eq = {:.1} K", gcm.t_eq);

    // Single-tracer box sanity (existing API).
    let mut c = ChemistryBox::new(0.0, 1.0, 0.1).unwrap();
    for _ in 0..1000 {
        c.step(0.1);
    }
    println!("ChemistryBox steady state = {:.3} (target {:.3})", c.concentration, c.steady_state());
}
