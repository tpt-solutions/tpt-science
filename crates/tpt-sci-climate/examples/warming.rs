//! Climate demo: a 0-D energy-balance model warms when CO2 is doubled, relaxing
//! to a new equilibrium temperature.
//!
//! Run with: `cargo run --example warming -p tpt-sci-climate`

use tpt_sci_climate::EnergyBalanceModel;

fn main() {
    let mut ebm = EnergyBalanceModel::new(1.0e7, 0.3, 0.61, 280.0).unwrap();
    let t0 = ebm.equilibrium_temperature();
    println!("pre-industrial equilibrium T = {t0:.2} K");

    ebm.co2 = 560.0; // doubled
    let t2 = ebm.equilibrium_temperature();
    println!("2xCO2 equilibrium T         = {t2:.2} K");
    println!("equilibrium warming         = {:.2} K", t2 - t0);

    // Time-march from pre-industrial toward the new equilibrium.
    for _ in 0..5000 {
        ebm.step(1.0);
    }
    println!("relaxed T after doubling     = {:.2} K", ebm.temperature());
}
