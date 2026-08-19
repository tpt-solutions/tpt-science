//! Ocean demo: a Gaussian sea-surface bump spreads as gravity waves across a
//! Coriolis-affected basin, solved with the 2-D shallow-water equations.
//!
//! Run with: `cargo run --example shallow_water -p tpt-sci-ocean`

use tpt_sci_ocean::ShallowWater;

fn main() {
    let mut sw = ShallowWater::new(96, 96, 1.0, 1.0, 9.81, 1e-4, 0.002);
    sw.perturb_center(1.0);
    let n = 600;
    for k in 0..n {
        sw.step(0.002);
        if k % 150 == 0 {
            println!("step {k}: max speed = {:.4} m/s", sw.max_speed());
        }
    }
    println!("Shallow-water circulation completed {n} steps.");
}
