//! Public-API invariants for the 2-D shallow-water solver.

use tpt_sci_ocean::ShallowWater;

#[test]
fn lake_at_rest_stays_at_rest() {
    let mut sw = ShallowWater::new(32, 32, 1.0, 1.0, 9.81, 1e-4, 0.001);
    sw.perturb_center(0.0);
    for _ in 0..20 {
        sw.step(0.001);
    }
    assert_eq!(sw.max_speed(), 0.0);
}

#[test]
fn perturbed_surface_evolves_boundedly() {
    let mut sw = ShallowWater::new(32, 32, 1.0, 1.0, 9.81, 1e-4, 0.001);
    sw.perturb_center(1.0);
    for _ in 0..50 {
        sw.step(0.001);
    }
    let speed = sw.max_speed();
    assert!(
        speed.is_finite() && speed > 0.0 && speed < 10.0,
        "speed {speed}"
    );
}
