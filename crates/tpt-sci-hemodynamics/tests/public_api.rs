//! Public-API invariants for vessel mechanics and the Windkessel outflow.

use tpt_sci_hemodynamics::{Vessel, Windkessel, tube_law_beta};

#[test]
fn stiffer_wall_gives_faster_pressure_waves() {
    let soft = Vessel::new(1.0, 0.0, 1.0, tube_law_beta(1.0e4, 1.0, 1.0)).unwrap();
    let stiff = Vessel::new(1.0, 0.0, 1.0, tube_law_beta(1.0e6, 1.0, 1.0)).unwrap();
    assert!(stiff.wave_speed(1.0) > soft.wave_speed(1.0));
}

#[test]
fn windkessel_decays_to_venous_pressure_with_no_inflow() {
    let mut wk = Windkessel::new(0.5, 1.0, 1.0, 5.0, 20.0).unwrap();
    let tau = wk.time_constant();
    let dt = tau / 200.0;
    for _ in 0..(6 * 200) {
        wk.step(0.0, dt);
    }
    let p = wk.proximal_pressure(0.0);
    assert!(
        (p - 5.0).abs() < 0.05 * 20.0,
        "pressure {p} did not relax to venous 5.0"
    );
}
