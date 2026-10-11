//! Public-API invariants for the 0-D energy-balance model.

use tpt_sci_climate::EnergyBalanceModel;

#[test]
fn time_march_converges_to_equilibrium() {
    let mut ebm = EnergyBalanceModel::new(1.0, 0.3, 0.61, 280.0).unwrap();
    let t_eq = ebm.equilibrium_temperature();
    for _ in 0..5000 {
        ebm.step(1.0);
    }
    assert!(
        (ebm.temperature() - t_eq).abs() < 1e-3,
        "{} vs {}",
        ebm.temperature(),
        t_eq
    );
}

#[test]
fn doubling_co2_warms_the_planet() {
    let mut ebm = EnergyBalanceModel::new(1.0, 0.3, 0.61, 280.0).unwrap();
    let t0 = ebm.equilibrium_temperature();
    ebm.co2 = 560.0;
    let t2 = ebm.equilibrium_temperature();
    assert!(t2 > t0 + 1.0, "warming {} K too small", t2 - t0);
}

#[test]
fn large_time_step_stays_stable() {
    // With C = 1 J·m⁻²·K⁻¹ the radiative relaxation time is ~0.3 s, far below
    // dt = 1000 s. Explicit Euler oscillates and diverges here; the step must not.
    let mut ebm = EnergyBalanceModel::new(1.0, 0.3, 0.61, 280.0).unwrap();
    let t_eq = ebm.equilibrium_temperature();
    for _ in 0..50 {
        ebm.step(1000.0);
        assert!(ebm.temperature().is_finite());
    }
    assert!(
        (ebm.temperature() - t_eq).abs() < 1e-6,
        "{} vs {t_eq}",
        ebm.temperature()
    );
}

#[test]
fn grey_radiative_transfer_rejects_out_of_range_inputs() {
    use tpt_sci_climate::{ClimateError, grey_radiative_transfer};
    assert!(grey_radiative_transfer(1361.0, 0.3, 0.5).is_ok());
    assert!(matches!(
        grey_radiative_transfer(0.0, 0.3, 0.5),
        Err(ClimateError::InvalidModel(_))
    ));
    assert!(matches!(
        grey_radiative_transfer(1361.0, 1.5, 0.5),
        Err(ClimateError::InvalidModel(_))
    ));
    assert!(matches!(
        grey_radiative_transfer(1361.0, 0.3, 1.0),
        Err(ClimateError::InvalidModel(_))
    ));
}
