//! Public-API invariants for the incompressible Navier–Stokes solver.

use tpt_sci_cfd_core::{Boundary, CollocatedGrid, Step};

#[test]
fn fluid_at_rest_stays_at_rest() {
    let grid = CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap();
    let mut step = Step::new(grid, 1e-2, 0.01, 1.0);
    for _ in 0..20 {
        assert!(step.advance());
    }
    assert_eq!(step.max_divergence(), 0.0);
}

#[test]
fn lid_driven_flow_is_divergence_free_after_projection() {
    let grid = CollocatedGrid::new(16, 16, 1.0, 1.0).unwrap();
    let mut step = Step::new(grid, 1e-2, 0.01, 1.0);
    step.set_boundary(Boundary::Top, 1.0);
    for _ in 0..50 {
        assert!(step.advance());
    }
    assert!(
        step.max_divergence() < 1e-8,
        "div = {}",
        step.max_divergence()
    );
}

#[test]
fn eddy_viscosity_rejects_invalid_inputs() {
    use tpt_sci_cfd_core::CfdError;
    use tpt_sci_cfd_core::turbulence::eddy_viscosity;
    assert!(eddy_viscosity(1e-3, 0.1, 0.05, 2.0).is_ok());
    assert!(matches!(
        eddy_viscosity(-1.0, 0.1, 0.05, 2.0),
        Err(CfdError::InvalidParameter(_))
    ));
    assert!(matches!(
        eddy_viscosity(1e-3, 0.1, 0.0, 2.0),
        Err(CfdError::InvalidParameter(_))
    ));
    assert!(eddy_viscosity(f64::NAN, 0.1, 0.05, 2.0).is_err());
}
