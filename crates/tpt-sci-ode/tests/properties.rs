//! Property-based checks for the ODE solver: linear systems scale with their
//! initial condition, and the solution of `y' = A y` is linear in `y0`.

use proptest::prelude::*;
use tpt_sci_ode::{Method, OdeProblemBuilder};

fn linear_solve(a: [[f64; 2]; 2], y0: [f64; 2], t: f64, method: Method) -> Option<[f64; 2]> {
    let prob = OdeProblemBuilder::new(
        move |_t, y: &[f64], dy: &mut [f64]| {
            dy[0] = a[0][0] * y[0] + a[0][1] * y[1];
            dy[1] = a[1][0] * y[0] + a[1][1] * y[1];
        },
        y0.to_vec(),
        0.0,
    )
    .rtol(1e-8)
    .atol(1e-10)
    .build()
    .ok()?;
    let y = prob.solve(method, t).ok()?;
    Some([y[0], y[1]])
}

proptest! {
    // Stable, non-stiff systems only: entries bounded so eigenvalues have |Re| <= ~2.
    #[test]
    fn solution_scales_with_initial_condition(
        a00 in -0.8f64..0.8, a01 in -0.8f64..0.8,
        a10 in -0.8f64..0.8, a11 in -0.8f64..0.8,
        y0 in prop::array::uniform2(-5.0f64..5.0),
        alpha in -3.0f64..3.0,
    ) {
        let a = [[a00, a01], [a10, a11]];
        let base = linear_solve(a, y0, 1.0, Method::Tsit45).unwrap();
        let scaled_y0 = [alpha * y0[0], alpha * y0[1]];
        let scaled = linear_solve(a, scaled_y0, 1.0, Method::Tsit45).unwrap();
        for i in 0..2 {
            let expected = alpha * base[i];
            prop_assert!((scaled[i] - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
                "component {i}: {} vs {}", scaled[i], expected);
        }
    }
}
