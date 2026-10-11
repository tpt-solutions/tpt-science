//! Event detection against closed-form crossing times, for every method.

use std::f64::consts::PI;

use tpt_sci_ode::events::{Direction, Event};
use tpt_sci_ode::{Method, OdeProblem, OdeProblemBuilder};

const METHODS: [Method; 4] = [
    Method::Tsit45,
    Method::TrBdf2,
    Method::Esdirk34,
    Method::Bdf,
];

/// `y'' = -y` as a first-order system `[y, v]`; the exact solution is
/// `y = cos t`, so `y` crosses zero (falling at π/2, rising at 3π/2, …).
fn oscillator() -> OdeProblem {
    OdeProblemBuilder::new(
        |_t, y, dy| {
            dy[0] = y[1];
            dy[1] = -y[0];
        },
        vec![1.0, 0.0],
        0.0,
    )
    .rtol(1e-9)
    .atol(1e-11)
    .build()
    .unwrap()
}

#[test]
fn non_terminal_events_count_every_zero_crossing() {
    // Zeros of cos t in (0, 4π): π/2, 3π/2, 5π/2, 7π/2 — two falling, two rising.
    let zero = Event::new(|_t, y| y[0]);
    for method in METHODS {
        let sol = oscillator()
            .solve_events(method, 4.0 * PI, std::slice::from_ref(&zero))
            .unwrap();
        assert_eq!(sol.hits.len(), 4, "{method:?}: wrong crossing count");
        assert!((sol.t_end - 4.0 * PI).abs() < 1e-12, "{method:?}");
        for (k, hit) in sol.hits.iter().enumerate() {
            let expected = PI / 2.0 + k as f64 * PI;
            // Crossing times inherit the per-step Hermite interpolation error
            // (O(h⁴)), so they are accurate to ~1e-5 here, not to round-off.
            assert!(
                (hit.t - expected).abs() < 1e-5,
                "{method:?}: hit {k} at {} expected {expected}",
                hit.t
            );
            let want = if k % 2 == 0 {
                Direction::Falling
            } else {
                Direction::Rising
            };
            assert_eq!(hit.direction, want, "{method:?}: hit {k}");
        }
    }
}

#[test]
fn direction_filter_keeps_only_matching_crossings() {
    let falling = Event::new(|_t, y| y[0]).direction(Direction::Falling);
    let rising = Event::new(|_t, y| y[0]).direction(Direction::Rising);
    let p = oscillator();
    let f = p
        .solve_events(Method::Tsit45, 4.0 * PI, &[falling])
        .unwrap();
    let r = p.solve_events(Method::Tsit45, 4.0 * PI, &[rising]).unwrap();
    assert_eq!(f.hits.len(), 2);
    assert_eq!(r.hits.len(), 2);
    assert!(f.hits.iter().all(|h| h.direction == Direction::Falling));
    assert!(r.hits.iter().all(|h| h.direction == Direction::Rising));
}

#[test]
fn terminal_event_stops_integration_at_crossing() {
    let stop = Event::new(|_t, y| y[0])
        .direction(Direction::Falling)
        .terminal();
    for method in METHODS {
        let sol = oscillator()
            .solve_events(method, 10.0, std::slice::from_ref(&stop))
            .unwrap();
        assert_eq!(sol.hits.len(), 1, "{method:?}");
        assert!(
            (sol.t_end - PI / 2.0).abs() < 1e-6,
            "{method:?}: t_end {}",
            sol.t_end
        );
        // y(t_end) is the root, so the state there is ~ [0, -1].
        assert!(sol.y_end[0].abs() < 1e-6, "{method:?}");
        assert!((sol.y_end[1] + 1.0).abs() < 1e-5, "{method:?}");
    }
}

#[test]
fn events_that_never_fire_leave_the_plain_solution_unchanged() {
    let never = Event::new(|_t, y| y[0] - 5.0); // |cos t| <= 1 never reaches 5
    let p = oscillator();
    let with = p.solve_events(Method::Tsit45, 3.0, &[never]).unwrap();
    let plain = p.solve(Method::Tsit45, 3.0).unwrap();
    assert!(with.hits.is_empty());
    assert_eq!(with.y_end, plain);
}
