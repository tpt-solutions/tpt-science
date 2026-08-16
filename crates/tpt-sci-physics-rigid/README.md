# tpt-sci-physics-rigid

A small, dependency-light **rigid-body (sphere) physics world** with analytic
collision resolution, implemented entirely from scratch (no wrapped physics
engine such as `rapier`, which is disqualified per ADR 0007 — Apache-2.0-only).

Bodies are spheres of arbitrary dimension (2-D or 3-D) described by `Body`,
simulated inside a `World` that supports constant gravity, axis-aligned
bounding walls, and pairwise elastic collisions with a configurable restitution
coefficient.

The integrator is a semi-implicit (symplectic) Euler scheme; collisions are
resolved with the standard two-sphere impulse plus a positional
(Baumgarte-style) correction to keep overlapping bodies from sinking into one
another.

Depends on `tpt-math-linalg` (published).

## Example

```rust
use tpt_sci_physics_rigid::{Body, World, PhysicsError};
use tpt_math_linalg::tpt_math_linalg_dense::DVector;

let mut world = World::new();
let a = Body::new(
    0,
    DVector::from_row_slice(&[0.0_f64, 0.0]),
    DVector::from_row_slice(&[1.0_f64, 0.0]),
    1.0,
    0.5,
)?;
world.add_body(a).unwrap();
let b = Body::new(
    1,
    DVector::from_row_slice(&[1.0_f64, 0.0]),
    DVector::from_row_slice(&[0.0_f64, 0.0]),
    1.0,
    0.5,
)?;
world.add_body(b).unwrap();

let p0 = world.body(0).unwrap().position.clone();
world.step(0.5);
let p1 = world.body(0).unwrap().position.clone();
assert!((p1[0] - p0[0]).abs() > 1e-9);
# Ok::<(), PhysicsError>(())
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
