# tpt-sci-ode

ODE/DAE solving substrate for the `tpt-science` pillar.

This crate is a thin, ergonomic wrapper over
[`diffsol`](https://crates.io/crates/diffsol) (JOSS 2026, Enzyme-based
autodiff, LLVM/Cranelift JIT). The ecosystem gap is well-covered by a
maintained, dual-licensed crate, so we wrap rather than reimplement. Every
crate in this pillar is dual-licensed `MIT OR Apache-2.0`.

Depends on `tpt-math-numeric` (published, `tpt-math` repo).

## Example

```rust
use tpt_sci_ode::{OdeProblem, Method};

// dy/dt = -y,  y(0) = 1  ->  y(1) = e^-1
let prob = OdeProblem::new(
    |_t, y, dydt| { dydt[0] = -y[0]; },
    vec![1.0],
    0.0,
).unwrap();
let y = prob.solve(Method::Bdf, 1.0).unwrap();
assert!((y[0] - std::f64::consts::E.recip()).abs() < 1e-5);
```

## License

Licensed under either of [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) at your option.
