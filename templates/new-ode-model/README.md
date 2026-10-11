# {{project-name}}

An ODE model built on `tpt-sci-ode`, generated from the tpt-science
`new-ode-model` template.

```sh
cargo run --release
```

The starter solves `y' = -k·y`, prints the trajectory as CSV, and checks the
result against the closed form. Replace the closure in `src/main.rs` with your
model.
