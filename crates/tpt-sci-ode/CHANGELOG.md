# Changelog

All notable changes to `tpt-sci-ode` are documented here. This project adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Event detection: `Event`, `Direction`, `EventHit` and `EventSolution`, with
  `solve_events`. Terminal events stop the integration.
- `Trajectory` with CSV output, returned by `solve_trajectory`.
- `JacobianPattern` (sparsity detection, dense conversion and column colouring)
  with `jacobian_colored`. `OdeProblemBuilder::jacobian_pattern` feeds the
  pattern into the sparse Newton paths.
- Optional `serde` feature for result types.

### Fixed

- The crate compiles again. The Cranelift crates (`cranelift-codegen`,
  `-frontend`, `-module`, `-jit`) are all on 0.136.
- TR-BDF2 had inconsistent stage row sums and quadrature weights, so it
  converged to the wrong answer. It now uses the standard ESDIRK form.
- ESDIRK34 weights are the solution of the order-3 conditions, not hand-adjusted
  values.
- `JitRhs` owns the Cranelift module that holds its machine code. Previously
  `compile_rhs` dropped the module before returning, which left a dangling
  function pointer.
- `JitRhs::call` checks buffer lengths in release builds as well, and a panic
  in the closure is re-raised on the caller's side instead of unwinding through
  native frames.
- RHS errors are returned as `OdeError` from every method. Previously they
  panicked on the explicit, dense-implicit and sparse-implicit paths.
- A Newton solve that exhausts its iteration budget is an error. Previously an
  unconverged SDIRK stage was accepted silently.
- Newton convergence is tested on the tolerance-weighted norm of the correction,
  not on absolute thresholds, so large-magnitude states converge.
- Steps are accepted only when they meet the error tolerance. At the minimum
  step size the solver now returns `StepTooSmall` instead of accepting an
  inaccurate step.
- The step-size controller uses the local error order (`q`) consistently. The
  table had the order `p` for three methods, which made the controller grow
  steps too aggressively.
- The TR-BDF2 error estimate compares against the full-step trapezoidal rule,
  which is `O(h³)` locally. It previously used a stage value that was `O(h²)`.
- `t_eval` must be finite and strictly increasing; out-of-order points are
  rejected.
- `OdeProblemBuilder::h0` is honoured. Previously the value was stored and
  ignored.
- The sparse Newton path uses a sparse left-looking LU (Gilbert–Peierls with
  partial pivoting). Previously it allocated a dense `n × n` matrix per solve.
- Singularity checks are relative to the matrix scale, not an absolute `1e-14`.
- A step that ends on a target time takes that time exactly, which avoids a
  roundoff sliver. This is defensive: no reproducing input was found.
- The `ODE_DEBUG` environment lookup on every accepted step has been removed.
- The benchmark uses `std::hint::black_box`.

### Changed

- `CsrMatrix::jacobian` takes an RHS that returns `Result` and returns
  `Result<CsrMatrix, OdeError>`. This is a breaking change to a public method.
- `JitRhs::call` panics on a length mismatch, as its docs now say. Use
  `call_safe` for an error instead.
- The JIT module docs state that the generated code is a call stub into the
  Rust closure, not a compiled RHS.

## [0.1.0] — 2026-08-22

Initial release of `tpt-sci-ode` to crates.io.

### Added

- From-scratch ODE/DAE integrators on the dual-licensed `tpt-math` dense
  linear algebra:
  - `Method::Tsit45` — explicit Runge–Kutta 4(5), non-stiff.
  - `Method::TrBdf2` — 2-stage SDIRK (TR-BDF2), A-stable, stiff.
  - `Method::Esdirk34` — 4-stage ESDIRK order 3(4), A-/L-stable, stiff.
  - `Method::Bdf` — variable-order (1–5) backward differentiation, stiff, with a
    Nordsieck predictor-corrector and Hermite dense output.
- Closure-first `OdeProblem` API (`Rhs = Fn(f64, &[f64], &mut [f64])`) and an
  `Rc<dyn RhsCallable>` pipeline shared by closures and JIT RHS.
- Adaptive-step driver with Hermite dense output so `solve_dense` lands exactly on
  requested `t_eval` points.
- `JitRhs` / `JitRhsBuilder` / `compile_rhs` — an optional Cranelift JIT-compiled
  right-hand side that plugs into the same `RhsCallable` trait as a plain closure.
- `verify-diffsol` dev feature — retains `diffsol` only as an optional, dev-only
  verification oracle (excluded from `cargo deny` license scanning).
- Sparse Newton path for the implicit solvers: for systems with ≥ 64 states,
  `sdirk_stage` (TR-BDF2 / ESDIRK34) and the BDF corrector build their
  finite-difference Jacobians directly in compressed CSR storage and factor
  the Newton matrix `I − γ·J` with the in-crate sparse LU (`sparse` module),
  never materialising a dense Jacobian. Small systems keep the dense path.
  The `CsrMatrix` API is public (`nrows`, `ncols`, `nnz`, `get`, `mat_vec`,
  `jacobian`, `scaled_identity_minus_scaled`, `sparse_solve`).

### Changed

- **Replaced the `diffsol` wrap with an in-house, dual-licensed ODE engine**
  (`linalg` module: row-major `DMat`, LU with partial pivoting, finite-difference
  Jacobian). No `diffsol` / `nalgebra` / `faer` remains in the shipped dependency
  graph. The public API (`OdeProblem`, `OdeProblemBuilder`, `Method`, `solve`,
  `solve_dense`) is unchanged so downstream crates need no edits.
- `solve_dense` now builds the solver once and walks a single trajectory
  (`solver.solve_dense(t_eval)`), removing the previous O(n) redundant
  re-integration from `t0` for each evaluation point.

[0.1.0]: https://github.com/tpt-solutions/tpt-science/releases/tag/v0.1.0
