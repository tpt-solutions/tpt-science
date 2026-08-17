# tpt-science

[![CI](https://github.com/tpt-solutions/tpt-science/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-science/actions/workflows/ci.yml)
[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Simulation and modeling substrate for the [TPT Solutions](https://github.com/tpt-solutions)
science verticals: differential equations, structured-grid PDE, multi-scale
simulation orchestration, probabilistic programming, rigid-body physics,
quantum simulation, tomographic imaging, and astrodynamics.

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

> **Not yet published to [crates.io](https://crates.io).** These crates are
> consumed as workspace/path dependencies (`publish = false`); there is no
> `v0.1.0` tag yet. APIs are unstable until the first tagged release.

## Status

This repo is scaffolded from `tpt-rust-map/template/`. Build order and crate
scope follow `todo.md` and `spec.txt`. Per `tpt-rust-map/TODO.md`, each crate's
own `flagged-needs-audit` / `flagged-deferred` status is resolved before that
crate is implemented — Phase 1 (scaffolding) does not require those audits.

| Crate | Domain | Status | Wrap / build | Notes |
|-------|--------|--------|--------------|-------|
| `tpt-sci-ode` | differential-equations | **implemented** | wrap `diffsol` | depends on `tpt-math-numeric` |
| `tpt-sci-grid` | pde | **implemented** | build from scratch | depends on `tpt-math-linalg` |
| `tpt-sci-sim-core` | simulation | **implemented** | build from scratch | depends on `tpt-sci-ode`, `tpt-sci-grid` |
| `tpt-sci-ppl` | probabilistic-programming | **implemented** | build from scratch (NUTS) | consolidates `tpt-augur`; `nuts-rs` wrap dropped |
| `tpt-sci-image` | imaging | **implemented** | build from scratch | depends on `tpt-math-signal-fft`, `tpt-math-linalg`; 2-D CT only |
| `tpt-sci-physics-rigid` | physics | **implemented** | build from scratch | rapier disqualified (ADR 0007) |
| `tpt-sci-quantum` | quantum | **implemented** | build from scratch | QuantRS2 disqualified (ADR 0007); ≤20 qubits; no tensor-networks |
| `tpt-sci-astro` | astrodynamics | **implemented** | build from scratch | two-body / Keplerian only |
| `tpt-sci-reaction-network` | systems-biology | **implemented** | build from scratch | Catalyst.jl-style species/rate/stoichiometry DSL → mass-action ODE; depends on `tpt-sci-ode`; DSL + custom rate laws; SSA/SBML out of v1 |

## no_std posture

All crates in this pillar are `std`-only by design. The `tpt-math` dense
linear-algebra substrate and the `diffsol` wrap target (`tpt-sci-ode`) are
themselves `std` crates, so every simulation/physics/quantum crate transitively
requires `std`. None currently target `no_std`.

Per ADR 0001, `no_std` is opted into per-crate, never forced workspace-wide: the
CI `no_std` job is an intentional no-op until a crate is explicitly confirmed
`no_std` and built with `-p <crate>` against `thumbv6m-none-eabi`. See
`CHANGELOG.md` for the per-crate audit table.

## Building

This workspace depends on the [`tpt-math`](https://github.com/tpt-solutions/tpt-math)
substrate (`tpt-math-numeric`, `tpt-math-linalg`, `tpt-math-signal-fft`),
resolved as regular versioned dependencies from crates.io — no sibling checkout needed:

```sh
git clone https://github.com/tpt-solutions/tpt-science.git
cd tpt-science
cargo build --workspace
cargo test --workspace
```

- **Edition:** `2024`.
- **MSRV:** `1.85` (pinned via `rust-version` in `[workspace.package]`).
- **Toolchain:** fixed by `rust-toolchain.toml` via `rustup`.

## Quickstart

```rust
use tpt_sci_ode::{OdeProblem, Method};

// dy/dt = -y  ->  y(t) = y0 * exp(-t)
let prob = OdeProblem::new(|_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0).unwrap();
let y = prob.solve(Method::Bdf, 1.0).unwrap();
assert!((y[0] - std::f64::consts::E.recip()).abs() < 1e-6);
```

See the crate READMEs for the full API:
[`tpt-sci-ode`](crates/tpt-sci-ode),
[`tpt-sci-grid`](crates/tpt-sci-grid),
[`tpt-sci-sim-core`](crates/tpt-sci-sim-core),
[`tpt-sci-ppl`](crates/tpt-sci-ppl),
[`tpt-sci-image`](crates/tpt-sci-image),
[`tpt-sci-physics-rigid`](crates/tpt-sci-physics-rigid),
[`tpt-sci-quantum`](crates/tpt-sci-quantum),
[`tpt-sci-astro`](crates/tpt-sci-astro),
[`tpt-sci-reaction-network`](crates/tpt-sci-reaction-network).

## Examples

Every crate ships a runnable `examples/` program that is meatier than the
README snippet — a good first place to see the API in action:

| Crate | Example | Demonstrates |
|-------|---------|--------------|
| `tpt-sci-ode` | `vander_pol` | Van der Pol limit-cycle integration |
| `tpt-sci-quantum` | `bell_ghz` | Bell/GHZ states + measurement statistics |
| `tpt-sci-reaction-network` | `sir_epidemic` | Full SIR run from the Catalyst.jl-style DSL |
| `tpt-sci-grid` | `diffusion_operator` | 1-D Laplacian vs. analytic 2nd derivative |
| `tpt-sci-sim-core` | `coupled_field` | ODE→diffusion cross-scale coupling + checkpoint |
| `tpt-sci-ppl` | `bayesian_linear` | Bayesian linear regression via from-scratch NUTS |
| `tpt-sci-image` | `ct_reconstruction` | Parallel-beam CT of a phantom (FBP) |
| `tpt-sci-physics-rigid` | `bouncing_balls` | Gravity + walls in a rigid-body world |
| `tpt-sci-astro` | `leo_orbit` | LEO propagation + J2 RAAN regression |

Run any of them with, e.g.:

```sh
cargo run --example vander_pol -p tpt-sci-ode
```

## Benchmarking & coverage

- **Benchmarks.** Numerics-heavy crates carry `criterion` benches under
  `benches/`. Run them with `cargo bench --workspace` (or per-crate). The CI
  `benches` job runs a shortened measurement so the suite stays fast.
- **Coverage.** The CI `coverage` job produces an lcov report via
  `cargo-llvm-cov` and uploads it as a build artifact.
- **Docs.** The CI `doc` job builds docs with `RUSTDOCFLAGS=-D warnings`,
  catching rustdoc warnings and broken intra-doc links (doctests are exercised
  by `cargo test`).

See `RELEASE.md` for the (currently dormant) publish/versioning process.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option.

Copyright (c) 2026 TPT Solutions.
