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
| `tpt-sci-ode` | differential-equations | **implemented** | build from scratch | from-scratch ODE/DAE engine (Tsit45, TR-BDF2, ESDIRK34, variable-order BDF); dense + sparse LU; forward sensitivities; `f32`/`f64`; optional Cranelift JIT (`cranelift-jit`); diffsol is a dev-only oracle |
| `tpt-sci-grid` | pde | **implemented** | build from scratch | 1-D/2-D/3-D finite-difference grids; dense and `sparse` (CSR) Laplacians; depends on `tpt-math-linalg` |
| `tpt-sci-sim-core` | simulation | **implemented** | build from scratch | multi-scale orchestration, ODE and diffusion sub-models, cross-scale coupling, checkpoints; depends on `tpt-sci-ode`, `tpt-sci-grid` |
| `tpt-sci-ppl` | probabilistic-programming | **implemented** | build from scratch (NUTS) | consolidates `tpt-augur`; `nuts-rs` wrap dropped; R-hat, ESS, divergence-rate diagnostics via `Trace` |
| `tpt-sci-image` | imaging | **implemented** | build from scratch | 2-D parallel-beam CT, 3-D volume CT, and cone-beam FDK; depends on `tpt-math-signal-fft`, `tpt-math-linalg` |
| `tpt-sci-physics-rigid` | physics | **implemented** | build from scratch | rigid bodies with quaternion orientation, Coulomb friction, sweep-and-prune broad phase; rapier disqualified (ADR 0007) |
| `tpt-sci-quantum` | quantum | **implemented** | build from scratch | state-vector (≤20 qubits) and density-matrix paths with Kraus noise channels; QuantRS2 disqualified (ADR 0007) |
| `tpt-sci-astro` | astrodynamics | **implemented** | build from scratch | Keplerian two-body; J2/J4 secular rates; atmospheric drag; third-body (Kozai-Lidov); solar radiation pressure |
| `tpt-sci-reaction-network` | systems-biology | **implemented** | build from scratch | Catalyst.jl-style DSL → mass-action ODE; exact SSA, tau-leaping/CLE, minimal SBML reader, conservation laws; SDE/jump out of scope |
| `tpt-sci-md` | molecular-dynamics | **implemented** | build from scratch | Lennard-Jones and EAM; direct-sum Ewald; SHAKE/RATTLE constraints; cell lists; RDF; depends on `tpt-math-linalg` |
| `tpt-sci-dft-classical` | materials | **implemented** | wrap `feos` (`feos-dft`) | PC-SAFT classical/soft-matter DFT via `feos`; from-scratch square-gradient solver in 1-D and 3-D; MIT OR Apache-2.0 |
| `tpt-sci-kinetics` | chemical-kinetics | **implemented** | build from scratch | depends on `tpt-sci-reaction-network`; Arrhenius, Langmuir–Hinshelwood (multi-site), Eley–Rideal, coverage-dependent activation |
| `tpt-sci-cfd-core` | fluid-dynamics | **implemented** | build from scratch | depends on `tpt-math-linalg`; incompressible Navier–Stokes with explicit and SIMPLE schemes, k-ε/k-ω SST, unstructured FVM/FEM |
| `tpt-sci-hemodynamics` | biomechanics | **implemented** | build from scratch | depends on `tpt-sci-cfd-core`, `tpt-sci-ode`; 1-D compliant vessel network, Womersley flow, Windkessel 0-D/1-D/3-D coupling |
| `tpt-sci-electrophys` | biomechanics | **implemented** | build from scratch | depends on `tpt-sci-ode`, `tpt-sci-grid`; Hodgkin–Huxley and Ten Tusscher ionic models, anisotropic tensor diffusion, bidomain |
| `tpt-sci-climate` | earth-science | **implemented** | build from scratch | depends on `tpt-sci-ode`, `tpt-math-linalg`; 0-D EBM, multi-band and correlated-k radiative transfer, 3-D tracer transport, primitive-equation GCM |
| `tpt-sci-ocean` | earth-science | **implemented** | build from scratch | depends on `tpt-sci-cfd-core`; 2-D shallow water, 3-D z-level ocean with non-hydrostatic projection, nudging, EnKF, 3D-Var-lite |
| `tpt-sci-dft-electronic` | materials | **implemented** | build from scratch | Kohn–Sham in 1-D and 3-D real space; LDA and PBE; norm-conserving local pseudopotentials; periodic k-point band structure (1-D) |

## no_std posture

All crates in this pillar are `std`-only by design. The `tpt-math` dense
linear-algebra substrate and `tpt-sci-ode` (which implements its own ODE engine
from scratch) are themselves `std` crates, so every simulation/physics/quantum
crate transitively requires `std`. None currently target `no_std`.

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
[`tpt-sci-reaction-network`](crates/tpt-sci-reaction-network),
[`tpt-sci-md`](crates/tpt-sci-md),
[`tpt-sci-dft-classical`](crates/tpt-sci-dft-classical),
[`tpt-sci-kinetics`](crates/tpt-sci-kinetics),
[`tpt-sci-cfd-core`](crates/tpt-sci-cfd-core),
[`tpt-sci-hemodynamics`](crates/tpt-sci-hemodynamics),
[`tpt-sci-electrophys`](crates/tpt-sci-electrophys),
[`tpt-sci-climate`](crates/tpt-sci-climate),
[`tpt-sci-ocean`](crates/tpt-sci-ocean),
[`tpt-sci-dft-electronic`](crates/tpt-sci-dft-electronic).

## Examples

Every crate ships a runnable `examples/` program that is meatier than the
README snippet — a good first place to see the API in action:

| Crate | Example | Demonstrates |
|-------|---------|--------------|
| `tpt-sci-ode` | `van_der_pol` | Van der Pol limit cycle + dense trajectory (`solve_dense`) |
| `tpt-sci-quantum` | `bell_ghz` | Bell/GHZ states + measurement statistics |
| `tpt-sci-reaction-network` | `sir` | Full SIR run, peak-infected tracking |
| `tpt-sci-grid` | `diffusion` | 1-D diffusion of a Gaussian bump on a uniform grid |
| `tpt-sci-sim-core` | `multi_scale_cookbook` | SIR → diffusion cross-scale coupling (composes reaction-network + grid) |
| `tpt-sci-ppl` | `posterior` | NUTS posterior + R-hat / ESS / divergence diagnostics |
| `tpt-sci-image` | `reconstruction` | Parallel-beam CT of a phantom (FBP) |
| `tpt-sci-physics-rigid` | `collision` | Spheres under gravity + walls, elastic collisions |
| `tpt-sci-astro` | `propagation` | LEO propagation + J2 RAAN regression |
| `tpt-sci-md` | `lennard_jones` | LJ fluid: velocity-Verlet trajectory + RDF |
| `tpt-sci-dft-classical` | `adsorption` | PC-SAFT density-profile solve (wrapped `feos`) |
| `tpt-sci-kinetics` | `reactor` | Arrhenius + Langmuir–Hinshelwood, ODE via `tpt-sci-ode` |
| `tpt-sci-cfd-core` | `cavity` | Lid-driven cavity, divergence-free projection |
| `tpt-sci-hemodynamics` | `arterial_segment` | Compliant vessel area/flow over a cardiac cycle |
| `tpt-sci-electrophys` | `ap_wave` | Action-potential propagation across a tissue sheet |
| `tpt-sci-climate` | `warming` | 0-D EBM CO₂-doubling equilibrium warming |
| `tpt-sci-ocean` | `shallow_water` | Shallow-water gravity-wave propagation |
| `tpt-sci-dft-electronic` | `harmonic_atom` | 1-D Kohn–Sham solve of a two-electron well |

`sim-core` also ships `decay_coupled` (ODE↔diffusion coupling + checkpoint).
Run any of them with, e.g.:

```sh
cargo run --example van_der_pol -p tpt-sci-ode
```

## Benchmarking & coverage

- **Benchmarks.** `criterion` benches live in `crates/*/benches/`:
  `tpt-sci-ode` (`solve`), `tpt-sci-grid` (`laplacian`), `tpt-sci-quantum`
  (`apply_gate`), and `tpt-sci-image` (`radon`) — every crate now ships a
  representative Criterion suite. Run a single crate with
  `cargo bench -p <crate>`, or the whole workspace with `cargo bench
  --workspace`. The CI `benches` job runs a shortened measurement so the suite
  stays fast:
  `cargo bench --workspace --benches -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10`.
- **Coverage.** The CI `coverage` job produces an lcov report via
  `cargo-llvm-cov` and uploads it as a build artifact.
- **Docs.** The CI `doc` job builds docs with `RUSTDOCFLAGS=-D warnings`,
  catching rustdoc warnings and broken intra-doc links (doctests are exercised
  by `cargo test`).

See `RELEASE.md` for the (currently dormant) publish/versioning process.

## Changelogs

Each crate keeps its own `CHANGELOG.md` (initial `v0.1.0` entry plus an
`[Unreleased]` section): see e.g.
[`crates/tpt-sci-ode/CHANGELOG.md`](crates/tpt-sci-ode/CHANGELOG.md). The
workspace-level [`CHANGELOG.md`](CHANGELOG.md) tracks cross-cutting changes.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option.

Copyright (c) 2026 TPT Solutions.
