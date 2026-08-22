# AGENTS.md — tpt-science

Simulation/modeling substrate (18 `tpt-sci-*` crates). Virtual Cargo workspace,
no root package. Read `README.md`, `todo.md`, `spec.txt`, `spec2.txt` for scope.

## Verify / lint / test (matches CI)

```sh
cargo fmt --check                                   # CI fmt (edition 2024, max_width 100)
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features       # run with RUSTDOCFLAGS=-D warnings
cargo deny check                                     # license/security hygiene
cargo bench --workspace                              # full criterion run
```

CI sets `RUSTFLAGS=-D warnings` and the `doc` job sets `RUSTDOCFLAGS=-D warnings`,
so warnings, rustdoc warnings, and broken intra-doc links fail the build.

## Toolchain

- `channel = "stable"` (rust-toolchain.toml), `rustfmt` + `clippy` components.
- Edition `2024`; `rust-version = "1.85"` (MSRV). `rustfmt.toml`: `max_width = 100`.
- Run `cargo fmt` before committing; CI fails on `--check`.

## Workspace layout

- 18 crates under `crates/`, in dependency order (see README status table).
- Cross-pillar `tpt-math-*` deps come from **crates.io** (workspace deps in
  `Cargo.toml`), NOT a sibling checkout. `feos` (`0.10`, `default-features = false`,
  feature `dft`) is the only wrap target (classical DFT, `tpt-sci-dft-classical`).
- All crates are `publish = false`; consumed as path deps. Not on crates.io yet.

## Hard constraints (easy to violate, costly to fix)

- **License (ADR 0007):** only MIT / MIT OR Apache-2.0 / other permissive deps
  allowed. **Apache-2.0-ONLY crates are disqualified** — never add `rapier`,
  `QuantRS2`, `SciRS2`, or similar. `deny.toml` enforces the allow-list.
- **`diffsol` is dev-only.** It may appear ONLY as an optional dev-dependency
  behind `tpt-sci-ode`'s `verify-diffsol` feature. Never add it to shipped deps.
  `deny.toml` sets `include-dev = false` so it's excluded from license scanning.
- **No `no_std` build in CI** is intentional (per-crate opt-in only, never
  workspace-wide). `tpt-sci-ode` uses Cranelift JIT (`unsafe`, allowed
  workspace-wide) — keep that path building.

## Useful feature flags

- `tpt-sci-grid`: `sparse` feature adds `CsrMatrix` + sparse Laplacians
  (dense `laplacian_3d` is a memory trap at realistic sizes — steer to `sparse`).
- `tpt-sci-ode`: default `cranelift-jit`; `verify-diffsol` runs trajectory
  comparisons against diffsol in tests.
- `docs.rs` per crate builds with `all-features = true`.

## Conventions

- **No external PRs** — see `CONTRIBUTING.md` (issues only; maintainers implement).
- Per crate: own `README.md`, `CHANGELOG.md` (`v0.1.0` + `[Unreleased]`), and
  `error.rs` (`Result`-returning APIs; the one place `ImageError`/`StateError`
  etc. live). Errors return `Result`, not silent zero-padding, except documented
  domain conventions (image coordinate out-of-range).
- Virtual workspace: the cross-crate "cookbook" example lives in
  `tpt-sci-sim-core/examples/multi_scale_cookbook.rs` (composes reaction-network
  + grid + sim-core), since there is no root package to host it.
- `cargo deny` must use **v2** (the action pins `cargo-deny-action@v2`); v1
  aborts on CVSS 4.0 advisories. `RUSTSEC-2024-0436` and `RUSTSEC-2020-0168` are
  intentionally ignored (transitive via diffsol/cranelift dev/build path).

## Benchmarks

`criterion` benches live in every `crates/*/benches/` (18 crates). Run a single
crate with `cargo bench -p <crate>`. CI runs a shortened measurement:
`cargo bench --workspace --benches -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10`.

## Coverage

`cargo llvm-cov --workspace --lcov --output-path lcov.info` (needs
`llvm-tools-preview` component). CI uploads `lcov.info` as an artifact.
