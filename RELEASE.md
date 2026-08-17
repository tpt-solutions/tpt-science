# Release & Publish

This document scopes the release/publish automation for `tpt-science`. It is a
**pre-publish checklist**, not a committed release process: at the time of
writing every crate sets `publish = false` and is consumed as a workspace/path
dependency, so no crate has been cut to [crates.io](https://crates.io).

Publishing only begins once a deliberate release decision flips `publish =
true` in the relevant `crates/*/Cargo.toml`. When that happens, the workflow in
`.github/workflows/publish.yml` (manual `workflow_dispatch`) enforces the
gates below and runs `cargo publish --dry-run` before any real publish.

## Pre-publish checklist (per crate)

1. **API stable.** Bump `version` in `[workspace.package]` (all crates share a
   version) and confirm the public API matches `spec.txt` scope for the release.
2. **Docs.** `cargo doc --workspace --no-deps --all-features` is clean
   (`RUSTDOCFLAGS=-D warnings` in CI `doc` job). `[package.metadata.docs.rs]`
   and `documentation = "https://docs.rs/..."` are present in every crate
   `Cargo.toml` (verified), so docs.rs will build automatically on publish.
3. **Changelog.** Add a dated `## [x.y.z]` entry to `CHANGELOG.md`.
4. **Tests/CI green.** `cargo fmt --check`, `cargo clippy --workspace
   --all-targets --all-features -- -D warnings`, `cargo test --workspace`,
   `cargo llvm-cov` (coverage), and `cargo bench --workspace` (benches) all pass.
5. **Dependency audit.** `cargo deny check` passes (CI `cargo-deny` job).
6. **In-crate deps.** If publishing a crate that depends on a sibling
   `tpt-sci-*` crate, the sibling must already be published (or the dependency
   must remain a `path` dependency, which crates.io rejects on publish — adjust
   to a versioned dependency first).
7. **Flip the flag.** Set `publish = true` in the crate's `Cargo.toml` (and any
   `path`-only in-crate dependencies to versioned deps).

## Publish order

The crates have a dependency DAG: `tpt-sci-ode` / `tpt-sci-grid` (leaves) →
`tpt-sci-sim-core` / `tpt-sci-reaction-network` → ... Publish leaves first, then
dependents, in topological order. The `publish.yml` workflow targets a single
crate per run precisely so this ordering can be honoured manually.

## Out of scope today

- Automated version bumps / tag creation / GitHub Release drafting.
- Continuous publishing on `main` (intentionally absent: `publish = false`).
- docs.rs custom build features beyond `all-features = true`.
