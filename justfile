# Developer task runner. Mirrors the CI jobs in .github/workflows/ci.yml.
# Install with `cargo install just`; run `just` to list recipes.
# The `examples` recipe is POSIX-sh; on Windows it runs under Git for Windows' sh.

default:
    @just --list

# Format check, lint, and test — the local equivalent of the core CI jobs.
check: fmt clippy test

# Full CI sequence, including docs, feature matrix, and examples.
ci: check doc feature-each examples

fmt:
    cargo fmt --check

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace --all-features

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

feature-each:
    cargo hack check --workspace --each-feature --all-targets

# Build every example in release mode and run each one.
examples:
    cargo build --release --workspace --examples --all-features
    @for f in crates/*/examples/*.rs; do \
        name=$(basename "$f" .rs); \
        echo "== $name"; \
        "target/release/examples/$name" > /dev/null || { echo "FAILED: $name"; exit 1; }; \
    done

# Shortened criterion run (same timing as the CI `benches` job).
bench:
    cargo bench --workspace --benches -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10

# Enable the repo's pre-commit hook (fmt + clippy) for this clone.
hooks:
    git config core.hooksPath .githooks

# Apply rustfmt to the whole workspace.
fmt-fix:
    cargo fmt
