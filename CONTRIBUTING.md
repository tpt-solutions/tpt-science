# Contributing to tpt-science

## Policy: reports only, no external code contributions

`tpt-science` is developed internally by TPT Solutions. To keep the
science substrate coherent and auditable, **we do not accept external
pull requests or code contributions.**

That said, we welcome two kinds of input from outside the core team:

- **Bug reports** — open a [GitHub Issue](https://github.com/tpt-solutions/tpt-science/issues)
  with a minimal reproduction, the affected crate(s), and the commit/tag or
  `main` revision you are on.
- **Feature requests** — open a [GitHub Issue](https://github.com/tpt-solutions/tpt-science/issues)
  describing the science use case, the expected API shape, and any
  reference (paper, textbook, or spec) the behaviour should follow.

Please do **not** open a PR proposing code changes. Issues are triaged by the
maintainers, who implement accepted changes internally.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.** See
[`SECURITY.md`](SECURITY.md) for the private reporting path and what to include
in a report.

## Local workflow (for maintainers)

```sh
cargo test --workspace          # full workspace test suite
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
cargo deny check                # license/security/dependency hygiene
```

## Crate addition order

Crates are added in dependency order, and per `tpt-rust-map/TODO.md` each
crate's `flagged-needs-audit` / `flagged-deferred` status must be resolved
*before* implementing *that* crate. The scaffold here is the Phase 1 seed;
crate work follows the phases in `todo.md`.

See the pillars' design rationale in [`spec.txt`](spec.txt) and the canonical
crate list in `tpt-rust-map/registry.toml`.
