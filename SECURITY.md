# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `v0.1.x` (unreleased) | ⚠️ Not yet — no tagged release exists |
| `main`  | Best-effort (pre-release) |

This repository is in active pre-`v0.1.0` development. Treat all APIs as
unstable until a `v0.1.0` tag is cut.

## Reporting a Vulnerability

If you discover a security vulnerability in `tpt-science` or any of its
`tpt-sci-*` crates:

1. **Do not open a public GitHub issue.**
2. Report it privately to **TPT Solutions** security contact
   (security at tpt-solutions — internal routing) or, if unavailable, via
   GitHub's private vulnerability reporting on the repository.
3. Include:
   - Affected crate(s) and version/commit,
   - A minimal reproduction,
   - The impact and any known mitigations.

We aim to acknowledge reports within **5 business days** and to provide a
remediation plan within **30 days**, depending on severity.

## Supply-chain posture

- `unsafe_code` is forbidden workspace-wide (`[workspace.lints.rust] unsafe_code = "forbid"`).
- Wrap targets must be dual-licensed MIT OR Apache-2.0 under the policy in
  `tpt-rust-map/docs/adr/0007`; Apache-2.0-ONLY crates (rapier, QuantRS2,
  SciRS2) are explicitly disallowed as dependencies.
- `cargo-deny` (advisories/licences/bans/sources) and `cargo-audit` run in CI;
  unknown registries/git sources and yanked certificates are rejected.
- `Cargo.lock` is committed so builds are reproducible.
