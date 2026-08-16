# tpt-science TODO

Simulation/modeling substrate for tpt-solutions' science verticals. See
`spec.txt` for the full design rationale and `tpt-rust-map/registry.toml`
for the canonical crate list (all `tpt-sci-*` rows already exist there
with `repo = "tpt-solutions/tpt-science"`).

Status snapshot at time of writing: this repo is spec-only (no Cargo
workspace yet). `tpt-math` and `tpt-engineering` (sibling pillar repos)
are already scaffolded and mid-implementation — use their `todo.md` /
`Cargo.toml` / `CRATE_AUDIT.md` as the template for how this repo's phases
should look once scaffolding starts.

## Phase 0 — Ecosystem Verification / Audits (pre-work, per-crate blocking)

Per `tpt-rust-map/TODO.md`: resolve each crate's own
`flagged-needs-audit` / `flagged-deferred` status before implementing
*that* crate — not before starting the repo. Phases 1–2 (scaffolding) can
start without these being resolved.

- [x] Re-confirm `diffsol` current license (MIT OR Apache-2.0) and
      maintenance status before wrapping in `tpt-sci-ode` — spec's
      research is from Aug 2026, re-check for drift.
- [x] Re-confirm `nuts-rs` current license and maintenance status before
      wrapping in `tpt-sci-ppl`.
- [x] Audit `tpt-soma` (local repo) for existing compartmental-ODE /
      structured-grid code before finalizing `tpt-sci-grid` /
      `tpt-sci-sim-core` scope.
- [x] Audit `tpt-cerebrum` (not local — fetch from
      `github.com/tpt-solutions/tpt-cerebrum`) for the same, before
      finalizing `tpt-sci-grid` / `tpt-sci-sim-core` scope.
- [x] Audit `tpt-augur` (local repo) before starting `tpt-sci-ppl` — this
      crate explicitly consolidates/supersedes it; confirm what carries
      over vs. gets rebuilt as the DSL layer.
- [x] Audit `tpt-physics-engine` (not local — fetch from
      `github.com/tpt-solutions/tpt-physics-engine`) before starting
      `tpt-sci-physics-rigid`. Higher stakes than a normal audit: rapier
      is disqualified (Apache-2.0-only, ADR 0007) so there's no wrap
      fallback if this duplicates existing from-scratch work.
- [x] Audit `tpt-selenograph` (not local — fetch from
      `github.com/tpt-solutions/tpt-selenograph`) before starting
      `tpt-sci-physics-rigid`, same reasoning as above.
- [x] Audit `tpt-q-phase` (not local — fetch from
      `github.com/tpt-solutions/tpt-q-phase`) before starting
      `tpt-sci-quantum`. Same higher-stakes reasoning — QuantRS2 is
      disqualified (Apache-2.0-only, ADR 0007), no wrap fallback.
- [x] Audit `tpt-spectra` (local repo) before starting `tpt-sci-image`.
- [x] Audit `tpt-system-zero` (not local — fetch from
      `github.com/tpt-solutions/tpt-system-zero`) before starting
      `tpt-sci-astro` — its own description ("orbital mechanics, J2...")
      suggests this may already exist there.
- [x] After each audit above: update the corresponding crate's `status`
      in `tpt-rust-map/registry.toml` (`flagged-needs-audit` → `planned`,
      or resolve differently if the audit changes scope/motivation).

## Phase 1 — Repo Scaffolding (complete)

- [x] Seed workspace from `tpt-rust-map/template/`: `Cargo.toml`,
      `LICENSE-MIT`, `LICENSE-APACHE`, `deny.toml`, `rust-toolchain.toml`,
      `rustfmt.toml`, `.github/` CI workflow.
      (Both `rust-toolchain.toml` and `rustfmt.toml` intentionally deviate
      from the template: edition `2024` and no `thumbv6m-none-eabi` target,
      matching this pillar's predominantly std-only posture.)
- [x] Fill `[workspace.package]` metadata in `Cargo.toml` (description,
      homepage, repository — matching `spec.txt` PROJECT OVERVIEW).
- [x] Verify all `tpt-sci-*` rows already in `tpt-rust-map/registry.toml`
      match this repo's final crate list — don't duplicate entries.
      (9 rows — ode, grid, sim-core, reaction-network, ppl, physics-rigid,
      quantum, image, astro — all present, no duplicates.)
- [x] Add `CONTRIBUTING.md`, `SECURITY.md`, `README.md` (mirror
      `tpt-engineering`'s versions).

## Phase 2 — Crate Scaffolding & Phase 3 — Implementation

One subsection per crate, in dependency order. Each crate: scaffold from
`template/` into `crates/<name>/`, implement, unit tests, doc comments,
register/update status in `registry.toml`.

### 3a. `tpt-sci-ode` (no audit blocker — can start first)
- [x] Scaffold crate, wrap `diffsol`.
- [x] Depends on `tpt-math-numeric` (published, `tpt-math` repo).
- [x] ODE/DAE solving API, tests, docs.

### 3b. `tpt-sci-grid` (no audit blocker — can start first)
- [x] Scaffold crate, build structured finite-difference grids/stencils.
- [x] Depends on `tpt-math-linalg` (published).
- [x] Cover reaction-diffusion / cable-equation / cortical-sheet spatial
      model use cases (motivated by tpt-soma, tpt-cerebrum).
- [x] Tests, docs.

### 3c. `tpt-sci-sim-core` (depends on 3a + 3b landing)
- [x] Scaffold crate: multi-scale simulation orchestration (time-stepping,
      cross-scale coupling, checkpointing).
- [x] Depends on `tpt-sci-ode`, `tpt-sci-grid`.
- [x] Tests, docs. (sub-models: `OdeSubModel` wrapping `tpt-sci-ode`,
      `DiffusionSubModel` using `tpt-sci-grid` laplacian; cross-scale
      coupling + checkpoint snapshot/restore verified by tests.)

### 3d. `tpt-sci-ppl` (blocked on tpt-augur audit)
- [x] Scaffold crate: model/DSL layer and NUTS sampler backend (built from
      scratch on `tpt-math-prob` + `tpt-math-autodiff-rev`; the spec's
      `nuts-rs` wrap was dropped per the "build our own internals" direction —
      see `crates/tpt-sci-ppl/src/lib.rs`).
- [x] Depends on `tpt-math-prob-bayes`, `tpt-math-prob-core`,
      `tpt-math-autodiff-rev` (all published).
- [x] Tests, docs.

### 3e. `tpt-sci-image` (blocked on tpt-spectra audit)
- [x] Scaffold crate: 2-D tomographic reconstruction (Radon transform,
      ram-lak filtered back-projection, naive back-projection), built from
      scratch. Scope is 2-D parallel-beam CT rather than fully n-dimensional
      (the "n-dimensional" wording in the original plan is not met; revisit if a
      vertical needs 3-D volumes).
- [x] Depends on `tpt-math-signal-fft`, `tpt-math-linalg` (published).
- [x] Tests, docs.

### 3f. `tpt-sci-physics-rigid` (blocked on tpt-physics-engine + tpt-selenograph audits)
- [x] Scaffold crate: rigid-body/collision physics, built from scratch
       (rapier disqualified, ADR 0007).
- [x] Depends on `tpt-math-linalg` (published).
- [x] Tests, docs.

### 3g. `tpt-sci-quantum` (blocked on tpt-q-phase audit)
- [x] Scaffold crate: qubit state-vector simulation (up to 20 qubits) plus a
      tensor-product (Kronecker) circuit formulation, built from scratch
      (QuantRS2 disqualified, ADR 0007). The `Circuit` type assembles the full
      real-embedded unitary via `tpt-math-linalg` and `State::apply_unitary`
      applies it; non-adjacent two-qubit gates are SWAP-decomposed.
- [x] Depends on `tpt-math-linalg`, `tpt-math-prob-core`, `num-complex`
      (published).
- [x] Tests, docs.

### 3h. `tpt-sci-astro` (blocked on tpt-system-zero audit)
- [x] Scaffold crate: two-body orbital mechanics / coordinate-frame
      primitives (Keplerian elements, ECI state vectors, Kepler propagation).
- [x] Depends on `tpt-math-linalg` (published).
- [x] Tests, docs.

## Phase 4 — Cross-cutting / CI / Release Readiness

- [x] Workspace-wide lint pass (`clippy::all` warn, per template) — clean
      (`cargo clippy --workspace --all-targets --all-features -- -D warnings`,
      zero warnings).
- [x] `no_std` audit per crate — all 8 crates are std-only by design (they pull
      `std` in transitively via `tpt-math-linalg`/`tpt-math-prob` and, for
      `tpt-sci-ode`, `diffsol`). Documented as a `## no_std posture` section in
      the workspace README; CI `no_std` job remains the intentional no-op
      placeholder (per ADR 0001, build `-p` against `thumbv6m-none-eabi` only
      once a crate is explicitly confirmed `no_std`).
- [x] `cargo deny` license/dependency check across workspace — passing. Added
      `RUSTSEC-2024-0436` (`paste` unmaintained) to the advisories `ignore`
      list with justification: `paste` is a build-time proc-macro pulled in
      transitively via the `diffsol` wrap target, stable and non-vulnerable.
- [x] README per crate + workspace-level README — wrote `README.md` for all 8
      crates (previously missing; each `Cargo.toml` already referenced
      `readme = "README.md"`); refreshed the workspace README status table
      (was stale — listed implemented crates as `planned`/`needs-audit`).
- [x] CHANGELOG — added `CHANGELOG.md` (v0.1.0 entry).
- [x] crates.io publish prep — publish metadata (description, keywords,
      categories, documentation, readme) verified present on all 8 crates. All
      crates remain `publish = false` and are consumed as workspace/path deps
      (see the workspace README "Not yet published" note); no crates.io release
      has been cut. Actual publishing is a separate release-decision step.
- [x] CI green across all crates — `cargo fmt --check`, `cargo clippy -D
      warnings`, and `cargo test --workspace` all pass locally; the `ci.yml`
      jobs (fmt, clippy, test, no_std, cargo-deny) are wired to match.

### 3i. `tpt-sci-reaction-network` (research pass complete — GREENLIT 2026-08)
- [x] Scaffold crate: species/rate/stoichiometry DSL for compartmental models
      (Rust equivalent of Julia's Catalyst.jl), built from scratch.
- [x] CRN IR: species, parameters, reactions (reactants/products/stoichiometry;
      mass-action rate law default + optional custom rate expressions).
- [x] Stoichiometry-matrix + reaction-rate-vector construction; ODE RHS
      builder `f(t,y,p) = S·r(y,p)` consumable by `tpt-sci-ode` (diffsol).
- [x] Optional textual chemical-notation DSL/parser (e.g. `kB, S + E --> SE`).
- [x] Depends on `tpt-sci-ode`.
- [x] Out of v1 (documented, not built): stochastic SSA (defer to `rebop`
      wrap), SDE/jump, SBML I/O, network analysis, conservation laws.
- [x] Update `registry.toml` status `planned` (done in research pass) + tests, docs.

## Deferred (not v1)

- [x] `tpt-sci-reaction-network` — **RESEARCH PASS COMPLETE (2026-08), GREENLIT.**
      Dedicated, targeted ecosystem survey done (previously blocked because only
      byproduct searches had been run). Finding: no dual-licensed Rust crate
      generates mass-action ODE RHS from a CRN IR (Catalyst.jl equivalent does
      not exist in Rust). `rebop` (MIT) is stochastic-only (Gillespie SSA) — the
      recommended wrap target for a *future* stochastic backend, not v1.
      `use-reaction` is representation primitives only; ODE solvers
      (diffsol/ode_solvers/russell_ode) are downstream consumers only. No
      existing CRN DSL in tpt-soma (spec-only) or tpt-cerebrum-simulacrum
      (neural-mass, not stoichiometric) to consolidate. DECISION: build from
      scratch. Promoted to implementation subsection **3i** above with explicit
      v1 scope and out-of-v1 list. `registry.toml` updated
      `flagged-deferred` → `planned` with the research-pass rationale.

## Phase 5 — Platform review follow-ups (2026-08-17)

Findings from a full-workspace review (bugs, gaps, adoption). Not yet
triaged into a phase/priority order — pull items into a numbered phase
when work starts.

### 5a. Correctness-adjacent footguns

- [ ] `tpt-sci-quantum`: `measure()` samples an outcome but does not
      collapse the state vector — surprising vs. standard simulator
      semantics (Qiskit/Cirq), a trap for multi-shot/mid-circuit-measurement
      use. Add state collapse (default-on, or an explicit
      `measure_collapsing()`).
- [ ] `tpt-sci-ode`: `solve_dense` re-runs a full independent integration
      from `t0` for every `t_eval` point instead of walking one trajectory —
      silent O(n) redundant work, a perf footgun for trajectory plotting.
- [ ] `tpt-sci-grid`: `Stencil` enum (`stencil.rs`) is defined but never
      used by `operator.rs`, which hardcodes its own stencil logic — remove
      the dead code or wire it in.
- [ ] `tpt-sci-image`: only crate with zero fallible public APIs — no
      `error.rs`; out-of-range coordinates are silently zero-padded instead
      of erroring, inconsistent with every other crate's `Result`
      convention and could mask bugs. Add an `ImageError` type.
- [ ] `tpt-sci-quantum`: `StateError` is defined inline in `lib.rs` instead
      of a separate `error.rs` — the one crate deviating from the per-crate
      `error.rs` convention. Move it out.

### 5b. Scope gaps (documented, but adoption-limiting)

- [ ] `tpt-sci-astro`: two-body/Keplerian only — no J2, drag, third-body,
      or N-body propagation. Biggest capability gap for real
      mission-design use; scope a J2 perturbation model if a consumer
      needs it.
- [ ] `tpt-sci-grid`: dense `DMatrix` only (O(n²) memory), 1D/2D max, no
      sparse backend, no 3D — despite motivating cortical-sheet/cable-
      equation use cases that often need volumes.
- [ ] `tpt-sci-physics-rigid`: naming/scope mismatch — "rigid-body physics"
      but no rotation/torque/orientation at all, point-mass sphere dynamics
      only. No friction, O(n²) collision with no broad-phase. Either scope
      down the name/docs or extend toward actual rigid-body mechanics.
- [ ] `tpt-sci-ppl`: no convergence diagnostics (no R-hat, no ESS;
      divergences tracked internally but silently discarded), no
      multi-chain support, `fit()` returns a bare `Vec<Vec<f64>>` instead
      of a `Trace`/`Sample` type. Most likely gap to block real Bayesian
      workflows.
- [ ] `tpt-sci-image`: 2D parallel-beam CT only, no 3D/volume support
      (known deviation from the original "n-dimensional" plan).
- [ ] `tpt-sci-reaction-network`: no stochastic SSA/Gillespie, no SDE/jump,
      no SBML I/O — already deferred with `rebop` flagged as the future SSA
      wrap target; lowest-priority gap of the set.

### 5c. Innovative / high-leverage additions

- [ ] Cross-crate cookbook example: reaction-network model driving
      `tpt-sci-sim-core`, coupled to a `tpt-sci-grid` diffusion field —
      demonstrates the "multi-scale platform" story the spec sells, which
      nothing currently shows end-to-end.
- [ ] `tpt-sci-ppl` diagnostics struct (R-hat, ESS, divergence rate)
      surfaced from `fit()`/a new `Trace` type — natural, scoped v1.1.
- [ ] Feature-flagged sparse-matrix backend for `tpt-sci-grid` (additive,
      doesn't have to replace dense) — unlocks realistically-sized PDE
      grids.

### 5d. Usability / automation

- [ ] Add `examples/` directories (workspace has none anywhere) — highest-
      leverage adoption change available; the first thing Rust users look
      for. See 5e for per-crate example ideas.
- [ ] Add a `cargo doc`/doctest-focused CI job to catch rustdoc warnings
      and broken intra-doc links explicitly (currently only implicit via
      `cargo test`).
- [ ] Add `[package.metadata.docs.rs]` config per crate ahead of the first
      real crates.io publish; `documentation = "https://docs.rs/..."`
      links in every `Cargo.toml` are currently dead.
- [ ] Add benchmark tracking (`criterion` or similar) — numerics-heavy
      workspace with no perf-regression tracking today.
- [ ] Add code coverage tracking.
- [ ] Scope a release/publish automation workflow for whenever `publish =
      false` flips (pre-publish checklist gap, not urgent today).
- [ ] Fix duplicate "no_std posture" section in the top-level `README.md`
      (appears twice, near-identical text — copy-paste artifact).
- [ ] Reconcile `rust-toolchain.toml` floating on `channel = "stable"`
      against the strict MSRV pin (`rust-version = "1.85"`); drop the
      apparently-unused `wasm32-unknown-unknown` target unless wasm work is
      actually planned.

### 5e. Adoption acceleration (examples/templates)

- [ ] Per-crate `examples/` with one runnable program meatier than the
      README snippet: Lotka-Volterra or Van der Pol for `tpt-sci-ode`;
      Bell-state + GHZ-state walkthrough with measurement stats for
      `tpt-sci-quantum`; full SIR or Michaelis-Menten run for
      `tpt-sci-reaction-network`; similar for the remaining crates.
- [ ] Workspace-level "cookbook" example composing 2-3 crates together
      (ties into 5c's multi-scale-platform example).
- [ ] Confirm adoption framing with maintainers: `CONTRIBUTING.md`
      explicitly refuses external PRs (issues only), so "faster adoption"
      most likely means faster integration by downstream internal
      consumers (tpt-soma, tpt-cerebrum, etc.), not public OSS onboarding.
      If so, prioritize examples framed around their actual use cases
      (compartmental ODEs, cortical-sheet diffusion) over generic
      Rust-crate polish.

## Out of scope

Unstructured FEM/mesh generation — genuine Rust ecosystem gap, but a
multi-year-scale problem not attempted alongside the other three pillars
here. Revisit only if `tpt-sci-grid` proves insufficient for a specific
vertical, and if so, treat it as its own repo, not a `tpt-science`
addition (see spec.txt ECOSYSTEM GAP JUSTIFICATION and OUT OF SCOPE).
