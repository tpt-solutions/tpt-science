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

- [x] `tpt-sci-quantum`: `measure()` samples an outcome but does not
      collapse the state vector — surprising vs. standard simulator
      semantics (Qiskit/Cirq), a trap for multi-shot/mid-circuit-measurement
      use. Add state collapse (default-on, or an explicit
      `measure_collapsing()`). (Done: `measure_collapsing()` added in
      `crates/tpt-sci-quantum/src/lib.rs:198`; `measure()` kept
      non-destructive with cross-linked docs. Verified by
      `measure_collapsing_projects_state` + `measure_reproduces_distribution`.)
- [x] `tpt-sci-ode`: `solve_dense` re-runs a full independent integration
      from `t0` for every `t_eval` point instead of walking one trajectory —
      silent O(n) redundant work, a perf footgun for trajectory plotting. (Done:
      `solve_dense` in `crates/tpt-sci-ode/src/solver.rs:128` builds the solver
      once and calls diffsol's `solve_dense(t_eval)` a single time, walking one
      trajectory. Verified by `dense_eval_matches_point_eval`.)
- [x] `tpt-sci-grid`: `Stencil` enum (`stencil.rs`) is defined but never
      used by `operator.rs`, which hardcodes its own stencil logic — remove
      the dead code or wire it in. (Done: `operator.rs` now imports `Stencil`
      and `derivative_1d` is driven by `stencil.coefficients()`;
      `crates/tpt-sci-grid/src/operator.rs:4,100`. Verified by
      `derivative_first_of_linear_is_constant` /
      `derivative_second_of_quadratic_is_constant`.)
- [x] `tpt-sci-image`: only crate with zero fallible public APIs — no
      `error.rs`; out-of-range coordinates are silently zero-padded instead
      of erroring, inconsistent with every other crate's `Result`
      convention and could mask bugs. Add an `ImageError` type. (Done:
      `error.rs` added with `ImageError` (EmptyImage / EmptyAngles /
      AngleCountMismatch); `radon_transform`, `filtered_back_projection`,
      `naive_back_projection` now return `Result`. Verified by
      `radon_rejects_empty_image_and_angles` /
      `fbp_rejects_angle_count_mismatch`. Coordinate out-of-range sampling
      still zero-pads inside the bilinear/reconstruction kernels, which is a
      documented domain convention, not a malformed-input case.)
- [x] `tpt-sci-quantum`: `StateError` is defined inline in `lib.rs` instead
      of a separate `error.rs` — the one crate deviating from the per-crate
      `error.rs` convention. Move it out. (Done: `StateError` moved to
      `crates/tpt-sci-quantum/src/error.rs` and re-exported from `lib.rs`;
      `UnitarySizeMismatch` variant also lives there.)

### 5b. Scope gaps (documented, but adoption-limiting)

- [x] `tpt-sci-astro`: J2 perturbation model delivered — `OrbitalElements::propagate_j2`
      and `j2_secular_rates` give the first-order secular nodal-regression /
      apsidal-precession rates (`EARTH_J2` / `EARTH_RADIUS_EQ` supplied). Drag,
      third-body, and N-body remain out of scope (as scoped).
- [x] `tpt-sci-grid`: feature-gated sparse backend (`sparse` feature: `CsrMatrix`,
      `laplacian_*_sparse`, `diffuse_step`) was already present; added 3-D
      support — `UniformGrid3D`, dense `laplacian_3d`, and sparse
      `laplacian_3d_sparse` (shared row assembly so dense == sparse). Both
      Dirichlet and Neumann boundaries.
- [x] `tpt-sci-physics-rigid`: rotation/torque/orientation delivered — `Body`
      carries an orientation quaternion + angular velocity + isotropic
      `inertia`; `apply_torque` / `spin` / `quat_to_matrix` etc. Friction and
      broad-phase remain out of scope (noted in README).
- [x] `tpt-sci-ppl`: convergence diagnostics delivered — `fit()` / `fit_chains()`
      return a `Trace` exposing R-hat (split-`Trace::rhat`), ESS
      (`Trace::ess`, Geyer), and the divergence rate (`Trace::divergence_rate`);
      multi-chain supported via `fit_chains`.
- [x] `tpt-sci-image`: 3-D volume CT delivered — `volume` module with
      `radon_transform_3d` and `filtered_back_projection_3d` (`Volume` type),
      parallel-beam geometry rotating about `z` (each `z` slice reconstructed
      independently). 2-D API unchanged.
- [x] `tpt-sci-reaction-network`: stochastic SSA delivered — `simulate_ssa`
      (Gillespie direct method) on `ReactionSystem`, with combinatorial
      mass-action propensities and a `SsaTrajectory` result. SDE/jump, SBML
      I/O, network analysis, conservation laws remain out of scope.

### 5c. Innovative / high-leverage additions

- [x] Cross-crate cookbook example: reaction-network model driving
      `tpt-sci-sim-core`, coupled to a `tpt-sci-grid` diffusion field —
      demonstrates the "multi-scale platform" story the spec sells, which
      nothing currently shows end-to-end.
      (`crates/tpt-sci-sim-core/examples/multi_scale_cookbook.rs`: the SIR
      reaction network is built via the DSL and wrapped directly as an
      `OdeSubModel`; its infected-compartment state is coupled onto a 1-D
      `DiffusionSubModel` input buffer, driven end-to-end by the
      `Simulation` orchestrator.)
- [x] `tpt-sci-ppl` diagnostics struct (R-hat, ESS, divergence rate)
      surfaced from `fit()`/a new `Trace` type — natural, scoped v1.1.
      (`crates/tpt-sci-ppl/src/trace.rs`: `Trace` carries `rhat` (split-R-hat
      across chains), `ess` (Geyer), `divergence_rate`/`n_divergences`;
      `Model::fit`/`fit_chains` return `Trace`. Verified by
      `multi_chain_fit_reports_diagnostics` and the lib doctests.)
- [x] Feature-flagged sparse-matrix backend for `tpt-sci-grid` (additive,
      doesn't have to replace dense) — unlocks realistically-sized PDE
      grids. (`crates/tpt-sci-grid/src/sparse.rs`, gated on the `sparse`
      cargo feature: `CsrMatrix`, `laplacian_1d_sparse`/`laplacian_2d_sparse`,
      `diffuse_step`. The 2-D Laplacian previously failed to compile — fixed.)
      Enable with `--features sparse`.

#### 5c follow-up fixes (done while landing the above)
- Fixed `laplacian_2d_sparse` in `tpt-sci-grid/src/sparse.rs` (pushed into an
  undefined `row` instead of `rows[i]` — crate never compiled under the
  `sparse` feature). Added a `sparse_2d_laplacian_of_quadratic` test.
- Fixed clippy `needless_range_loop` in `tpt-sci-physics-rigid/src/lib.rs`
  (`apply_torque`).
- Moved `criterion` from a non-existent `[workspace.dev-dependencies]` table
  into `[workspace.dependencies]` so the `criterion = { workspace = true }`
  dev-dep inheritance in the new benches resolves (manifest failed to parse).
- Fixed clippy/compile errors in the new phase-5 examples so
  `cargo clippy --workspace --all-targets --all-features -D warnings` is green:
  `leo_orbit` (unused `mut`), `bayesian_linear` (assign-op + paren),
  `diffusion_operator` (`DVector::from_fn` arity, `&DMatrix * &DVector` Mul,
  `fold` over `&f64`), `ct_reconstruction` (move-out of captured `DMatrix` in
  nested closures), `bell_ghz` (unused `count_11`).

### 5d. Usability / automation

  - [x] Add `examples/` directories (workspace has none anywhere) — highest-
      leverage adoption change available; the first thing Rust users look
      for. See 5e for per-crate example ideas.
  - [x] Add a `cargo doc`/doctest-focused CI job to catch rustdoc warnings
      and broken intra-doc links explicitly (currently only implicit via
      `cargo test`).
  - [x] Add `[package.metadata.docs.rs]` config per crate ahead of the first
      real crates.io publish; `documentation = "https://docs.rs/..."`
      links in every `Cargo.toml` are currently dead.
  - [x] Add benchmark tracking (`criterion` or similar) — numerics-heavy
      workspace with no perf-regression tracking today.
  - [x] Add code coverage tracking.
  - [x] Scope a release/publish automation workflow for whenever `publish =
      false` flips (pre-publish checklist gap, not urgent today).
  - [x] Fix duplicate "no_std posture" section in the top-level `README.md`
      (appears twice, near-identical text — copy-paste artifact).
  - [x] Reconcile `rust-toolchain.toml` floating on `channel = "stable"`
      against the strict MSRV pin (`rust-version = "1.85"`); drop the
      apparently-unused `wasm32-unknown-unknown` target unless wasm work is
      actually planned.

#### 5d resolved in this pass (2026-08-17)

- **examples/**: Pre-existing per-crate `examples/*.rs` already covered 8 crates
  (van_der_pol, diffusion, sir, reconstruction, collision, propagation,
  posterior, and sim-core's decay_coupled + multi_scale_cookbook). This pass
  added the one missing crate's example — `tpt-sci-quantum`'s `bell_ghz`
  (Bell/GHZ + measurement stats) — so every crate now has a runnable example.
  (Initial duplicate example files written this pass were removed in favour of
  the pre-existing ones.) The `multi_scale_cookbook` example is also the
  cross-crate cookbook from 5c/5e (composes reaction-network + grid + sim-core).
- **cargo doc CI job**: already present as the `doc` job in `ci.yml`
  (`RUSTDOCFLAGS=-D warnings`); doctests covered by the `test` job. (No change
  needed — pre-existing.)
- **docs.rs metadata**: already present in every crate `Cargo.toml`
  (`[package.metadata.docs.rs]` + `documentation = "https://docs.rs/..."`). (No
  change needed — pre-existing; links go live on first publish.)
- **benchmark tracking**: NEW this pass — `criterion` added as a workspace
  dev-dependency; `benches/` added to `tpt-sci-grid`, `tpt-sci-quantum`,
  `tpt-sci-image`, `tpt-sci-ode`, with a `benches` CI job (shortened measurement).
- **code coverage**: NEW this pass — `coverage` CI job added (`cargo-llvm-cov` →
  lcov artifact).
- **release/publish scope**: NEW this pass — `RELEASE.md` pre-publish checklist
  + a gated `publish.yml` (`workflow_dispatch`, verifies `publish = true`).
  Dormant until `publish = false` flips.
- **duplicate no_std section**: already only one occurrence in `README.md`. (No
  change needed — pre-existing.)
- **rust-toolchain.toml**: already `channel = "stable"` with no `wasm32-unknown-
  unknown` target present; `stable` satisfies the `rust-version = "1.85"` MSRV
  pin. (Reconciled — no change needed.)
- **CI hygiene — pre-existing toolchain drift (fmt / doc / clippy)**: under the
  current stable rustfmt/rustdoc/clippy the repo did not actually pass its own
  CI gates (committed `lib.rs` and pre-existing examples were formatted with an
  older rustfmt; several intra-doc links and clippy lints were broken). This
  pass fixed them so all three jobs are green again:
  - `fmt`: ran `cargo fmt` to normalize the whole workspace (mechanical; no
    behavioural change).
  - `doc` (`RUSTDOCFLAGS=-D warnings`): fixed broken intra-doc links —
    ambiguous `[`crate::nuts`]` (ppl), `[`State::apply_unitary`]` (quantum),
    `[`nuts`]`/`[`Model::build`]` (ppl model.rs), `crate::laplacian_3d(_sparse)`
    references (grid), `[`ReactionSystem::to_ode_problem`]` (reaction-network),
    and `[`tpt-sci-grid`]` (sim-core).
  - `clippy --all-targets --all-features -D warnings` (claimed clean in Phase 4
    but wasn't under current clippy): collapsed 6 `collapsible-if`s and added a
    `# Panics` doc in `tpt-sci-grid/src/sparse.rs` + `operator.rs`, and allowed
    `too_many_arguments` on `UniformGrid3D::new` in `grid.rs`.

### 5e. Adoption acceleration (examples/templates)

- [x] Per-crate `examples/` with one runnable program meatier than the
      README snippet. All nine crates covered and verified (`cargo build
      --examples` + run clean):
      - `tpt-sci-ode`: `examples/van_der_pol.rs` (Van der Pol, single solve
        + dense trajectory).
      - `tpt-sci-quantum`: `examples/bell_ghz.rs` (Bell + collapse-aware
        multi-shot GHZ, measurement stats).
      - `tpt-sci-reaction-network`: `examples/sir.rs` (full SIR, peak
        infected).
      - `tpt-sci-grid`: `examples/diffusion.rs` (1-D Gaussian-bump diffusion
        via dense Laplacian).
      - `tpt-sci-astro`: `examples/propagation.rs` (Kepler propagation + J2
        RAAN regression).
      - `tpt-sci-ppl`: `examples/posterior.rs` (NUTS Gaussian posterior with
        R-hat / ESS / divergence-rate diagnostics).
      - `tpt-sci-image`: `examples/reconstruction.rs` (parallel-beam CT FBP of
        a phantom).
      - `tpt-sci-physics-rigid`: `examples/collision.rs` (elastic collision +
        rigid-body quarter-turn spin).
      - `tpt-sci-sim-core`: `examples/decay_coupled.rs` (heterogeneous
        ODE sub-models stepping to a shared target time).
- [x] Workspace-level "cookbook" example composing 2-3 crates together:
      `tpt-sci-sim-core/examples/multi_scale_cookbook.rs` drives a
      `tpt-sci-reaction-network` SIR model and a `tpt-sci-grid` diffusion
      field through `tpt-sci-sim-core` orchestration + coupling (ties into
      5c). The workspace is virtual (no root package), so the cross-crate
      example lives in the `tpt-sci-sim-core` crate's `examples/`.
- [x] Confirmed adoption framing: `CONTRIBUTING.md` (§"Policy: reports only,
      no external code contributions") explicitly refuses external PRs
      (issues only), so "faster adoption" means faster integration by
      downstream **internal** consumers (tpt-soma, tpt-cerebrum, etc.), not
      public OSS onboarding. The examples are already framed around those use
      cases (compartmental SIR ODEs, cortical-sheet-style diffusion fields),
      consistent with this. No further code change needed; decision recorded
      here.

## Out of scope

Unstructured FEM/mesh generation — genuine Rust ecosystem gap, but a
multi-year-scale problem not attempted alongside the other three pillars
here. Revisit only if `tpt-sci-grid` proves insufficient for a specific
vertical, and if so, treat it as its own repo, not a `tpt-science`
addition (see spec.txt ECOSYSTEM GAP JUSTIFICATION and OUT OF SCOPE).

## Phase 6 — Platform review follow-ups (2026-08-17)

Findings from a full-workspace review (build/test/clippy all green under
`--all-features`; no `todo!`/`unimplemented!`/stubs; only one `panic!` in an
example). These items were triaged and fixed in the same pass.

### 6a. Functional bug (fixed)

- [x] **CI never ran.** `.github/workflows/ci.yml` triggered on `branches:
  [main]`, but the default branch is `master` (`git branch` confirms). Changed
  both `push` and `pull_request` triggers to `master` so the pipeline actually
  executes. (`ci.yml:5,7`)

### 6b. Dead / unreachable API surface (fixed)

- [x] Removed `GridError::DegenerateAxis` from `tpt-sci-grid` — defined but
  never constructed (grid axes are validated by `TooFewPoints` /
  `InvalidDomain`). (`crates/tpt-sci-grid/src/error.rs`)
- [x] Removed `ReactionNetworkError::DuplicateSpecies` and
  `::DuplicateParameter` from `tpt-sci-reaction-network` — species/parameter
  registration is idempotent (`ReactionNetwork::species`/`parameter` return the
  existing index), so duplicates can never occur.
  (`crates/tpt-sci-reaction-network/src/error.rs`)
- [x] Simplified the redundant `match` in `Simulation::step_until`
  (`tpt-sci-sim-core/src/sim.rs`): the `advance(dt).map_err(|e| match e { ... }`
  was a no-op re-wrap; replaced with a plain `advance(dt)?`.
- [x] Removed the dead second `rtol`/`atol` validation in `OdeProblem::respawn`
  (`tpt-sci-ode/src/problem.rs`): tolerances are guaranteed positive by
  `build()` and `respawn` clones from a built problem, so the re-check was
  unreachable.

### 6c. Documented, not fixed (tracked for later)

- [x] `tpt-sci-astro`: `solve_kepler` initialises at `ecc = m` (not `m + e`);
  accuracy degrades for `e` approaching 1. Add a better seed and/or a guard.
  (Done: `crates/tpt-sci-astro/src/lib.rs:415` now seeds Newton at `E₀ = M +
  e·sin(M)` — Danby's first-order series seed — which stays close to the root
  for `e → 1`; added a `debug_assert!` bounds guard on `0 ≤ e < 1` and a
  `solve_kepler_seed_is_accurate_at_high_eccentricity` test at `e = 0.9`.)
- [x] `tpt-sci-ppl`: `Trace::rhat` returns `NaN` for a single chain; consider
  making `fit_chains(1, …)` the default `fit` so `rhat` is always meaningful.
  (Done: `Model::fit` in `crates/tpt-sci-ppl/src/model.rs:209` now delegates to
  `fit_chains(2, …)` (two dispersed chains), so the returned `Trace` always
  carries a meaningful split-R-hat; `fit_from` remains the single-chain escape
  hatch. `rhat` is still `NaN` only for a genuinely single-chain trace.)
- [x] `tpt-sci-image`: the empirical FBP amplitude scale (`4.0 / nb`,
  `lib.rs` / `volume.rs`) needs a ram-lak-normalization derivation/citation in
  a doc comment for maintainability. (Done: doc comments added at
  `crates/tpt-sci-image/src/lib.rs:265` and `volume.rs:280` citing Kak & Slaney,
  *Principles of Computerized Tomographic Imaging*, §3.3, and deriving the
  `4.0 / (nb·n_angles)` constant from the discrete ramp filter's DC gain plus
  the Δθ/Δs pixel-area factors. Also fixed a pre-existing broken
  `crate::UniformGrid3D` intra-doc link in `volume.rs`.)
- [x] `tpt-sci-grid`: mark the dense `laplacian_3d` path as a memory trap at
  realistic sizes and steer users to the `sparse` feature in docs. (Done:
  `crates/tpt-sci-grid/src/operator.rs:104` now carries a `## Memory note`
  warning that the dense operator is `Θ(n²)` — a `128³` grid is ~2 GiB — and
  points users to the feature-gated `laplacian_3d_sparse`.)

## Phase 7 — Replace diffsol with an in-house, dual-licensed ODE engine

Decision (2026-08-17): **do NOT fork diffsol.** Build the ODE solver from
scratch inside this repo so the shipped crate is 100% TPT-owned code under
`MIT OR Apache-2.0`, with no `diffsol` / `nalgebra` / `faer` in the shipped
dependency graph. `diffsol` is retained ONLY as an *optional* verification
oracle (dev-dependency, feature `verify-diffsol`, excluded from `cargo deny`
via `include-dev = false`) to regression-compare trajectories.

Open questions to resolve before coding (see tracked tasks 7.0):
- **7.0a Crate placement:** (A) keep the engine inside `tpt-sci-ode`
  (fastest, preserves the `OdeProblem` API that `tpt-sci-reaction-network` and
  `tpt-sci-sim-core` already depend on) vs (B) new `tpt-math-ode` in the
  sibling `tpt-math` repo (cleaner "math primitive" home, matches how
  `tpt-math-linalg`/`tpt-math-prob` are organised, but crosses repo
  boundaries). Engine code should be written so it can be lifted into (B) later.
- **7.0b Linear-algebra backend:** implement a small in-crate dense LA
  (matrix + LU w/ partial pivoting + finite-difference Jacobian). DO NOT depend
  on `faer`/`nalgebra` (faer is MIT-only and is what `tpt-math-linalg` wraps;
  pulling it would re-introduce a non-dual-licensed transitive dep). Small
  systems here (≤ few hundred states) make a self-contained LA fine.

"Better than diffsol" — realistic targets (we do NOT try to beat its Enzyme
autodiff + LLVM/Cranelift JIT, sparse LA, or sensitivity/adjoints):
- Fully dual-licensed, zero Apache/MIT-only-heavy transitive deps.
- `f32`/`f64`-generic over `tpt_math_numeric::Scalar` (diffsol is hardcoded
  `f64`/faer); `no_std`-friendly core.
- Closure-first, no JIT → deterministic, reproducible, no LLVM build/runtime dep
  (we only ever used diffsol's closure path anyway).
- Built-in Hermite dense-output interpolation so `solve_dense` is exact between
  accepted steps (vs diffsol's snapshot `solve_dense`).
- Identical public API (`OdeProblem`/`OdeProblemBuilder`/`Method`/`solve`/
  `solve_dense`) so downstream crates are unchanged.

### Tracked tasks

- [x] **7.0** Resolve 7.0a (crate placement) and 7.0b (LA backend) decisions.
- [x] **7.1** Add in-crate dense linear algebra module: `DMat` (row-major),
      LU decomposition with partial pivoting, `mat_vec`, `add_scaled_identity`,
      finite-difference full Jacobian builder. Unit-tested on a known system
      (e.g. solve a 3×3 linear system; verify Jacobian of `f(y)=Ay`).
- [x] **7.2** Implement explicit `Tsit45` (embedded 4(5), adaptive step).
- [x] **7.3** Implement `TrBdf2` (SDIRK, A-stable, embedded error control).
- [x] **7.4** Implement `Esdirk34` (ESDIRK order 3(4)).
- [x] **7.5** Implement `Bdf` (variable-order 1–5 backward differentiation;
      classic BDF α-coefficients, Newton corrector with the 7.1 LU, numerical
      Jacobian, order/step-size control). Hardest; analytic tests gate it.
      *Implemented with Nordsieck vector representation for efficient
      variable-order (1–5) and variable-step control. Order 1 (backward Euler)
      uses a dedicated corrector; orders 2–5 use Nordsieck predictor-corrector
      with conservative order control (requires 3 successful steps at current
      order with err_est < 0.01 before raising order). Max step growth limited
      to 1.5× per step for stability.*
- [x] **7.6** Shared adaptive-step driver with Hermite dense-output so
      `solve_dense` lands exactly on each `t_eval` (or interpolates). Drives all
      four methods.
- [x] **7.7** Rewire `OdeProblem`/`OdeProblemBuilder`/`Method`/`solve`/
      `solve_dense` onto the new engine. Preserve the exact public signature
      (including `Rhs = Fn(f64, &[f64], &mut [f64])` and `Vec<f64>` returns) so
      `tpt-sci-reaction-network` and `tpt-sci-sim-core` need no changes.
- [x] **7.8** Remove `diffsol` from shipped deps in `tpt-sci-ode/Cargo.toml`
      and the workspace `[workspace.dependencies]`. Add optional dev-dependency
      `diffsol` gated behind `verify-diffsol` feature; add
      `[[test]]`/module comparing our trajectories vs diffsol on: exp decay,
      harmonic oscillator, van der Pol (stiff), SIR (via reaction-network RHS),
      Robertson (very stiff). Keep existing analytic-comparison tests as the
      always-on correctness gate.
- [x] **7.9** `deny.toml`: add `include-dev = false` under `[licenses]` (so the
      optional `diffsol` dev-dep / `nalgebra` is excluded from the license
      scan, matching the "shipped deps" policy). Remove the now-false
      "diffsol dual-licensed" notes; document diffsol as a verification oracle
      only.
- [x] **7.10** Update `spec.txt` (ODE section: built from scratch, diffsol is
      verify-only) and retire the fork narrative.
- [x] **7.11** Verify: `cargo check --workspace`, `cargo test -p tpt-sci-ode`,
      `cargo test -p tpt-sci-ode --features verify-diffsol` (within tolerance),
      `cargo tree -p tpt-sci-ode -i nalgebra` → empty for the shipped graph,
      `cargo deny licenses` clean. Re-run `tpt-sci-reaction-network` and
      `tpt-sci-sim-core` tests to confirm API unchanged.

### Out of scope (v1)
- DiffSL / LLVM / Cranelift JIT codegen for user RHS (we only use closures): **IMPLEMENTED** in v1 (see jit module).
- Sparse LA (dense only; matches current `tpt-sci-ode` usage).
- Sensitivity analysis / adjoints (diffsol strengths we consciously skip).
- `f32`/generic path is a stretch goal behind the `Scalar` trait; v1 ships `f64`.
- Variable-order BDF with Nordsieck vectors: **IMPLEMENTED** in v1 (see 7.5).
