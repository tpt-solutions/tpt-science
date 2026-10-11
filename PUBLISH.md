# tpt-science — crates.io Publish Tracker

18 `tpt-sci-*` crates. Each batch depends only on crates that are already on
crates.io (or on an earlier batch), so `cargo publish` never fails on a missing
internal dep. External `tpt-math-*` crates are already on crates.io.

> Before publishing each crate:
> `cargo publish -p <crate> --dry-run` then `cargo publish -p <crate>`.

---

## Next release (planned)

### Versioning rule

Pre-1.0 semver: a breaking change bumps the minor (`0.1.x` → `0.2.0`);
additions and fixes bump the patch (`0.1.0` → `0.1.1`).

### Mechanics

- Crates inherit `version.workspace = true` (root `Cargo.toml`, `0.1.0`).
  A crate that changes version needs an explicit `version = "x.y.z"` in its own
  `Cargo.toml`.
- Root `[workspace.dependencies]` requirements must match the new versions:
  `tpt-sci-ode`, `tpt-sci-cfd-core`, `tpt-sci-climate` move to `"0.2.0"`, and
  `tpt-sci-units` is new at `"0.1.0"`. Patch bumps need no requirement change.
- Each changed crate's `CHANGELOG.md`: rename `## [Unreleased]` to
  `## [x.y.z] — YYYY-MM-DD` and add a fresh empty `## [Unreleased]`.
- Dependents on a bumped crate keep working on crates.io against the old
  version (`0.1.0` requirements still resolve to `0.1.0`). They only need a
  republish if they have their own changes; see "Unchanged" below.

### Batch A — no unpublished internal deps

| Crate | crates.io now | Next | Why |
| --- | --- | --- | --- |
| `tpt-sci-units` | — (new) | `0.1.0` | New crate. `kinetics`, `climate` and `astro` depend on it. |
| `tpt-sci-ode` | `0.1.0` | **`0.2.0`** | Breaking: `CsrMatrix::jacobian` now takes a `Result`-returning RHS and returns `Result`. Also adds events, trajectories, Jacobian patterns, `serde`. |
| `tpt-sci-cfd-core` | `0.1.0` | **`0.2.0`** | Breaking: `turbulence::eddy_viscosity` returns `Result<f64, CfdError>`. |
| `tpt-sci-dft-electronic` | `0.1.0` | `0.1.1` | Fixes: Jacobi eigensolver, Hartree sign, total energy. |
| `tpt-sci-image` | `0.1.0` | `0.1.1` | Adds `rayon` feature. |
| `tpt-sci-physics-rigid` | `0.1.0` | `0.1.1` | Module split (paths unchanged), tests added. |
| `tpt-sci-md` | `0.1.0` | `0.1.1` | Adds EAM, Ewald, PPPM, SHAKE/RATTLE, cell lists, `step_with`. |
| `tpt-sci-ppl` | `0.1.0` | `0.1.1` | Fixes NUTS U-turn handling and subtree sampling bias. Adds `gradient` module, `serde` derive. |

- [ ] `tpt-sci-units`      `0.1.0`
- [ ] `tpt-sci-ode`        `0.2.0`
- [ ] `tpt-sci-cfd-core`   `0.2.0`
- [ ] `tpt-sci-dft-electronic` `0.1.1`
- [ ] `tpt-sci-image`      `0.1.1`
- [ ] `tpt-sci-physics-rigid` `0.1.1`
- [ ] `tpt-sci-md`         `0.1.1`
- [ ] `tpt-sci-ppl`        `0.1.1`

### Batch B — needs Batch A

| Crate | crates.io now | Next | Needs | Why |
| --- | --- | --- | --- | --- |
| `tpt-sci-astro` | `0.1.0` | `0.1.1` | units | Constants read from `tpt-sci-units`. Adds environment and frames modules. |
| `tpt-sci-reaction-network` | `0.1.0` | `0.1.1` | ode `0.2.0` | Adds `simulate_ssa_ensemble`. |
| `tpt-sci-climate` | `0.1.0` | **`0.2.0`** | units, ode `0.2.0` | Breaking: `grey_radiative_transfer` returns `Result`. Constants from units. |
| `tpt-sci-ocean` | `0.1.0` | `0.1.1` | cfd-core `0.2.0` | Fixes y-momentum advection and forward-Euler instability. |
| `tpt-sci-hemodynamics` | `0.1.0` | `0.1.1` | cfd-core `0.2.0`, ode `0.2.0` | Fixes spatial coupling and implicit friction. |

- [ ] `tpt-sci-astro`             `0.1.1`
- [ ] `tpt-sci-reaction-network`  `0.1.1`
- [ ] `tpt-sci-climate`           `0.2.0`
- [ ] `tpt-sci-ocean`             `0.1.1`
- [ ] `tpt-sci-hemodynamics`      `0.1.1`

### Batch C — needs Batch B

| Crate | crates.io now | Next | Needs | Why |
| --- | --- | --- | --- | --- |
| `tpt-sci-kinetics` | `0.1.0` | `0.1.1` | units, reaction-network `0.1.1`, ode `0.2.0` | Module split (paths unchanged). `R_GAS` re-exported from units. Adds `rate_constant_at`. |

- [ ] `tpt-sci-kinetics`          `0.1.1`

### Unchanged — no release needed

- `tpt-sci-grid` `0.1.0` (no unreleased changes)
- `tpt-sci-quantum` `0.1.0` (no unreleased changes)
- `tpt-sci-dft-classical` `0.1.0` (no unreleased changes)
- `tpt-sci-electrophys` `0.1.0` (no unreleased changes; still builds against published `ode 0.1.0`)
- `tpt-sci-sim-core` `0.1.0` (no unreleased changes; still builds against published `ode 0.1.0`)

Republish `electrophys` and `sim-core` only if you want them to pick up
`ode 0.2.0` on crates.io. That would be a `0.1.1` with the new requirement.

### Not on the list — decide separately

- `tpt-sci` (umbrella, re-exports all domain crates). Not yet published. If you
  publish it, it goes last as `0.1.0` and needs every crate above at its new
  version.
- `tpt-sci-cli`, `tpt-sci-py`, `tpt-sci-wasm`: `publish = false`. Out of scope.

### Dependency graph — normal (non-dev) internal edges

```
units ─────────────► astro
units ─────────────► climate ──┐
units ─────────────► kinetics  │
ode ───────────────► reaction-network ──► kinetics
ode ───────────────► climate
ode ───────────────► hemodynamics
ode ───────────────► electrophys   (not republished)
ode ───────────────► sim-core      (not republished)
cfd-core ──────────► ocean
cfd-core ──────────► hemodynamics
grid ──────────────► cfd-core, climate, ocean, dft-electronic,
                     electrophys, sim-core, dft-classical
```

Dev-only edges (`cfd-core → kinetics`, `sim-core → climate / electrophys /
reaction-network`, `ppl → ode`) do not set the batch order. `cargo publish
--dry-run` will flag any that do.

---

## 0.1.0 — published (2026-08-22)

### Batch 1 — foundations (no `tpt-sci-*` deps)

- [x] `tpt-sci-ode`
- [x] `tpt-sci-grid`
- [x] `tpt-sci-ppl`
- [x] `tpt-sci-image`
- [x] `tpt-sci-physics-rigid`

### Batch 2 — foundations (no `tpt-sci-*` deps)

- [x] `tpt-sci-quantum`
- [x] `tpt-sci-astro`
- [x] `tpt-sci-md`
- [x] `tpt-sci-dft-classical`
- [x] `tpt-sci-cfd-core`

### Batch 3 — dependents

- [x] `tpt-sci-reaction-network`  (needs: ode)
- [x] `tpt-sci-climate`           (needs: ode)
- [x] `tpt-sci-electrophys`       (needs: ode, grid)
- [x] `tpt-sci-ocean`             (needs: cfd-core)
- [x] `tpt-sci-hemodynamics`      (needs: cfd-core, ode)

### Batch 4 — final dependents

- [x] `tpt-sci-sim-core`       (needs: ode, grid)
- [x] `tpt-sci-kinetics`       (needs: reaction-network, ode)
- [x] `tpt-sci-dft-electronic` (needs: grid)

> The 0.1.0 batches predate `tpt-sci-units`, so `astro`, `climate` and
> `kinetics` on crates.io do not depend on it.
