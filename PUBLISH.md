# tpt-science — crates.io Publish Tracker

18 `tpt-sci-*` crates, grouped into batches of 5 in **dependency order**.
Every crate in a batch has all of its `tpt-sci-*` dependencies already published
in an earlier batch, so `cargo publish` will never fail on a missing internal
dep (external `tpt-math-*` crates are already on crates.io).

> Before publishing each crate:
> `cargo publish -p <crate> --dry-run` then `cargo publish -p <crate>`.
> If a downstream crate depends on an already-published sibling, its
> `version = "0.1.0"` requirement is already satisfied.

## Batch 1 — foundations (no `tpt-sci-*` deps)

- [x] `tpt-sci-ode`
- [x] `tpt-sci-grid`
- [x] `tpt-sci-ppl`
- [x] `tpt-sci-image`
- [x] `tpt-sci-physics-rigid`

## Batch 2 — foundations (no `tpt-sci-*` deps)

- [ ] `tpt-sci-quantum`
- [ ] `tpt-sci-astro`
- [ ] `tpt-sci-md`
- [ ] `tpt-sci-dft-classical`
- [ ] `tpt-sci-cfd-core`

## Batch 3 — dependents (deps all in Batches 1–2)

- [ ] `tpt-sci-reaction-network`  (needs: ode)
- [ ] `tpt-sci-climate`           (needs: ode)
- [ ] `tpt-sci-electrophys`       (needs: ode, grid)
- [ ] `tpt-sci-ocean`             (needs: cfd-core)
- [ ] `tpt-sci-hemodynamics`      (needs: cfd-core, ode)

## Batch 4 — final dependents (deps all in Batches 1–3)

- [ ] `tpt-sci-sim-core`       (needs: ode, grid, reaction-network)
- [ ] `tpt-sci-kinetics`       (needs: reaction-network, ode)
- [ ] `tpt-sci-dft-electronic` (no `tpt-sci-*` deps)

---
### Dependency graph (internal edges only)
```
ode ──────────────► reaction-network ──► sim-core
 │  └────────────► reaction-network ──► kinetics
 │  └────────────► climate
 │  └────────────► electrophys (also ← grid)
 │  └────────────► hemodynamics (also ← cfd-core)
grid ─────────────► electrophys
cfd-core ─────────► ocean
cfd-core ─────────► hemodynamics (also ← ode)
```
No internal deps: ppl, image, physics-rigid, quantum, astro, md,
dft-classical, dft-electronic.
