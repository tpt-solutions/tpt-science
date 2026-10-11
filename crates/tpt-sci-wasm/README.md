# tpt-sci-wasm

WebAssembly bindings for tpt-science.

```sh
wasm-pack build --target web crates/tpt-sci-wasm
```

Exports `run_spec(json)` (declarative reaction-network runs, see `tpt-sci-cli`)
and `ebm_equilibrium_temperature(co2_ppm)`.
