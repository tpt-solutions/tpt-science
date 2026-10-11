# tpt-sci-wasm

WebAssembly bindings for tpt-science.

```sh
wasm-pack build --target web crates/tpt-sci-wasm
```

Exports `run_spec(json)` (declarative reaction-network runs, see `tpt-sci-cli`)
and `ebm_equilibrium_temperature(co2_ppm)`.
Also `ct_phantom(n)`, `ct_reconstruct(n, angles, filtered)` (CT back-projection)
and `lid_cavity_speed(n, reynolds, steps)` (lid-driven cavity speed field).

## Playground

`www/index.html` runs every export in the browser. Generate the glue first:

```sh
cargo build --target wasm32-unknown-unknown --release -p tpt-sci-wasm
wasm-bindgen --target web --out-dir crates/tpt-sci-wasm/www/pkg \
  target/wasm32-unknown-unknown/release/tpt_sci_wasm.wasm
```

Then serve `crates/tpt-sci-wasm/www/` over HTTP (ES modules do not load from
`file://`), for example `python -m http.server -d crates/tpt-sci-wasm/www`.
The `wasm-bindgen` CLI version must match the `wasm-bindgen` crate version in
`Cargo.lock`.
