# tpt-sci-py

Python bindings for tpt-science, built with [PyO3](https://pyo3.rs).

This crate is excluded from the Cargo workspace, so CI does not need libpython
to link it. Build it on its own:

```sh
cd crates/tpt-sci-py
maturin build --release        # wheel
maturin develop --release      # editable install into the current virtualenv
```

```python
import tpt_sci

csv = tpt_sci.run_spec(
    '{"network": "k, A --> B", "parameters": {"k": 1.0}, '
    '"initial": {"A": 1.0}, "t_final": 1.0}'
)
print(csv)

print(tpt_sci.ebm_equilibrium_temperature(560.0))  # K, 0-D energy-balance model
```

Errors surface as `ValueError`. The Python API is intentionally small and
mirrors the WebAssembly bindings (`tpt-sci-wasm`).
