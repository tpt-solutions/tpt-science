# tpt-sci-cli

A declarative runner for reaction networks: describe a model in a short JSON
file, and the `tpt-sci` command integrates it and writes the trajectory as CSV.

```sh
cargo run --release -p tpt-sci-cli -- run crates/tpt-sci-cli/examples/sir.json
cargo run --release -p tpt-sci-cli -- run crates/tpt-sci-cli/examples/sir.json --out sir.csv
```

## Spec

```json
{
  "network": "beta, S + I --> 2 I\ngamma, I --> R",
  "parameters": { "beta": 0.0004, "gamma": 0.2 },
  "initial": { "S": 990, "I": 10, "R": 0 },
  "t_final": 60,
  "points": 60,
  "method": "bdf"
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `network` | required | Reactions in the textual DSL, one per line (`rate, reactants --> products`). |
| `parameters` | `{}` | Values for the named rate constants. |
| `initial` | required | Initial amount of each species; unlisted species start at 0. |
| `t0` | `0` | Start time. |
| `t_final` | required | End time, after `t0`. |
| `points` | `100` | Evenly spaced output times in `(t0, t_final]`. |
| `method` | `"bdf"` | `"tsit45"`, `"trbdf2"`, `"esdirk34"` or `"bdf"`. |
| `rtol`, `atol` | `1e-6`, `1e-9` | Integration tolerances. |

Rates follow mass-action kinetics on molecule counts, so a bimolecular constant
is a per-count rate: `beta = 0.0004` with 1000 individuals is the per-capita
rate `0.4`. Unknown fields are rejected, so a misspelt option is an error
rather than being ignored.

## Library

The same logic is available as `tpt_sci_cli::run_to_csv(json)` and
`tpt_sci_cli::run(&Spec)`, which returns a `tpt_sci_ode::Trajectory`.
