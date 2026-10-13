# tpt-sci-chemoinformatics

From-scratch cheminformatics for the tpt-science pillar: parse SMILES, compute
molecular descriptors, and fit lightweight QSPR models. The included domain
model targets **TRPA1 / TRPV1** activity, the receptors behind the "bite" and
"tingle" of Sichuan pepper, pungent compounds and capsaicin. These receptors are
activated by binding and gating, not by acidity, so the descriptor set contains
no pH or pKa term.

## What it does

| Stage | Module | Notes |
|---|---|---|
| SMILES → graph | `smiles`, `mol` | Organic subset, bracket atoms, branches, rings (incl. `%nn`), stereo tags parsed. Unsupported syntax returns `UnsupportedSyntax`, never a guess. |
| Descriptors | `descriptors` | MW, heavy atoms, HBD/HBA, rotatable bonds, TPSA and logP (approximate Ertl / Crippen-style), rings, Fsp3, polyene/amide/phenol/carbonyl counts, Wiener index. |
| Fingerprints | `fingerprint` | Hashed Morgan (radius 2, 1024 bits) and Tanimoto. |
| Data | `dataset` | CSV with `smiles,target[,name]`. Bring your own labels. |
| Models | `model` | Ridge regression on standardised descriptors, Tanimoto kNN, R²/RMSE/MAE, k-fold and leave-one-out CV (seeded, deterministic). |
| Domain | `trp` | `TrpModel` averages ridge and kNN and flags predictions whose nearest training neighbour is below a similarity threshold. |

## Limitations (read before use)

- **Predictions are only as good as your labelled data.** `trp::demo_set` holds
  four compounds per target with approximate literature potencies. It exists for
  tests and examples. Its cross-validated metrics are negative, meaning the
  models do not beat the mean, and the example prints a warning saying so.
- Descriptor values are approximations, not reference implementations of
  Crippen or Ertl. Use them for ranking and as features.
- Predictions with `in_domain == false` are extrapolations.
- Not a docking or free-energy method. Output is a relative activity score in the
  units of your labels (for example pEC50).

## Example

```sh
cargo run -p tpt-sci-chemoinformatics --example trp_qspr
```

```rust
use tpt_sci_chemoinformatics::{Dataset, Molecule, DescriptorSet, Target, TrpConfig, TrpModel};

let mol = Molecule::from_smiles("C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O")?; // hydroxy-α-sanshool
assert_eq!(mol.formula(), "C16H25NO2");

let data = Dataset::from_csv(std::fs::read_to_string("trp_curated.csv")?.as_str())?;
let model = TrpModel::fit(Target::Trpv1, &data, &TrpConfig::default())?;
let p = model.predict_smiles("COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O")?;
println!("{:.2} in_domain={}", p.consensus, p.in_domain);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Not yet implemented

- Optional `serde` serialisation of result types.
- Optional `rayon` parallel cross-validation.
- Bayesian QSPR via `tpt-sci-ppl`.

## Status

`publish = false` for now. Consumed as a path dependency through the `tpt-sci`
umbrella (`features = ["chemoinformatics"]`).
