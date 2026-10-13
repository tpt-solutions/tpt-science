# Changelog

All notable changes to `tpt-sci-chemoinformatics` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Initial crate: SMILES parser (organic subset, bracket atoms, rings, branches,
  aromatic lowercase atoms, bond/stereo tags), `Molecule` graph with implicit
  hydrogens, ring perception and Hückel aromaticity.
- `DescriptorSet` (18 structural descriptors, fixed order), Morgan fingerprints
  and Tanimoto similarity.
- `Dataset::from_csv`, `Ridge` and `KnnRegressor`, `Metrics`, `cross_validate`.
- `TrpModel` for TRPA1 / TRPV1 with an applicability-domain flag and a small
  illustrative `demo_set`.
