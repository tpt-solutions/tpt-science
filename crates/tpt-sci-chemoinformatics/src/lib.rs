//! # tpt-sci-chemoinformatics
//!
//! A from-scratch QSPR (quantitative structure–property relationship) pipeline:
//!
//! 1. [`smiles`] / [`mol`] — parse SMILES into a [`Molecule`] graph (implicit
//!    hydrogens, rings, aromaticity).
//! 2. [`descriptors`] — physico-chemical descriptors (MW, TPSA, logP, H-bond
//!    counts, rotatable bonds, polyene/amide/phenol counts, ...).
//! 3. [`fingerprint`] — Morgan/ECFP-style fingerprints and Tanimoto similarity.
//! 4. [`dataset`] / [`model`] — CSV datasets, ridge and kNN regressors,
//!    cross-validation.
//! 5. [`trp`] — a TRPA1 / TRPV1 activity model (the "bite"/"tingle" receptors,
//!    independent of acidity) with an applicability-domain flag.
//!
//! Predictions are only as good as the labelled data you provide. The bundled
//! [`trp::demo_set`] is a tiny illustrative set, not a validated predictor.
//!
//! ```
//! use tpt_sci_chemoinformatics::{DescriptorSet, Molecule};
//!
//! // Hydroxy-α-sanshool, the tingling amide of Sichuan pepper.
//! let mol = Molecule::from_smiles("C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O").unwrap();
//! assert_eq!(mol.formula(), "C16H25NO2");
//! let d = DescriptorSet::compute(&mol);
//! assert_eq!(d.get("amides"), Some(1.0));
//! assert_eq!(d.get("cc_double_bonds"), Some(4.0));
//! ```

#![forbid(unsafe_code)]

pub mod dataset;
pub mod descriptors;
pub mod elements;
mod error;
pub mod fingerprint;
pub mod model;
pub mod mol;
pub mod smiles;
pub mod trp;

pub use dataset::{Dataset, Featurized, Record};
pub use descriptors::{DESCRIPTOR_NAMES, DescriptorSet};
pub use error::{ChemoinformaticsError, Result};
pub use fingerprint::Fingerprint;
pub use model::{KnnRegressor, Metrics, Ridge, cross_validate};
pub use mol::{Atom, Bond, Molecule};
pub use trp::{Prediction, Target, TrpConfig, TrpModel};
