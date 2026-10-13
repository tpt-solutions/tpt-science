//! TRPA1 / TRPV1 activity QSPR.
//!
//! TRPA1 and TRPV1 are the sensory-neuron ion channels behind pungency and
//! tingle (capsaicin, allyl isothiocyanate, the sanshools of Sichuan pepper).
//! Their activation is a receptor-binding/gating effect, not an acidity effect,
//! so **no pH or pKa term is used**: the model sees only structural descriptors
//! and circular fingerprints.
//!
//! [`TrpModel`] combines a ridge regression on [`DescriptorSet`] values with a
//! Tanimoto kNN on [`Fingerprint`]s and reports an applicability-domain flag.
//! Labels are whatever the user supplies (typically pEC50 = -log10(EC50 / M)).
//!
//! # Demo data
//! [`demo_set`] ships a handful of compounds with *approximate, order-of-
//! magnitude* literature potencies so that tests and examples run offline. They
//! are **illustrative only**; consult the primary literature (e.g. Caterina et
//! al. 1997 *Nature*; Jordt et al. 2004 *Nature*; Bandell et al. 2004 *Neuron*;
//! Koo et al. 2007 *J Neurophysiol*; Bautista et al. 2008 *Nat Neurosci*;
//! McNamara et al. 2005 *Br J Pharmacol*) and supply a properly curated dataset
//! (e.g. exported from ChEMBL under its licence) for any real use.

use crate::dataset::{Dataset, Record};
use crate::descriptors::DescriptorSet;
use crate::error::Result;
use crate::fingerprint::Fingerprint;
use crate::model::{KnnRegressor, Metrics, Ridge, cross_validate};
use crate::mol::Molecule;

/// Which receptor a model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Transient receptor potential ankyrin 1 (electrophile / "tingle" sensor).
    Trpa1,
    /// Transient receptor potential vanilloid 1 (capsaicin / heat sensor).
    Trpv1,
}

/// Hyper-parameters for [`TrpModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrpConfig {
    /// Ridge L2 penalty on standardised descriptors.
    pub lambda: f64,
    /// Number of neighbours for the kNN component.
    pub k: usize,
    /// Minimum Tanimoto similarity to the training set for a prediction to be
    /// flagged in-domain.
    pub domain_threshold: f64,
}

impl Default for TrpConfig {
    fn default() -> Self {
        TrpConfig {
            lambda: 1.0,
            k: 3,
            domain_threshold: 0.25,
        }
    }
}

/// A single prediction with its provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Ridge-on-descriptors estimate.
    pub ridge: f64,
    /// Tanimoto-kNN estimate.
    pub knn: f64,
    /// Mean of the two estimates.
    pub consensus: f64,
    /// Highest Tanimoto similarity to any training compound.
    pub max_similarity: f64,
    /// Whether `max_similarity` meets the applicability-domain threshold.
    pub in_domain: bool,
}

/// A fitted TRPA1 or TRPV1 activity model.
#[derive(Debug, Clone, PartialEq)]
pub struct TrpModel {
    target: Target,
    ridge: Ridge,
    knn: KnnRegressor,
    config: TrpConfig,
}

impl TrpModel {
    /// Fit on labelled records.
    ///
    /// # Errors
    /// On SMILES parse failures, empty data or invalid hyper-parameters.
    pub fn fit(target: Target, data: &Dataset, config: &TrpConfig) -> Result<TrpModel> {
        let f = data.featurize()?;
        Ok(TrpModel {
            target,
            ridge: Ridge::fit(&f.x, &f.y, config.lambda)?,
            knn: KnnRegressor::fit(&f.fingerprints, &f.y, config.k)?,
            config: *config,
        })
    }

    /// The receptor this model was fit for.
    pub fn target(&self) -> Target {
        self.target
    }

    /// Predict for an already-parsed molecule.
    ///
    /// # Errors
    /// Never fails for a model produced by [`TrpModel::fit`]; the `Result` keeps
    /// the signature stable.
    pub fn predict(&self, mol: &Molecule) -> Result<Prediction> {
        let ridge = self.ridge.predict(DescriptorSet::compute(mol).as_slice())?;
        let (knn, max_similarity) = self.knn.predict(&Fingerprint::morgan(mol, 2));
        Ok(Prediction {
            ridge,
            knn,
            consensus: 0.5 * (ridge + knn),
            max_similarity,
            in_domain: max_similarity >= self.config.domain_threshold,
        })
    }

    /// Parse `smiles` and predict.
    ///
    /// # Errors
    /// On a SMILES parse failure.
    pub fn predict_smiles(&self, smiles: &str) -> Result<Prediction> {
        self.predict(&Molecule::from_smiles(smiles)?)
    }

    /// Cross-validated metrics of the consensus prediction (`folds == 0` for
    /// leave-one-out). Deterministic for a given `seed`.
    ///
    /// # Errors
    /// On parse failures or datasets with fewer than two compounds.
    pub fn cross_validate(
        data: &Dataset,
        config: &TrpConfig,
        folds: usize,
        seed: u64,
    ) -> Result<Metrics> {
        let f = data.featurize()?;
        cross_validate(&f.y, folds, seed, |train, test| {
            let x: Vec<Vec<f64>> = train.iter().map(|&i| f.x[i].clone()).collect();
            let fps: Vec<Fingerprint> = train.iter().map(|&i| f.fingerprints[i].clone()).collect();
            let y: Vec<f64> = train.iter().map(|&i| f.y[i]).collect();
            let ridge = Ridge::fit(&x, &y, config.lambda)?;
            let knn = KnnRegressor::fit(&fps, &y, config.k)?;
            test.iter()
                .map(|&i| Ok(0.5 * (ridge.predict(&f.x[i])? + knn.predict(&f.fingerprints[i]).0)))
                .collect()
        })
    }
}

fn rec(name: &str, smiles: &str, target: f64) -> Record {
    Record {
        name: Some(name.to_string()),
        smiles: smiles.to_string(),
        target,
    }
}

/// Small illustrative labelled set (approximate pEC50 values; see the module
/// docs — **not** a validated training set).
pub fn demo_set(target: Target) -> Dataset {
    Dataset::new(match target {
        Target::Trpv1 => vec![
            rec("capsaicin", "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O", 6.3),
            rec("dihydrocapsaicin", "COc1cc(CNC(=O)CCCCCCC(C)C)ccc1O", 6.2),
            rec(
                "hydroxy-alpha-sanshool",
                "C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O",
                6.0,
            ),
            rec("piperine", "O=C(/C=C/C=C/c1ccc2OCOc2c1)N1CCCCC1", 4.4),
        ],
        Target::Trpa1 => vec![
            rec("allyl isothiocyanate", "C=CCN=C=S", 5.0),
            rec("cinnamaldehyde", "O=C/C=C/c1ccccc1", 4.2),
            rec("acrolein", "C=CC=O", 5.3),
            rec("menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", 4.2),
        ],
    })
}

/// SMILES of botanical compounds useful for screening against a fitted model.
pub fn botanical_smiles() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "hydroxy-alpha-sanshool",
            "C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O",
        ),
        (
            "hydroxy-beta-sanshool",
            "C/C=C/C=C/C=C/CC/C=C/C(=O)NCC(C)(C)O",
        ),
        (
            "EGCG",
            "Oc1cc(O)c2C[C@@H](OC(=O)c3cc(O)c(O)c(O)c3)[C@H](Oc2c1)c1cc(O)c(O)c(O)c1",
        ),
        (
            "epicatechin",
            "Oc1cc(O)c2C[C@@H](O)[C@H](Oc2c1)c1ccc(O)c(O)c1",
        ),
        ("gallic acid", "OC(=O)c1cc(O)c(O)c(O)c1"),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ]
}
