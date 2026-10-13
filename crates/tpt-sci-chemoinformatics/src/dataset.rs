//! Labelled datasets: CSV loading and featurisation.

use crate::descriptors::DescriptorSet;
use crate::error::{ChemoinformaticsError, Result};
use crate::fingerprint::Fingerprint;
use crate::mol::Molecule;

/// One labelled compound.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// Optional human-readable name.
    pub name: Option<String>,
    /// SMILES string.
    pub smiles: String,
    /// Activity label (e.g. pEC50 / pIC50) as supplied by the user.
    pub target: f64,
}

/// A labelled dataset of compounds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dataset {
    /// The records, in input order.
    pub records: Vec<Record>,
}

/// Descriptor matrix, fingerprints and labels for a [`Dataset`].
#[derive(Debug, Clone)]
pub struct Featurized {
    /// One descriptor row per record.
    pub x: Vec<Vec<f64>>,
    /// One fingerprint per record.
    pub fingerprints: Vec<Fingerprint>,
    /// Labels.
    pub y: Vec<f64>,
}

impl Dataset {
    /// Build a dataset from records.
    pub fn new(records: Vec<Record>) -> Self {
        Dataset { records }
    }

    /// Parse CSV text with a header row containing `smiles` and `target`
    /// columns and an optional `name` column (any order, case-insensitive).
    /// Blank lines and lines starting with `#` are skipped. Fields must not be
    /// quoted or contain commas (SMILES never do).
    ///
    /// # Errors
    /// [`ChemoinformaticsError::Dataset`] for a missing column, short row or
    /// non-numeric / non-finite target.
    pub fn from_csv(text: &str) -> Result<Dataset> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.trim()))
            .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));
        let (_, header) = lines
            .next()
            .ok_or_else(|| ChemoinformaticsError::Dataset("empty CSV".into()))?;
        let cols: Vec<String> = header.split(',').map(|c| c.trim().to_lowercase()).collect();
        let find = |name: &str| cols.iter().position(|c| c == name);
        let si = find("smiles")
            .ok_or_else(|| ChemoinformaticsError::Dataset("missing 'smiles' column".into()))?;
        let ti = find("target")
            .ok_or_else(|| ChemoinformaticsError::Dataset("missing 'target' column".into()))?;
        let ni = find("name");
        let mut records = Vec::new();
        for (line_no, line) in lines {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < cols.len() {
                return Err(ChemoinformaticsError::Dataset(format!(
                    "line {line_no}: expected {} fields, found {}",
                    cols.len(),
                    f.len()
                )));
            }
            let target: f64 = f[ti].parse().map_err(|_| {
                ChemoinformaticsError::Dataset(format!(
                    "line {line_no}: target '{}' is not a number",
                    f[ti]
                ))
            })?;
            if !target.is_finite() {
                return Err(ChemoinformaticsError::Dataset(format!(
                    "line {line_no}: target is not finite"
                )));
            }
            records.push(Record {
                name: ni.map(|i| f[i].to_string()).filter(|s| !s.is_empty()),
                smiles: f[si].to_string(),
                target,
            });
        }
        if records.is_empty() {
            return Err(ChemoinformaticsError::Dataset("no data rows".into()));
        }
        Ok(Dataset { records })
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the dataset has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Parse every SMILES and compute descriptors and fingerprints.
    ///
    /// # Errors
    /// Propagates the first SMILES parse error.
    pub fn featurize(&self) -> Result<Featurized> {
        let mut out = Featurized {
            x: Vec::new(),
            fingerprints: Vec::new(),
            y: Vec::new(),
        };
        for r in &self.records {
            let mol = Molecule::from_smiles(&r.smiles)?;
            out.x.push(DescriptorSet::compute(&mol).as_slice().to_vec());
            out.fingerprints.push(Fingerprint::morgan(&mol, 2));
            out.y.push(r.target);
        }
        Ok(out)
    }
}
