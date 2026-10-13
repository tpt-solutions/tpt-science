use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ChemoinformaticsError>;

/// Errors raised while parsing molecules, computing descriptors, loading
/// datasets, or fitting/evaluating models.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ChemoinformaticsError {
    /// The SMILES string is malformed (unbalanced branch, dangling bond, ...).
    #[error("invalid SMILES at position {pos}: {msg}")]
    Smiles {
        /// Zero-based byte offset of the problem.
        pos: usize,
        /// Human-readable description.
        msg: String,
    },

    /// An element symbol is not in the supported table.
    #[error("unknown or unsupported element symbol: {0}")]
    UnknownElement(String),

    /// An atom in the organic subset exceeds its maximum valence.
    #[error("atom {index} ({symbol}) has valence {valence}, above its maximum")]
    Valence {
        /// Atom index in input order.
        index: usize,
        /// Element symbol.
        symbol: String,
        /// Bond-order sum found.
        valence: u32,
    },

    /// Valid SMILES syntax that this crate does not (yet) interpret.
    #[error("unsupported SMILES syntax: {0}")]
    UnsupportedSyntax(String),

    /// A dataset (CSV) could not be read.
    #[error("invalid dataset: {0}")]
    Dataset(String),

    /// A model could not be fit or evaluated.
    #[error("model error: {0}")]
    Model(String),

    /// A hyper-parameter was out of range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
