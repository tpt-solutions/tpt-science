//! Errors produced by state construction and gate application.

use thiserror::Error;

/// Errors produced by state construction and gate application.
///
/// Note: `Eq` is intentionally not derived (some variants carry an `f64`,
/// which has no total equality), only `PartialEq`.
#[derive(Debug, Error, PartialEq)]
pub enum StateError {
    /// The requested number of qubits exceeds the supported limit.
    #[error("cannot simulate {0} qubits: state vector would be too large (max 20)")]
    TooManyQubits(usize),

    /// A gate referenced a qubit index that does not exist.
    #[error("qubit index {qubit} is invalid for a {n}-qubit state")]
    InvalidQubit {
        /// The offending qubit index.
        qubit: usize,
        /// The number of qubits in the state.
        n: usize,
    },

    /// A two-qubit gate was given the same qubit for control and target.
    #[error("control and target must differ, but both were {0}")]
    SameQubits(usize),

    /// A unitary matrix supplied to [`crate::State::apply_unitary`] had the wrong
    /// dimension for this state's qubit count.
    #[error(
        "unitary must be {expected}x{expected} (2^(n+1) for an n-qubit state), but got {got}x{got}"
    )]
    UnitarySizeMismatch {
        /// The required row/column count.
        expected: usize,
        /// The row count that was actually supplied.
        got: usize,
    },

    /// A flat data vector supplied to [`crate::density::Matrix::from_row_major`]
    /// (or a Kraus operator passed to
    /// [`crate::density::DensityMatrix::apply_kraus`]) had the wrong number of
    /// entries/dimension.
    #[error("matrix must be {expected}x{expected}, but got {got}x{got}")]
    MatrixSizeMismatch {
        /// The required row/column count.
        expected: usize,
        /// The row/column count that was actually supplied.
        got: usize,
    },

    /// [`crate::density::DensityMatrix::from_mixture`] was given an empty list
    /// of states.
    #[error("a mixture must contain at least one state")]
    EmptyMixture,

    /// [`crate::density::DensityMatrix::from_mixture`] was given mismatched
    /// numbers of states and probabilities.
    #[error("mixture has {states} states but {probs} probabilities")]
    MixtureSizeMismatch {
        /// The number of states supplied.
        states: usize,
        /// The number of probabilities supplied.
        probs: usize,
    },

    /// [`crate::density::DensityMatrix::from_mixture`] was given states with
    /// differing qubit counts.
    #[error("mixture states must all have {expected} qubits, but found one with {got}")]
    QubitCountMismatch {
        /// The qubit count of the first state in the mixture.
        expected: usize,
        /// The qubit count of the offending state.
        got: usize,
    },

    /// [`crate::density::DensityMatrix::from_mixture`] was given probabilities
    /// that do not sum to `1` (within tolerance).
    #[error("mixture probabilities must sum to 1, but summed to {0}")]
    ProbabilitiesNotNormalized(f64),

    /// [`crate::density::DensityMatrix::from_mixture`] was given a negative
    /// probability.
    #[error("mixture probabilities must be non-negative")]
    NegativeProbability,

    /// A noise-channel constructor (e.g.
    /// [`crate::density::DensityMatrix::bit_flip_kraus`] or
    /// [`crate::density::DensityMatrix::depolarizing_kraus`]) was given an
    /// error probability outside `[0, 1]`.
    #[error("error probability must be in [0, 1], but got {0}")]
    InvalidProbability(f64),
}
