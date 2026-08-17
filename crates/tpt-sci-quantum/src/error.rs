//! Errors produced by state construction and gate application.

use thiserror::Error;

/// Errors produced by state construction and gate application.
#[derive(Debug, Error, PartialEq, Eq)]
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
}
