//! Density-matrix representation and Kraus-channel noise application.
//!
//! [`DensityMatrix`] models an `n`-qubit *mixed* state as a `2^n × 2^n`
//! complex Hermitian, trace-1, positive-semidefinite matrix `ρ`, alongside
//! (not replacing) the pure-state [`crate::State`] vector representation.
//! Mixed states arise from classical uncertainty about the prepared pure
//! state, or from tracing out an environment — most commonly from applying
//! **noise channels**, which is the main reason to reach for this module.
//!
//! ## Representation
//!
//! Internally `ρ` is stored as the real `2·2^n × 2·2^n` block embedding used
//! throughout this crate (see [`crate::tensor`]): each complex entry
//! `a + bi` becomes the real `2×2` block `[[a, -b], [b, a]]`. This lets gate
//! application and Kraus-channel application reuse ordinary real matrix
//! multiplication (`tpt-math-linalg`'s `DMatrix<f64>`) instead of re-deriving
//! complex arithmetic, and it reuses [`crate::tensor::Circuit`]'s
//! Kronecker-product unitary assembly directly for gates. A convenient
//! identity of the embedding — `embed(A†) == embed(A).transpose()` — is what
//! lets both unitary conjugation `ρ ↦ U ρ U†` and the general Kraus map
//! `ρ ↦ Σ_k K_k ρ K_k†` be implemented as plain real matrix products.
//!
//! Because storage is `O(4^n)` (`2^n × 2^n` complex, or equivalently
//! `2^(n+1) × 2^(n+1)` real), density-matrix simulation is practical only for
//! a much smaller qubit count than the pure-state [`crate::State`] path
//! (which stores just `O(2^n)` amplitudes and supports up to 20 qubits) — a
//! dozen or so qubits is already multiple gigabytes of `f64` storage for the
//! dense matrix.
//!
//! ## Examples
//!
//! Build a Bell state as a density matrix and confirm it is the outer
//! product of the pure state with itself:
//!
//! ```
//! use tpt_sci_quantum::{State, DensityMatrix};
//!
//! let mut state = State::new(2).unwrap();
//! state.h(0).unwrap();
//! state.cnot(0, 1).unwrap();
//!
//! let rho = DensityMatrix::from_pure_state(&state);
//! assert!((rho.trace().re - 1.0).abs() < 1e-9);
//! let p = rho.probabilities();
//! assert!((p[0] - 0.5).abs() < 1e-9);
//! assert!((p[3] - 0.5).abs() < 1e-9);
//! ```
//!
//! Apply single-qubit bit-flip noise to one qubit of a two-qubit register:
//!
//! ```
//! use tpt_sci_quantum::{State, DensityMatrix};
//!
//! let state = State::new(2).unwrap(); // |00>
//! let mut rho = DensityMatrix::from_pure_state(&state);
//! let kraus = DensityMatrix::bit_flip_kraus(2, 0, 1.0).unwrap(); // deterministic flip
//! rho.apply_kraus(&kraus).unwrap();
//! let p = rho.probabilities();
//! assert!((p[1] - 1.0).abs() < 1e-9); // |01> (qubit 0 flipped)
//! ```

use num_complex::Complex;
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

use crate::tensor::{self, Circuit};
use crate::{State, StateError};

/// Identity gate, `I = [[1, 0], [0, 1]]`, used internally by the noise-channel
/// constructors.
const IDENTITY: [Complex<f64>; 4] = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0),
];

/// A dense complex square matrix, stored row-major.
///
/// Used to pass Kraus operators to [`DensityMatrix::apply_kraus`]. Most
/// callers will not build one by hand: [`DensityMatrix::bit_flip_kraus`] and
/// [`DensityMatrix::depolarizing_kraus`] construct ready-made single-qubit
/// noise channels, already embedded into the full `n`-qubit space.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    dim: usize,
    data: Vec<Complex<f64>>,
}

impl Matrix {
    /// Build a `dim × dim` matrix from `data` in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::MatrixSizeMismatch`] if `data.len() != dim * dim`.
    pub fn from_row_major(dim: usize, data: Vec<Complex<f64>>) -> Result<Self, StateError> {
        if data.len() != dim * dim {
            return Err(StateError::MatrixSizeMismatch {
                expected: dim * dim,
                got: data.len(),
            });
        }
        Ok(Matrix { dim, data })
    }

    /// The row/column count of this (square) matrix.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The entry at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= dim()` or `col >= dim()`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Complex<f64> {
        self.data[row * self.dim + col]
    }

    /// The raw row-major entries.
    #[must_use]
    pub fn data(&self) -> &[Complex<f64>] {
        &self.data
    }
}

/// Embed a single-qubit `gate`, scaled by the real `coeff`, into the full
/// `n`-qubit space acting on `qubit` (identity elsewhere), returning it as a
/// [`Matrix`]. Reuses [`tensor::expand_single`] (the same Kronecker-product
/// assembly `Circuit::unitary` uses for single-qubit gates) and then
/// un-embeds the result back into an ordinary complex [`Matrix`].
fn embed_single_qubit_kraus(
    n: usize,
    qubit: usize,
    gate: &[Complex<f64>; 4],
    coeff: f64,
) -> Matrix {
    let scaled = [
        gate[0].scale(coeff),
        gate[1].scale(coeff),
        gate[2].scale(coeff),
        gate[3].scale(coeff),
    ];
    let real = tensor::expand_single(n, qubit, &scaled);
    let d = 1usize << n;
    let mut data = vec![Complex::new(0.0, 0.0); d * d];
    for (i, row) in data.chunks_mut(d).enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = tensor::unembed_entry(&real, i, j);
        }
    }
    Matrix { dim: d, data }
}

/// An `n`-qubit mixed state stored as a `2^n × 2^n` density matrix `ρ`.
///
/// See the [module docs](self) for the storage representation and its
/// `O(4^n)` memory cost.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix {
    n: usize,
    /// Real `2d × 2d` block embedding of the complex `d × d` density matrix
    /// (`d = 2^n`); see [`tensor::embed_complex_square`].
    real: DMatrix<f64>,
}

impl DensityMatrix {
    /// The number of qubits represented by this density matrix.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n
    }

    /// The Hilbert-space dimension `2^n`.
    #[must_use]
    pub fn dim(&self) -> usize {
        1usize << self.n
    }

    /// Build the density matrix `ρ = |ψ⟩⟨ψ|` of a pure [`State`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_quantum::{State, DensityMatrix};
    /// let mut s = State::new(1).unwrap();
    /// s.h(0).unwrap();
    /// let rho = DensityMatrix::from_pure_state(&s);
    /// assert!((rho.entry(0, 0).re - 0.5).abs() < 1e-9);
    /// assert!((rho.entry(0, 1).re - 0.5).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn from_pure_state(state: &State) -> Self {
        let n = state.n_qubits();
        let d = 1usize << n;
        let amps = state.amplitudes();
        let mut entries = vec![Complex::new(0.0, 0.0); d * d];
        for i in 0..d {
            for j in 0..d {
                entries[i * d + j] = amps[i] * amps[j].conj();
            }
        }
        let real = tensor::embed_complex_square(&entries, d);
        DensityMatrix { n, real }
    }

    /// Build a classical mixture `ρ = Σ_k p_k |ψ_k⟩⟨ψ_k|` of pure states with
    /// probabilities `probs`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::EmptyMixture`] if `states` is empty,
    /// [`StateError::MixtureSizeMismatch`] if `states.len() != probs.len()`,
    /// [`StateError::QubitCountMismatch`] if the states have differing qubit
    /// counts, [`StateError::NegativeProbability`] if any probability is
    /// negative, or [`StateError::ProbabilitiesNotNormalized`] if the
    /// probabilities do not sum to `1` (within `1e-6`).
    pub fn from_mixture(states: &[State], probs: &[f64]) -> Result<Self, StateError> {
        if states.is_empty() {
            return Err(StateError::EmptyMixture);
        }
        if states.len() != probs.len() {
            return Err(StateError::MixtureSizeMismatch {
                states: states.len(),
                probs: probs.len(),
            });
        }
        if probs.iter().any(|&p| p < 0.0) {
            return Err(StateError::NegativeProbability);
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(StateError::ProbabilitiesNotNormalized(sum));
        }
        let n = states[0].n_qubits();
        for s in states {
            if s.n_qubits() != n {
                return Err(StateError::QubitCountMismatch {
                    expected: n,
                    got: s.n_qubits(),
                });
            }
        }
        let size = (1usize << n) * 2;
        let mut real = DMatrix::from_fn(size, size, |_, _| 0.0);
        for (s, &p) in states.iter().zip(probs) {
            let component = Self::from_pure_state(s).real * p;
            real = real + component;
        }
        Ok(DensityMatrix { n, real })
    }

    /// The complex entry `ρ[(i, j)]`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= dim()` or `j >= dim()`.
    #[must_use]
    pub fn entry(&self, i: usize, j: usize) -> Complex<f64> {
        tensor::unembed_entry(&self.real, i, j)
    }

    /// The trace `Tr(ρ) = Σ_i ρ[(i, i)]`.
    ///
    /// Equal to `1` (up to floating-point rounding) for any physically valid
    /// density matrix; useful as a validity check after gate/channel
    /// application.
    #[must_use]
    pub fn trace(&self) -> Complex<f64> {
        let d = self.dim();
        (0..d).map(|i| self.entry(i, i)).sum()
    }

    /// The measurement probabilities `ρ[(i, i)]` for each computational basis
    /// state (the diagonal of `ρ`, which is real and non-negative for a
    /// physical density matrix).
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        let d = self.dim();
        (0..d).map(|i| self.entry(i, i).re).collect()
    }

    /// Expectation value `⟨Z^{⊗n}⟩ = Σ_i (-1)^{popcount(i)} ρ[(i, i)]` of the
    /// all-qubit parity operator, analogous to [`State::expectation_z`].
    #[must_use]
    pub fn expectation_z(&self) -> f64 {
        let d = self.dim();
        (0..d)
            .map(|i| {
                let sign = if i.count_ones() & 1 == 0 { 1.0 } else { -1.0 };
                sign * self.entry(i, i).re
            })
            .sum()
    }

    /// Apply a real-embedded unitary `u ρ u^T` (`u^T` stands in for `u†`
    /// under the real embedding) built by [`Circuit::unitary`].
    fn conjugate_by(&mut self, u: &DMatrix<f64>) {
        let ut = u.transpose();
        self.real = u.clone() * self.real.clone() * ut;
    }

    /// Apply a single-qubit `gate` to `qubit` via unitary conjugation
    /// `ρ ↦ U ρ U†`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit >= n_qubits()`.
    pub fn apply_gate(&mut self, gate: &[Complex<f64>; 4], qubit: usize) -> Result<(), StateError> {
        let n = self.n;
        if qubit >= n {
            return Err(StateError::InvalidQubit { qubit, n });
        }
        let mut c = Circuit::new(n);
        c.gate(qubit, *gate);
        self.conjugate_by(&c.unitary());
        Ok(())
    }

    /// Apply a controlled-NOT: flip `target` whenever `control` is set.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if either qubit is out of range,
    /// or [`StateError::SameQubits`] if `control == target`.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), StateError> {
        let n = self.n;
        if control >= n {
            return Err(StateError::InvalidQubit { qubit: control, n });
        }
        if target >= n {
            return Err(StateError::InvalidQubit { qubit: target, n });
        }
        if control == target {
            return Err(StateError::SameQubits(control));
        }
        let mut c = Circuit::new(n);
        c.cnot(control, target);
        self.conjugate_by(&c.unitary());
        Ok(())
    }

    /// Apply the Hadamard gate `H` to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn h(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_gate(&crate::H, qubit)
    }

    /// Apply the Pauli-X gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn x(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_gate(&crate::X, qubit)
    }

    /// Apply the Pauli-Y gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn y(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_gate(&crate::Y, qubit)
    }

    /// Apply the Pauli-Z gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn z(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_gate(&crate::Z, qubit)
    }

    /// Apply a set of Kraus operators, `ρ ↦ Σ_k K_k ρ K_k†`.
    ///
    /// `kraus_ops` must satisfy `Σ_k K_k† K_k = I` for the result to remain a
    /// physically valid (trace-1) density matrix; this is not checked here
    /// (validate with [`DensityMatrix::trace`] after the call if needed). Each
    /// operator must be `dim() × dim()`, i.e. already embedded into the full
    /// `n`-qubit space — see [`DensityMatrix::bit_flip_kraus`] and
    /// [`DensityMatrix::depolarizing_kraus`] for ready-made single-qubit
    /// channels.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::MatrixSizeMismatch`] if any operator is not
    /// `dim() × dim()`.
    pub fn apply_kraus(&mut self, kraus_ops: &[Matrix]) -> Result<(), StateError> {
        let d = self.dim();
        for k in kraus_ops {
            if k.dim() != d {
                return Err(StateError::MatrixSizeMismatch {
                    expected: d,
                    got: k.dim(),
                });
            }
        }
        let size = self.real.nrows();
        let mut acc = DMatrix::from_fn(size, size, |_, _| 0.0);
        for k in kraus_ops {
            let k_real = tensor::embed_complex_square(k.data(), d);
            let kt = k_real.transpose();
            let term = k_real * self.real.clone() * kt;
            acc = acc + term;
        }
        self.real = acc;
        Ok(())
    }

    /// Kraus operators for the single-qubit **bit-flip channel** on `qubit`
    /// within an `n`-qubit register: with probability `p` the qubit is
    /// flipped by `X`, and with probability `1 - p` it is left alone.
    ///
    /// `K_0 = √(1-p)·I`, `K_1 = √p·X`, each embedded into the full `n`-qubit
    /// space (identity on all other qubits).
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit >= n`, or
    /// [`StateError::InvalidProbability`] if `p` is outside `[0, 1]`.
    pub fn bit_flip_kraus(n: usize, qubit: usize, p: f64) -> Result<Vec<Matrix>, StateError> {
        if qubit >= n {
            return Err(StateError::InvalidQubit { qubit, n });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(StateError::InvalidProbability(p));
        }
        let k0 = embed_single_qubit_kraus(n, qubit, &IDENTITY, (1.0 - p).sqrt());
        let k1 = embed_single_qubit_kraus(n, qubit, &crate::X, p.sqrt());
        Ok(vec![k0, k1])
    }

    /// Kraus operators for the single-qubit **depolarizing channel** on
    /// `qubit` within an `n`-qubit register: `ρ ↦ (1-p)·ρ + p·I/2` (the qubit
    /// is replaced by the maximally mixed state with probability `p`).
    ///
    /// `K_0 = √(1 - 3p/4)·I`, `K_1 = √(p/4)·X`, `K_2 = √(p/4)·Y`,
    /// `K_3 = √(p/4)·Z`, each embedded into the full `n`-qubit space
    /// (identity on all other qubits). At `p = 0` this is the identity
    /// channel; at `p = 1` it fully depolarizes `qubit` to the maximally
    /// mixed state regardless of the input.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit >= n`, or
    /// [`StateError::InvalidProbability`] if `p` is outside `[0, 1]`.
    pub fn depolarizing_kraus(n: usize, qubit: usize, p: f64) -> Result<Vec<Matrix>, StateError> {
        if qubit >= n {
            return Err(StateError::InvalidQubit { qubit, n });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(StateError::InvalidProbability(p));
        }
        let c0 = (1.0 - 0.75 * p).sqrt();
        let c = (p / 4.0).sqrt();
        let k0 = embed_single_qubit_kraus(n, qubit, &IDENTITY, c0);
        let k1 = embed_single_qubit_kraus(n, qubit, &crate::X, c);
        let k2 = embed_single_qubit_kraus(n, qubit, &crate::Y, c);
        let k3 = embed_single_qubit_kraus(n, qubit, &crate::Z, c);
        Ok(vec![k0, k1, k2, k3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const TOL: f64 = 1e-9;

    #[test]
    fn from_pure_state_matches_outer_product() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        let amps = s.amplitudes().to_vec();

        let rho = DensityMatrix::from_pure_state(&s);
        for i in 0..4 {
            for j in 0..4 {
                let expected = amps[i] * amps[j].conj();
                let got = rho.entry(i, j);
                assert_abs_diff_eq!(got.re, expected.re, epsilon = TOL);
                assert_abs_diff_eq!(got.im, expected.im, epsilon = TOL);
            }
        }
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);
        assert_abs_diff_eq!(rho.trace().im, 0.0, epsilon = TOL);
    }

    #[test]
    fn trace_stays_one_after_gate_application() {
        let s = State::new(2).unwrap();
        let mut rho = DensityMatrix::from_pure_state(&s);
        rho.h(0).unwrap();
        rho.apply_cnot(0, 1).unwrap();
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);
        assert_abs_diff_eq!(rho.trace().im, 0.0, epsilon = TOL);
    }

    #[test]
    fn trace_stays_one_after_kraus_application() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        let mut rho = DensityMatrix::from_pure_state(&s);

        let bit_flip = DensityMatrix::bit_flip_kraus(2, 0, 0.3).unwrap();
        rho.apply_kraus(&bit_flip).unwrap();
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);

        let depolarizing = DensityMatrix::depolarizing_kraus(2, 1, 0.4).unwrap();
        rho.apply_kraus(&depolarizing).unwrap();
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);
    }

    #[test]
    fn depolarizing_p0_is_identity() {
        let mut s = State::new(1).unwrap();
        s.h(0).unwrap();
        let rho_before = DensityMatrix::from_pure_state(&s);
        let mut rho = rho_before.clone();

        let kraus = DensityMatrix::depolarizing_kraus(1, 0, 0.0).unwrap();
        rho.apply_kraus(&kraus).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let a = rho.entry(i, j);
                let b = rho_before.entry(i, j);
                assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
                assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
            }
        }
    }

    #[test]
    fn depolarizing_p1_is_maximally_mixed() {
        // Start from a pure (non-mixed) state so full depolarization is a
        // real change, not a no-op.
        let mut s = State::new(1).unwrap();
        s.h(0).unwrap();
        let mut rho = DensityMatrix::from_pure_state(&s);

        let kraus = DensityMatrix::depolarizing_kraus(1, 0, 1.0).unwrap();
        rho.apply_kraus(&kraus).unwrap();

        assert_abs_diff_eq!(rho.entry(0, 0).re, 0.5, epsilon = TOL);
        assert_abs_diff_eq!(rho.entry(1, 1).re, 0.5, epsilon = TOL);
        assert_abs_diff_eq!(rho.entry(0, 1).norm(), 0.0, epsilon = TOL);
        assert_abs_diff_eq!(rho.entry(1, 0).norm(), 0.0, epsilon = TOL);
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);
    }

    #[test]
    fn bit_flip_p1_matches_x_gate() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();

        // Density-matrix path: bit-flip channel with p=1 on qubit 0.
        let mut rho = DensityMatrix::from_pure_state(&s);
        let kraus = DensityMatrix::bit_flip_kraus(2, 0, 1.0).unwrap();
        rho.apply_kraus(&kraus).unwrap();

        // Reference: apply an X gate directly to the same pure state.
        let mut s_flipped = s.clone();
        s_flipped.x(0).unwrap();
        let rho_expected = DensityMatrix::from_pure_state(&s_flipped);

        for i in 0..4 {
            for j in 0..4 {
                let a = rho.entry(i, j);
                let b = rho_expected.entry(i, j);
                assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
                assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
            }
        }
    }

    #[test]
    fn gate_application_matches_pure_state_path() {
        // Cross-check: applying gates to a DensityMatrix built from a pure
        // state (no noise) gives the same measurement statistics as applying
        // the same gates directly to the State.
        let mut s = State::new(3).unwrap();
        s.h(0).unwrap();

        let mut rho = DensityMatrix::from_pure_state(&State::new(3).unwrap());
        rho.h(0).unwrap();

        s.cnot(0, 1).unwrap();
        rho.apply_cnot(0, 1).unwrap();
        s.cnot(1, 2).unwrap();
        rho.apply_cnot(1, 2).unwrap();

        let p_state = s.probabilities();
        let p_rho = rho.probabilities();
        for (a, b) in p_state.iter().zip(p_rho.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = TOL);
        }
        assert_abs_diff_eq!(s.expectation_z(), rho.expectation_z(), epsilon = TOL);
    }

    #[test]
    fn from_mixture_of_one_state_matches_pure() {
        let mut s = State::new(1).unwrap();
        s.h(0).unwrap();
        let rho_pure = DensityMatrix::from_pure_state(&s);
        let rho_mix = DensityMatrix::from_mixture(&[s], &[1.0]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let a = rho_pure.entry(i, j);
                let b = rho_mix.entry(i, j);
                assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
                assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
            }
        }
    }

    #[test]
    fn from_mixture_error_paths() {
        let s0 = State::new(1).unwrap();
        let mut s1 = State::new(1).unwrap();
        s1.x(0).unwrap();

        assert_eq!(
            DensityMatrix::from_mixture(&[], &[]),
            Err(StateError::EmptyMixture)
        );
        assert_eq!(
            DensityMatrix::from_mixture(std::slice::from_ref(&s0), &[0.5, 0.5]),
            Err(StateError::MixtureSizeMismatch {
                states: 1,
                probs: 2
            })
        );
        assert_eq!(
            DensityMatrix::from_mixture(&[s0.clone(), s1.clone()], &[0.5, 0.6]),
            Err(StateError::ProbabilitiesNotNormalized(1.1))
        );
        assert_eq!(
            DensityMatrix::from_mixture(&[s0, s1], &[-0.5, 1.5]),
            Err(StateError::NegativeProbability)
        );
    }

    #[test]
    fn maximally_mixed_two_state_average() {
        let s0 = State::new(1).unwrap(); // |0>
        let mut s1 = State::new(1).unwrap();
        s1.x(0).unwrap(); // |1>
        let rho = DensityMatrix::from_mixture(&[s0, s1], &[0.5, 0.5]).unwrap();
        assert_abs_diff_eq!(rho.entry(0, 0).re, 0.5, epsilon = TOL);
        assert_abs_diff_eq!(rho.entry(1, 1).re, 0.5, epsilon = TOL);
        assert_abs_diff_eq!(rho.entry(0, 1).norm(), 0.0, epsilon = TOL);
        assert_abs_diff_eq!(rho.trace().re, 1.0, epsilon = TOL);
    }

    #[test]
    fn invalid_qubit_and_probability_errors() {
        let s = State::new(2).unwrap();
        let mut rho = DensityMatrix::from_pure_state(&s);
        assert_eq!(rho.h(5), Err(StateError::InvalidQubit { qubit: 5, n: 2 }));
        assert_eq!(rho.apply_cnot(1, 1), Err(StateError::SameQubits(1)));
        assert_eq!(
            DensityMatrix::bit_flip_kraus(2, 0, 1.5),
            Err(StateError::InvalidProbability(1.5))
        );
        assert_eq!(
            DensityMatrix::depolarizing_kraus(2, 5, 0.1),
            Err(StateError::InvalidQubit { qubit: 5, n: 2 })
        );
    }

    #[test]
    fn matrix_from_row_major_rejects_wrong_length() {
        let data = vec![Complex::new(1.0, 0.0); 3];
        assert!(Matrix::from_row_major(2, data).is_err());
    }
}
