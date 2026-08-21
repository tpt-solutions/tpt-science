//! # tpt-sci-quantum
//!
//! A from-scratch **qubit state-vector simulator** for the `tpt-science` pillar.
//!
//! The simulator represents an `n`-qubit pure state as a length-`2^n` vector of
//! complex amplitudes (in the computational basis `|0...0>` … `|1...1>`). Gates
//! are applied in place by iterating over the relevant pairs of basis states, and
//! measurement samples a basis index from the Born rule distribution.
//!
//! No external quantum-simulation crates are used: the linear algebra is just
//! complex scalars and explicit index arithmetic over the amplitude vector.
//!
//! ## Examples
//!
//! Build the Bell state `|Φ+⟩ = (|00⟩ + |11⟩)/√2` and check its marginals:
//!
//! ```
//! use tpt_sci_quantum::State;
//!
//! let mut state = State::new(2).unwrap();
//! state.h(0).unwrap();
//! state.cnot(0, 1).unwrap();
//!
//! let p = state.probabilities();
//! assert!((p[0] - 0.5).abs() < 1e-9);
//! assert!((p[3] - 0.5).abs() < 1e-9);
//! ```

use num_complex::Complex;
use std::f64::consts::SQRT_2;
use tpt_math_prob_core::Rng;

/// `1/√2`, used by several gate constants.
const INV_SQRT2: f64 = 1.0 / SQRT_2;

/// Tensor-product (Kronecker) circuit formulation built on `tpt-math-linalg`.
pub mod tensor;
pub use tensor::{Circuit, embed_gate_2x2, kron};

/// Density-matrix representation and Kraus-channel noise application.
pub mod density;
pub use density::{DensityMatrix, Matrix};

/// Errors produced by state construction and gate application.
pub mod error;
pub use error::StateError;

/// A pure `n`-qubit state stored as a length-`2^n` amplitude vector.
///
/// Amplitudes are stored in computational-basis order: `amps[i]` is the
/// amplitude of `|i⟩` where `i` is the integer whose binary little-endian
/// representation encodes the per-qubit basis states.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    amps: Vec<Complex<f64>>,
}

impl State {
    /// Create the all-zero state `|0...0⟩` for `n` qubits.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::TooManyQubits`] if `n == 0` (a zero-qubit state is
    /// meaningless) or `n > 20` (the `2^20` amplitude vector is the supported
    /// upper bound).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_quantum::State;
    /// let s = State::new(3).unwrap();
    /// assert_eq!(s.n_qubits(), 3);
    /// assert_eq!(s.amplitude(0), num_complex::Complex::new(1.0, 0.0));
    /// ```
    pub fn new(n: usize) -> Result<Self, StateError> {
        if n == 0 || n > 20 {
            return Err(StateError::TooManyQubits(n));
        }
        let dim = 1usize << n;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        Ok(State { amps })
    }

    /// Alias for [`State::new`] retained for readability in examples.
    ///
    /// # Errors
    ///
    /// See [`State::new`].
    pub fn zero(n: usize) -> Result<Self, StateError> {
        Self::new(n)
    }

    /// The number of qubits represented by this state.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.amps.len().trailing_zeros() as usize
    }

    /// The raw amplitude vector (computational-basis order).
    #[must_use]
    pub fn amplitudes(&self) -> &[Complex<f64>] {
        &self.amps
    }

    /// The amplitude of the basis state `|index⟩`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^n`.
    #[must_use]
    pub fn amplitude(&self, index: usize) -> Complex<f64> {
        self.amps[index]
    }

    /// The Born-rule probabilities `|amp|^2` for each basis state.
    ///
    /// The returned vector sums to `1` up to floating-point rounding.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(Complex::norm_sqr).collect()
    }

    /// The Euclidean norm `sqrt(Σ |amp|^2)` of the state vector.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.amps.iter().map(Complex::norm_sqr).sum::<f64>().sqrt()
    }

    /// Rescale the state so that its norm becomes `1`.
    ///
    /// # Panics
    ///
    /// Panics if the current norm is zero (a zero vector has no normalization).
    pub fn normalize(&mut self) {
        let n = self.norm();
        let inv = 1.0 / n;
        for a in &mut self.amps {
            *a = a.scale(inv);
        }
    }

    /// Sample a basis index from the Born-rule distribution.
    ///
    /// Uses `rng` to draw a uniform value in `[0, 1)` and returns the first
    /// index whose cumulative probability exceeds it.
    ///
    /// This method is **non-destructive**: it only reads the state and reports
    /// a sampled outcome. Repeated calls therefore keep returning outcomes from
    /// the same (uncollapsed) distribution. Use [`State::measure_collapsing`]
    /// when you need the standard post-measurement collapse semantics (as in
    /// Qiskit/Cirq), where the state is projected onto the measured basis
    /// state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_quantum::State;
    /// use tpt_math_prob_core::SplitMix64;
    ///
    /// let mut state = State::new(1).unwrap();
    /// state.h(0).unwrap();
    /// let mut rng = SplitMix64::seed_from_u64(1);
    /// let idx = state.measure(&mut rng);
    /// assert!(idx == 0 || idx == 1);
    /// ```
    #[must_use]
    pub fn measure(&self, rng: &mut impl Rng) -> usize {
        let r = rng.next_f64();
        let mut cumulative = 0.0;
        for (i, a) in self.amps.iter().enumerate() {
            cumulative += a.norm_sqr();
            if r < cumulative {
                return i;
            }
        }
        self.amps.len() - 1
    }

    /// Sample a basis index and collapse the state onto that outcome.
    ///
    /// Unlike [`State::measure`], which only samples without disturbing the
    /// state, this consumes the measurement: the state is projected onto the
    /// measured computational-basis state (all other amplitudes zeroed) and the
    /// surviving amplitude is renormalised. This matches the standard simulator
    /// semantics (Qiskit/Cirq), where a subsequent measurement of the same
    /// register is deterministic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_quantum::State;
    /// use tpt_math_prob_core::SplitMix64;
    ///
    /// let mut state = State::new(1).unwrap();
    /// state.x(0).unwrap(); // |1⟩
    /// let mut rng = SplitMix64::seed_from_u64(7);
    /// let idx = state.measure_collapsing(&mut rng);
    /// assert_eq!(idx, 1);
    /// assert!((state.amplitude(1).norm() - 1.0).abs() < 1e-9);
    /// ```
    pub fn measure_collapsing(&mut self, rng: &mut impl Rng) -> usize {
        let idx = self.measure(rng);
        let amp = self.amps[idx];
        let norm = amp.norm();
        for a in &mut self.amps {
            *a = Complex::new(0.0, 0.0);
        }
        if norm > 0.0 {
            // Preserve the (now unit-magnitude) phase of the surviving amplitude.
            self.amps[idx] = amp.scale(1.0 / norm);
        }
        idx
    }

    /// Expectation value `⟨Z^{⊗n}⟩` of the all-qubit parity operator.
    ///
    /// Equal to `Σ_i (-1)^{popcount(i)} |amp_i|^2`. For the all-zero state this
    /// is `1`; for a uniform superposition of even/odd parity states it is `0`.
    #[must_use]
    pub fn expectation_z(&self) -> f64 {
        self.amps
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let sign = if i.count_ones() & 1 == 0 { 1.0 } else { -1.0 };
                sign * a.norm_sqr()
            })
            .sum()
    }

    /// Apply a single-qubit 2×2 gate to `qubit`.
    ///
    /// `gate` is row-major: `[g00, g01, g10, g11]` for
    /// `[[g00, g01], [g10, g11]]`. For each pair of basis states that differ
    /// only in `qubit`, the two amplitudes are transformed by the matrix.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit >= n_qubits()`.
    pub fn apply_single(
        &mut self,
        gate: &[Complex<f64>; 4],
        qubit: usize,
    ) -> Result<(), StateError> {
        let n = self.n_qubits();
        if qubit >= n {
            return Err(StateError::InvalidQubit { qubit, n });
        }
        let (g00, g01, g10, g11) = (gate[0], gate[1], gate[2], gate[3]);
        let flip = 1usize << qubit;
        for i in 0..(1usize << n) {
            if i & flip == 0 {
                let j = i | flip;
                let a0 = self.amps[i];
                let a1 = self.amps[j];
                self.amps[i] = g00 * a0 + g01 * a1;
                self.amps[j] = g10 * a0 + g11 * a1;
            }
        }
        Ok(())
    }

    /// Apply a controlled-NOT: flip `target` whenever `control` is set.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if either qubit is out of range, or
    /// [`StateError::SameQubits`] if `control == target`.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), StateError> {
        let n = self.n_qubits();
        if control >= n {
            return Err(StateError::InvalidQubit { qubit: control, n });
        }
        if target >= n {
            return Err(StateError::InvalidQubit { qubit: target, n });
        }
        if control == target {
            return Err(StateError::SameQubits(control));
        }
        let c = 1usize << control;
        let t = 1usize << target;
        for i in 0..(1usize << n) {
            if i & c != 0 && i & t == 0 {
                self.amps.swap(i, i | t);
            }
        }
        Ok(())
    }

    /// Apply a single-qubit `gate` to `target`, controlled on `control == 1`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if either qubit is out of range, or
    /// [`StateError::SameQubits`] if `control == target`.
    pub fn apply_controlled(
        &mut self,
        control: usize,
        target: usize,
        gate: &[Complex<f64>; 4],
    ) -> Result<(), StateError> {
        let n = self.n_qubits();
        if control >= n {
            return Err(StateError::InvalidQubit { qubit: control, n });
        }
        if target >= n {
            return Err(StateError::InvalidQubit { qubit: target, n });
        }
        if control == target {
            return Err(StateError::SameQubits(control));
        }
        let (g00, g01, g10, g11) = (gate[0], gate[1], gate[2], gate[3]);
        let c = 1usize << control;
        let t = 1usize << target;
        for i in 0..(1usize << n) {
            if i & c != 0 && i & t == 0 {
                let j = i | t;
                let a0 = self.amps[i];
                let a1 = self.amps[j];
                self.amps[i] = g00 * a0 + g01 * a1;
                self.amps[j] = g10 * a0 + g11 * a1;
            }
        }
        Ok(())
    }

    /// Apply the Hadamard gate `H` to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn h(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&H, qubit)
    }

    /// Apply the Pauli-X (bit-flip) gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn x(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&X, qubit)
    }

    /// Apply the Pauli-Y gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn y(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&Y, qubit)
    }

    /// Apply the Pauli-Z (phase-flip) gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn z(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&Z, qubit)
    }

    /// Apply the phase gate `S` to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn s(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&S, qubit)
    }

    /// Apply the π/4 (T) gate to `qubit`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if `qubit` is out of range.
    pub fn t(&mut self, qubit: usize) -> Result<(), StateError> {
        self.apply_single(&T, qubit)
    }

    /// Apply a controlled-NOT gate.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::InvalidQubit`] if either qubit is out of range, or
    /// [`StateError::SameQubits`] if `control == target`.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), StateError> {
        self.apply_cnot(control, target)
    }
}

/// Hadamard gate `H = (1/√2)·[[1,1],[1,-1]]`.
pub const H: [Complex<f64>; 4] = [
    Complex::new(INV_SQRT2, 0.0),
    Complex::new(INV_SQRT2, 0.0),
    Complex::new(INV_SQRT2, 0.0),
    Complex::new(-INV_SQRT2, 0.0),
];

/// Pauli-X (bit-flip) gate `X = [[0,1],[1,0]]`.
pub const X: [Complex<f64>; 4] = [
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0),
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
];

/// Pauli-Y gate `Y = [[0,-i],[i,0]]`.
pub const Y: [Complex<f64>; 4] = [
    Complex::new(0.0, 0.0),
    Complex::new(0.0, -1.0),
    Complex::new(0.0, 1.0),
    Complex::new(0.0, 0.0),
];

/// Pauli-Z (phase-flip) gate `Z = [[1,0],[0,-1]]`.
pub const Z: [Complex<f64>; 4] = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(-1.0, 0.0),
];

/// Phase gate `S = [[1,0],[0,i]]`.
pub const S: [Complex<f64>; 4] = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 1.0),
];

/// π/4 (T) gate `T = [[1,0],[0,e^{iπ/4}]]`.
pub const T: [Complex<f64>; 4] = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(INV_SQRT2, INV_SQRT2),
];

#[cfg(test)]
impl State {
    /// Test-only accessor exposing the amplitude vector mutably.
    fn amplitudes_mut_for_test(&mut self) -> &mut [Complex<f64>] {
        &mut self.amps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_math_prob_core::SplitMix64;

    const TOL: f64 = 1e-9;

    #[test]
    fn new_rejects_zero_and_too_many() {
        assert_eq!(State::new(0), Err(StateError::TooManyQubits(0)));
        assert_eq!(State::new(21), Err(StateError::TooManyQubits(21)));
        assert!(State::new(1).is_ok());
        assert!(State::new(20).is_ok());
    }

    #[test]
    fn hadamard_on_zero_state() {
        let mut s = State::new(1).unwrap();
        s.h(0).unwrap();
        let inv = INV_SQRT2;
        assert_abs_diff_eq!(s.amplitude(0).re, inv, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(1).re, inv, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(0).im, 0.0, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(1).im, 0.0, epsilon = TOL);
    }

    #[test]
    fn x_flips_zero_to_one() {
        let mut s = State::new(1).unwrap();
        s.x(0).unwrap();
        assert_abs_diff_eq!(s.amplitude(1).re, 1.0, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(0).re, 0.0, epsilon = TOL);
    }

    #[test]
    fn bell_state() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        let inv = INV_SQRT2;
        assert_abs_diff_eq!(s.amplitude(0).re, inv, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(1).re, 0.0, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(2).re, 0.0, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(3).re, inv, epsilon = TOL);

        let p = s.probabilities();
        assert_abs_diff_eq!(p[0], 0.5, epsilon = TOL);
        assert_abs_diff_eq!(p[3], 0.5, epsilon = TOL);
        // All-qubit parity ⟨Z⊗Z⟩: both |00⟩ and |11⟩ have even popcount ⇒ +1.
        assert_abs_diff_eq!(s.expectation_z(), 1.0, epsilon = TOL);
    }

    #[test]
    fn ghz_state() {
        let mut s = State::new(3).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        s.cnot(1, 2).unwrap();
        let inv = INV_SQRT2;
        assert_abs_diff_eq!(s.amplitude(0).re, inv, epsilon = TOL);
        assert_abs_diff_eq!(s.amplitude(7).re, inv, epsilon = TOL);
        for i in [1usize, 2, 3, 4, 5, 6] {
            assert_abs_diff_eq!(s.amplitude(i).norm(), 0.0, epsilon = TOL);
        }
        // GHZ parity: |000⟩ even (+1), |111⟩ odd (-1) ⇒ expectation 0.
        assert_abs_diff_eq!(s.expectation_z(), 0.0, epsilon = TOL);
    }

    #[test]
    fn measure_reproduces_distribution() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();

        let mut rng = SplitMix64::seed_from_u64(12345);
        let trials = 2000;
        let mut count0 = 0usize;
        let mut count3 = 0usize;
        for _ in 0..trials {
            match s.measure(&mut rng) {
                0 => count0 += 1,
                3 => count3 += 1,
                _ => {}
            }
        }
        let f0 = count0 as f64 / trials as f64;
        let f3 = count3 as f64 / trials as f64;
        assert!((f0 - 0.5).abs() < 0.1, "f0 = {f0}");
        assert!((f3 - 0.5).abs() < 0.1, "f3 = {f3}");
    }

    #[test]
    fn measure_collapsing_projects_state() {
        // |1> collapses deterministically to |1>.
        let mut s = State::new(1).unwrap();
        s.x(0).unwrap();
        let mut rng = SplitMix64::seed_from_u64(7);
        let idx = s.measure_collapsing(&mut rng);
        assert_eq!(idx, 1);
        assert!((s.amplitude(1).norm() - 1.0).abs() < TOL);
        assert!((s.amplitude(0).norm()).abs() < TOL);

        // A superposition collapses to one of its basis states and stays there.
        let mut s2 = State::new(2).unwrap();
        s2.h(0).unwrap();
        s2.cnot(0, 1).unwrap();
        let mut rng2 = SplitMix64::seed_from_u64(99);
        let idx2 = s2.measure_collapsing(&mut rng2);
        assert!(idx2 == 0 || idx2 == 3);
        // After collapse, exactly one basis state carries unit probability.
        let p = s2.probabilities();
        let max_p: f64 = p.iter().cloned().fold(0.0, f64::max);
        assert!((max_p - 1.0).abs() < TOL);
        assert!((s2.norm() - 1.0).abs() < TOL);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let mut s = State::new(3).unwrap();
        s.h(0).unwrap();
        s.h(1).unwrap();
        s.h(2).unwrap();
        let total: f64 = s.probabilities().iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = TOL);
        assert_abs_diff_eq!(s.norm(), 1.0, epsilon = TOL);
    }

    #[test]
    fn error_paths() {
        let mut s = State::new(2).unwrap();
        assert_eq!(s.h(5), Err(StateError::InvalidQubit { qubit: 5, n: 2 }));
        assert_eq!(s.cnot(1, 1), Err(StateError::SameQubits(1)));
        assert_eq!(
            s.cnot(5, 0),
            Err(StateError::InvalidQubit { qubit: 5, n: 2 })
        );
    }

    #[test]
    fn controlled_gate_matches_analytic() {
        // |0⟩ on target: controlled-X does nothing.
        let mut s = State::new(2).unwrap();
        s.x(0).unwrap(); // control = 1
        s.apply_controlled(0, 1, &X).unwrap();
        assert_abs_diff_eq!(s.amplitude(3).re, 1.0, epsilon = TOL);

        // |0⟩ on control: target stays |0⟩.
        let mut s2 = State::new(2).unwrap();
        s2.apply_controlled(0, 1, &X).unwrap();
        assert_abs_diff_eq!(s2.amplitude(0).re, 1.0, epsilon = TOL);
    }

    #[test]
    fn normalize_restores_unit_norm() {
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        // Scale artificially then renormalize.
        for a in s.amplitudes_mut_for_test() {
            *a = a.scale(2.0);
        }
        assert_abs_diff_eq!(s.norm(), 2.0, epsilon = TOL);
        s.normalize();
        assert_abs_diff_eq!(s.norm(), 1.0, epsilon = TOL);
    }
}
