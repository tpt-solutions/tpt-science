//! Tensor-product (Kronecker) formulation of qubit circuits.
//!
//! The in-place [`State`] simulator mutates amplitudes directly with explicit
//! index arithmetic. This module provides the complementary *dense* view: a
//! circuit is recorded as a sequence of single- and two-qubit gates and
//! assembled into a single complex unitary `U`, then applied as
//! `|ψ'⟩ = U|ψ⟩`.
//!
//! Because `tpt-math-linalg` is real-only (`f64`), every complex scalar
//! `a + bi` is embedded as the real `2×2` block `[[a, -b], [b, a]]`; a
//! complex `d×d` matrix therefore becomes a real `2d×2d` matrix handled by
//! `DMatrix<f64>`. This is the tensor-product (Kronecker) representation of a
//! circuit — exact for the qubits present — and is most useful at modest
//! width (it is `O(4^n)` to build and `O(4^n)` to apply). The in-place
//! [`State`] path remains the scalable one for wide registers.

use num_complex::Complex;
use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

use crate::{State, StateError};

/// Real `2×2` block embedding of a complex scalar `a + bi`.
#[inline]
fn block(z: Complex<f64>) -> [f64; 4] {
    [z.re, -z.im, z.im, z.re]
}

/// Kronecker product `A ⊗ B` over `DMatrix<f64>`.
///
/// Entry `(i, j)` of the result is `A[(i/r, j/s)] · B[(i%r, j%s)]` where
/// `r, s` are the row/column counts of `B`.
pub fn kron(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let (p, q) = (a.nrows(), a.ncols());
    let (r, s) = (b.nrows(), b.ncols());
    DMatrix::from_fn(p * r, q * s, |i, j| {
        let ai = i / r;
        let aj = j / s;
        let bi = i % r;
        let bj = j % s;
        a[(ai, aj)] * b[(bi, bj)]
    })
}

/// Embed a complex `2×2` gate (row-major `[g00, g01, g10, g11]`) as a real `4×4`.
pub fn embed_gate_2x2(gate: &[Complex<f64>; 4]) -> DMatrix<f64> {
    DMatrix::from_fn(4, 4, |i, j| {
        let qr = i / 2;
        let qc = j / 2;
        let sr = i % 2;
        let sc = j % 2;
        let z = gate[qr * 2 + qc];
        block(z)[sr * 2 + sc]
    })
}

/// Embed a complex `4×4` gate (row-major, 16 entries) as a real `8×8`.
pub fn embed_gate_4x4(gate: &[Complex<f64>; 16]) -> DMatrix<f64> {
    DMatrix::from_fn(8, 8, |i, j| {
        let qr = i / 2;
        let qc = j / 2;
        let sr = i % 2;
        let sc = j % 2;
        let z = gate[qr * 4 + qc];
        block(z)[sr * 2 + sc]
    })
}

/// A `2×2` real identity, the embedding of the complex identity gate.
fn identity_2x2() -> DMatrix<f64> {
    DMatrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 })
}

/// Real `2d×2d` block embedding of an arbitrary `d×d` complex matrix given
/// row-major as `entries` (length `d*d`).
///
/// Generalizes [`embed_gate_2x2`]/[`embed_gate_4x4`] to any dimension `d`.
/// Used by the [`crate::density`] module to embed density matrices and
/// (possibly non-unitary) Kraus operators into the same real-embedded
/// representation used for unitaries, so that gate/channel application can
/// reuse ordinary real `DMatrix` multiplication.
///
/// A useful identity of this embedding: for any complex matrix `A`,
/// `embed(A†) == embed(A).transpose()` (conjugate-transpose becomes plain
/// real transpose), which is what lets [`crate::density::DensityMatrix`]
/// implement `ρ ↦ K ρ K†` as ordinary real matrix products.
pub(crate) fn embed_complex_square(entries: &[Complex<f64>], d: usize) -> DMatrix<f64> {
    debug_assert_eq!(entries.len(), d * d);
    DMatrix::from_fn(2 * d, 2 * d, |i, j| {
        let qr = i / 2;
        let qc = j / 2;
        let sr = i % 2;
        let sc = j % 2;
        let z = entries[qr * d + qc];
        block(z)[sr * 2 + sc]
    })
}

/// Extract the complex entry `(i, j)` of a `d×d` complex matrix from its
/// real `2d×2d` block embedding (the inverse of [`embed_complex_square`]).
pub(crate) fn unembed_entry(real: &DMatrix<f64>, i: usize, j: usize) -> Complex<f64> {
    Complex::new(real[(2 * i, 2 * j)], real[(2 * i + 1, 2 * j)])
}

/// Complex `4×4` matrix (row-major, 16 entries) of a controlled-`gate`.
///
/// `gate` is applied to `target` whenever `control` is set. The two qubits
/// span the `lo = min(control, target)` (low bit) and `hi = max(...)`
/// (high bit) positions of the 2-qubit index, so the result is independent of
/// which physical qubit is the higher index.
fn controlled_2q_matrix(
    control: usize,
    target: usize,
    gate: &[Complex<f64>; 4],
) -> [Complex<f64>; 16] {
    let lo = control.min(target);
    let mut m = [Complex::new(0.0, 0.0); 16];
    for idx_in in 0..4usize {
        let b_lo = idx_in & 1;
        let b_hi = (idx_in >> 1) & 1;
        let control_bit = if control == lo { b_lo } else { b_hi };
        if control_bit == 0 {
            m[idx_in * 4 + idx_in] = Complex::new(1.0, 0.0);
        } else {
            let target_in = if target == lo { b_lo } else { b_hi };
            for out_t in 0..2usize {
                let amp = gate[target_in * 2 + out_t];
                if amp.norm_sqr() < 1e-30 {
                    continue;
                }
                let mut idx_out = 0usize;
                if control == lo {
                    idx_out |= control_bit;
                } else {
                    idx_out |= control_bit << 1;
                }
                if target == lo {
                    idx_out |= out_t;
                } else {
                    idx_out |= out_t << 1;
                }
                m[idx_out * 4 + idx_in] = amp;
            }
        }
    }
    m
}

/// Complex `4×4` SWAP matrix (row-major): `|01⟩ ↔ |10⟩`.
fn swap_matrix() -> [Complex<f64>; 16] {
    let mut m = [Complex::new(0.0, 0.0); 16];
    m[0] = Complex::new(1.0, 0.0);
    m[6] = Complex::new(1.0, 0.0);
    m[9] = Complex::new(1.0, 0.0);
    m[15] = Complex::new(1.0, 0.0);
    m
}

/// An elementary gate recorded by [`Circuit`].
#[derive(Clone)]
enum Elem {
    /// Single-qubit gate on one qubit (embedded as a `4×4` real block).
    Single {
        qubit: usize,
        gate: [Complex<f64>; 4],
    },
    /// Two-qubit gate on an *adjacent* pair `(lo, lo + 1)`, embedded as an
    /// `8×8` real block. Non-adjacent gates are decomposed into a sequence of
    /// these plus SWAPs.
    Pair {
        lo: usize,
        hi: usize,
        matrix: [Complex<f64>; 16],
    },
}

/// A circuit recorded as a sequence of single- and two-qubit gates.
///
/// Gates are stored verbatim and assembled into a single real-embedded unitary
/// by [`Circuit::unitary`], which can then be applied to a [`State`] with
/// [`State::apply_unitary`].
///
/// # Examples
///
/// ```
/// use tpt_sci_quantum::Circuit;
/// use tpt_sci_quantum::State;
///
/// let mut c = Circuit::new(2);
/// c.h(0).cnot(0, 1);
/// let u = c.unitary();
/// let s = State::new(2).unwrap();
/// let s2 = s.apply_unitary(&u).unwrap();
/// let p = s2.probabilities();
/// assert!((p[0] - 0.5).abs() < 1e-9);
/// assert!((p[3] - 0.5).abs() < 1e-9);
/// ```
#[derive(Clone, Default)]
pub struct Circuit {
    n: usize,
    elems: Vec<Elem>,
}

impl Circuit {
    /// Create an empty circuit acting on `n` qubits.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Circuit {
            n,
            elems: Vec::new(),
        }
    }

    /// The number of qubits this circuit acts on.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n
    }

    /// Record an arbitrary single-qubit `gate` on `qubit`.
    pub fn gate(&mut self, qubit: usize, gate: [Complex<f64>; 4]) -> &mut Self {
        self.elems.push(Elem::Single { qubit, gate });
        self
    }

    /// Record a Hadamard gate on `qubit`.
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::H)
    }

    /// Record a Pauli-X gate on `qubit`.
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::X)
    }

    /// Record a Pauli-Y gate on `qubit`.
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::Y)
    }

    /// Record a Pauli-Z gate on `qubit`.
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::Z)
    }

    /// Record a phase `S` gate on `qubit`.
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::S)
    }

    /// Record a `π/4` `T` gate on `qubit`.
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        self.gate(qubit, crate::T)
    }

    /// Record a controlled-NOT: flip `target` whenever `control` is set.
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.controlled(control, target, crate::X)
    }

    /// Record a `gate` applied to `target`, controlled on `control == 1`.
    ///
    /// Non-adjacent `control`/`target` are handled by a SWAP decomposition:
    /// the target is swapped down next to the control, the controlled gate is
    /// applied on the adjacent pair, and the swaps are undone.
    pub fn controlled(
        &mut self,
        control: usize,
        target: usize,
        gate: [Complex<f64>; 4],
    ) -> &mut Self {
        if control == target {
            return self;
        }
        let c = control;
        let t = target;
        let lo = c.min(t);
        let hi = c.max(t);

        let mut ops: Vec<Elem> = Vec::new();
        // Bring `target` (at `hi`) down to `lo + 1` by adjacent swaps.
        for k in ((lo + 1)..hi).rev() {
            ops.push(Elem::Pair {
                lo: k,
                hi: k + 1,
                matrix: swap_matrix(),
            });
        }
        ops.push(Elem::Pair {
            lo,
            hi: lo + 1,
            matrix: controlled_2q_matrix(lo, lo + 1, &gate),
        });
        // Undo the swaps, restoring `target` to `hi`.
        for k in (lo + 1)..hi {
            ops.push(Elem::Pair {
                lo: k,
                hi: k + 1,
                matrix: swap_matrix(),
            });
        }
        self.elems.extend(ops);
        self
    }

    /// Assemble the full real-embedded unitary (`2^(n+1) × 2^(n+1)`) of the
    /// recorded circuit.
    ///
    /// Each elementary gate is expanded to a full-size matrix via Kronecker
    /// products of per-qubit factors and the matrices are multiplied in
    /// application order. The result can be applied with [`State::apply_unitary`].
    #[must_use]
    pub fn unitary(&self) -> DMatrix<f64> {
        let n = self.n;
        let expected = (1usize << n) * 2;
        let mut acc = DMatrix::from_fn(expected, expected, |i, j| if i == j { 1.0 } else { 0.0 });
        for e in &self.elems {
            let m = match e {
                Elem::Single { qubit, gate } => expand_single(n, *qubit, gate),
                Elem::Pair { lo, hi, matrix } => expand_pair(n, *lo, *hi, matrix),
            };
            acc = m * acc;
        }
        acc
    }
}

/// Expand a single-qubit gate to a full `2^(n+1) × 2^(.)` real-embedded matrix.
///
/// `pub(crate)` so the [`crate::density`] module can embed single-qubit Kraus
/// operators (e.g. bit-flip/depolarizing noise) into the full `n`-qubit space
/// the same way single-qubit gates are embedded here.
pub(crate) fn expand_single(n: usize, qubit: usize, gate: &[Complex<f64>; 4]) -> DMatrix<f64> {
    let block = embed_gate_2x2(gate);
    let mut u: Option<DMatrix<f64>> = None;
    for q in 0..n {
        let f = if q == qubit {
            block.clone()
        } else {
            identity_2x2()
        };
        u = Some(match u {
            None => f,
            Some(prev) => kron(&f, &prev),
        });
    }
    u.expect("n >= 1")
}

/// Expand an adjacent two-qubit gate `(lo, lo + 1)` to a full real-embedded matrix.
fn expand_pair(n: usize, lo: usize, hi: usize, matrix: &[Complex<f64>; 16]) -> DMatrix<f64> {
    debug_assert_eq!(hi, lo + 1);
    let block = embed_gate_4x4(matrix);
    let mut u: Option<DMatrix<f64>> = None;
    for q in 0..n {
        if q == hi {
            // Consumed by the combined `lo`/`hi` block; skip.
            continue;
        }
        let f = if q == lo {
            block.clone()
        } else {
            identity_2x2()
        };
        u = Some(match u {
            None => f,
            Some(prev) => kron(&f, &prev),
        });
    }
    u.expect("n >= 2")
}

impl State {
    /// Apply a real-embedded unitary `u` (as produced by [`Circuit::unitary`]
    /// or built directly) to this state, returning the evolved state.
    ///
    /// The state's complex amplitude vector is embedded into a real
    /// `DVector<f64>` (`2·2^n` long, interleaved real/imaginary parts),
    /// multiplied by `u`, and un-embedded.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::UnitarySizeMismatch`] if `u` is not
    /// `2^(n+1) × 2^(n+1)` for this state's `n` qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_sci_quantum::State;
    /// use tpt_sci_quantum::tensor::kron;
    /// use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
    ///
    /// // H ⊗ I on two qubits, built from per-qubit factors.
    /// let h = tpt_sci_quantum::embed_gate_2x2(&tpt_sci_quantum::H);
    /// let i2 = DMatrix::from_fn(2, 2, |a, b| if a == b { 1.0 } else { 0.0 });
    /// let u = kron(&h, &i2);
    /// let s = State::new(2).unwrap();
    /// let s2 = s.apply_unitary(&u).unwrap();
    /// assert!((s2.amplitude(0).re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    /// ```
    pub fn apply_unitary(&self, u: &DMatrix<f64>) -> Result<State, StateError> {
        let n = self.n_qubits();
        let d = 1usize << n;
        let expected = d * 2;
        if u.nrows() != expected || u.ncols() != expected {
            return Err(StateError::UnitarySizeMismatch {
                expected,
                got: u.nrows(),
            });
        }
        let w = DVector::from_fn(expected, |k| {
            let i = k / 2;
            let part = k % 2;
            let z = self.amps[i];
            if part == 0 { z.re } else { z.im }
        });
        let out = u.clone() * w;
        let mut amps = vec![Complex::new(0.0, 0.0); d];
        for i in 0..d {
            amps[i] = Complex::new(out[2 * i], out[2 * i + 1]);
        }
        Ok(State { amps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{H, X};
    use approx::assert_abs_diff_eq;

    const TOL: f64 = 1e-9;

    #[test]
    fn kron_identity_replicates_factor() {
        let a = embed_gate_2x2(&H);
        let i2 = DMatrix::from_fn(2, 2, |p, q| if p == q { 1.0 } else { 0.0 });
        let k = kron(&a, &i2);
        assert_eq!(k.nrows(), 8);
        assert_eq!(k.ncols(), 8);
    }

    #[test]
    fn single_h_matches_state_vector() {
        let mut s = State::new(1).unwrap();
        s.h(0).unwrap();
        let expected = s.amplitudes().to_vec();

        let mut c = Circuit::new(1);
        c.h(0);
        let u = c.unitary();
        let s2 = State::new(1).unwrap().apply_unitary(&u).unwrap();
        for (a, b) in expected.iter().zip(s2.amplitudes()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
            assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
        }
    }

    #[test]
    fn bell_state_matches_unitary_path() {
        // In-place path.
        let mut s = State::new(2).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        let expected = s.amplitudes().to_vec();

        // Tensor (Kronecker) path.
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let u = c.unitary();
        let s2 = State::new(2).unwrap().apply_unitary(&u).unwrap();
        for (a, b) in expected.iter().zip(s2.amplitudes()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
            assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
        }
    }

    #[test]
    fn ghz_with_nonadjacent_cnot() {
        // In-place path: H(0), CNOT(0,1), CNOT(0,2) — note (0,2) is non-adjacent.
        let mut s = State::new(3).unwrap();
        s.h(0).unwrap();
        s.cnot(0, 1).unwrap();
        s.cnot(0, 2).unwrap();
        let expected = s.amplitudes().to_vec();

        // Tensor path must decompose the non-adjacent CNOT via SWAPs.
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).cnot(0, 2);
        let u = c.unitary();
        let s2 = State::new(3).unwrap().apply_unitary(&u).unwrap();
        for (a, b) in expected.iter().zip(s2.amplitudes()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = TOL);
            assert_abs_diff_eq!(a.im, b.im, epsilon = TOL);
        }
        // GHZ: only |000⟩ and |111⟩ populated.
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        assert_abs_diff_eq!(s2.amplitude(0).re, inv, epsilon = TOL);
        assert_abs_diff_eq!(s2.amplitude(7).re, inv, epsilon = TOL);
    }

    #[test]
    fn controlled_gate_matches_analytic() {
        // Controlled-X with |0⟩ control leaves |00⟩ unchanged.
        let mut c = Circuit::new(2);
        c.controlled(0, 1, X);
        let u = c.unitary();
        let s2 = State::new(2).unwrap().apply_unitary(&u).unwrap();
        assert_abs_diff_eq!(s2.amplitude(0).re, 1.0, epsilon = TOL);
        assert_abs_diff_eq!(s2.amplitude(3).norm(), 0.0, epsilon = TOL);

        // With a |1⟩ control (X(0) first), the target flips: |10⟩ → |11⟩.
        let mut c2 = Circuit::new(2);
        c2.x(0).controlled(0, 1, X);
        let s3 = State::new(2).unwrap().apply_unitary(&c2.unitary()).unwrap();
        assert_abs_diff_eq!(s3.amplitude(3).re, 1.0, epsilon = TOL);
    }

    #[test]
    fn unitary_size_mismatch_is_rejected() {
        let wrong = DMatrix::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
        let s = State::new(2).unwrap();
        assert_eq!(
            s.apply_unitary(&wrong),
            Err(StateError::UnitarySizeMismatch {
                expected: 8,
                got: 4
            })
        );
    }
}
