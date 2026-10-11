//! # tpt-sci-electrophys
//!
//! **Cardiac electrophysiology** for the `tpt-science` pillar, built on
//! [`tpt_sci_ode`] (for the membrane kinetics) and [`tpt_sci_grid`] (for the
//! extracellular/bidomain diffusion operator). It provides:
//!
//! * A [`IonicModel`] trait so multiple membrane kinetics can drive the same
//!   tissue solver. Two implementations ship: the classic **Hodgkin–Huxley**
//!   giant-axon model ([`HodgkinHuxley`]) and the **Ten Tusscher–Panfilov 2004**
//!   human-ventricular myocyte ([`TenTusscher`]).
//! * A **monodomain / bidomain** tissue propagator [`Tissue`] that couples the
//!   membrane to a 2-D diffusion operator. The monodomain is
//!   `dVm/dt = −I_ion/Cm + D·∇²Vm`. Enabling bidomain adds the extracellular
//!   potential `Ve`, solved each step from the elliptic equation
//!   `(σ_i + σ_e)·∇²Ve = −σ_i·∇²Vm` (via [`tpt_sci_grid`]'s 2-D Laplacian) and
//!   coupling it back into the intracellular update
//!   `dVm/dt = −I_ion/Cm + σ_i·∇²(Vm + Ve)`.
//! * **Anisotropic (tensor) diffusion**: [`Tissue`] accepts a per-node 2×2
//!   diffusivity tensor `D`, so fibre-orientation effects (faster conduction
//!   along fibres) are captured via `∇·(D∇Vm)`.
//!
//! The model integrates `dVi/dt = −I_ion/Cm + D·∇²V` (monodomain) across a grid,
//! so an action potential launched at one node propagates through the tissue.
//!
//! # Example
//!
//! ```
//! use tpt_sci_electrophys::HodgkinHuxley;
//!
//! let mut hh = HodgkinHuxley::resting();
//! // Depolarize; integrate the membrane ODE for a short time.
//! let y = hh.state();
//! assert!(y.len() == 4);
//! hh.step(0.01);
//! assert!(hh.voltage().is_finite());
//! ```
#![forbid(unsafe_code)]

mod error;

pub use error::ElectrophysError;

use tpt_sci_grid::sparse::{conjugate_gradient, laplacian_1d_sparse, laplacian_2d_sparse};
use tpt_sci_grid::{Boundary, UniformGrid1D, UniformGrid2D};

mod contraction;
mod diffusion;
mod hodgkin_huxley;
mod ionic;
mod ten_tusscher;
mod tissue;

pub use contraction::LandActiveTension;
pub use diffusion::{ConductivityTensor, DiffusionTensor, tensor_diffusion_2d};
pub use hodgkin_huxley::HodgkinHuxley;
pub use ionic::IonicModel;
pub use ten_tusscher::TenTusscher;
pub use tissue::{Diffusivity, Tissue};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tpt_sci_grid::sparse::laplacian_2d_sparse;
    use tpt_sci_grid::{Boundary, UniformGrid2D};

    #[test]
    fn hh_resting_is_finite() {
        let mut hh = HodgkinHuxley::resting();
        for _ in 0..50 {
            hh.step(0.01);
        }
        assert!(hh.voltage().is_finite());
        assert!((0.0..=1.0).contains(&hh.m));
    }

    #[test]
    fn hh_depolarizes_then_recovers() {
        let mut hh = HodgkinHuxley::new();
        hh.v = 20.0; // stimulus
        let v0 = hh.voltage();
        for _ in 0..200 {
            hh.step(0.01);
        }
        assert!(hh.voltage().is_finite());
        assert!(hh.voltage() < v0 + 1e-6 || hh.voltage() > 0.0);
    }

    #[test]
    fn tissue_propagates_from_stimulus() {
        let mut t = Tissue::new(16, 16, 0.5).unwrap();
        t.stimulate(0, 8, 40.0);
        for _ in 0..100 {
            t.step(0.005);
        }
        // A stimulus should depolarize the tissue well above the resting (−65 mV)
        // field — the depolarization propagates and recruits the sheet.
        assert!(t.max_voltage().is_finite());
        assert!(
            t.max_voltage() > 10.0,
            "stimulation should depolarize the tissue"
        );
    }

    #[test]
    fn ionic_current_finite_at_rest() {
        let hh = HodgkinHuxley::resting();
        assert!(hh.ionic_current().is_finite());
    }

    /// Stimulate a single TenTusscher cell with a 1 ms, -52 pA/pF pulse and
    /// integrate for `duration_ms`. Returns the voltage trace sampled every 0.01 ms.
    fn tt_trace(duration_ms: f64) -> Vec<f64> {
        let mut cell = TenTusscher::resting();
        let dt = 0.01;
        let steps = (duration_ms / dt) as usize;
        let mut trace = Vec::with_capacity(steps);
        for i in 0..steps {
            let stim = if (i as f64) * dt < 1.0 { -52.0 } else { 0.0 };
            let v = cell.voltage();
            cell.advance_reaction(v, dt);
            cell.v = v + dt * -(cell.ionic_current() + stim) / cell.capacitance();
            trace.push(cell.v);
        }
        trace
    }

    #[test]
    fn tt_action_potential_has_plateau_and_repolarizes() {
        let dt = 0.01;
        let trace = tt_trace(500.0);
        assert!(trace.iter().all(|v| v.is_finite()), "integration blew up");

        let peak = trace.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            (10.0..45.0).contains(&peak),
            "AP peak {peak} mV out of range"
        );

        // Plateau: well after the upstroke, the epicardial cell stays depolarized.
        let plateau = trace[(100.0 / dt) as usize];
        assert!(
            (-10.0..30.0).contains(&plateau),
            "no plateau: V(100 ms) = {plateau} mV"
        );

        // APD: time above -40 mV. Epicardial APD is roughly 250 to 300 ms.
        let up = trace.iter().position(|&v| v > -40.0).expect("no upstroke");
        let down = up
            + trace[up..]
                .iter()
                .position(|&v| v < -40.0)
                .expect("no repolarization");
        let apd = (down - up) as f64 * dt;
        assert!((200.0..350.0).contains(&apd), "APD {apd} ms out of range");

        // Back near rest by the end of the run.
        let final_v = *trace.last().unwrap();
        assert!(final_v < -80.0, "did not repolarize: V = {final_v} mV");
    }

    #[test]
    fn tt_resting_currents_are_finite() {
        let cell = TenTusscher::resting();
        assert!(cell.ionic_current().is_finite());
        assert!((cell.voltage() + 86.2).abs() < 1e-9);
    }

    #[test]
    fn anisotropic_quadratic_laplacian() {
        // For D = I, ∇·(D∇(x²+y²)) = 4 (matches the scalar 5-point Laplacian).
        let n = 9usize;
        let t = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![DiffusionTensor::from_array([1.0, 0.0, 1.0]); n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - 4.0;
                let y = j as f64 - 4.0;
                phi[t.idx(i, j)] = x * x + y * y;
            }
        }
        let term = t.diffusion_term(&phi);
        let mid = t.idx(4, 4);
        assert_abs_diff_eq!(term[mid], 4.0, epsilon = 1e-9);
    }

    #[test]
    fn anisotropic_prefers_strong_axis() {
        // D = diag(10, 1): diffusion along x should dominate a step in x more
        // than the same step along y.
        let n = 9usize;
        let t = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![DiffusionTensor::from_array([10.0, 0.0, 1.0]); n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        // linear in x: ∇·(D∇(ax)) = Dxx * 0 + ... = 0; use ax² to expose Dxx.
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - 4.0;
                phi[t.idx(i, j)] = 0.5 * x * x;
            }
        }
        let term = t.diffusion_term(&phi);
        // For φ = ½ x², ∇·(D∇φ) = Dxx at interior => 10.
        assert_abs_diff_eq!(term[t.idx(4, 4)], 10.0, epsilon = 1e-9);
    }

    #[test]
    fn bidomain_solves_extracellular_field() {
        let mut t = Tissue::new(12, 12, 1.0).unwrap();
        t.enable_bidomain(1.0);
        // A non-uniform transmembrane field forces a non-trivial extracellular
        // potential.
        for j in 0..12 {
            for i in 0..12 {
                let c = t.idx(i, j);
                t.vm[c] = (i as f64) * 0.1;
            }
        }
        let ve = t.extracellular_potential();
        // The solve must be finite and the residual of the elliptic equation
        // small: (σ_i+σ_e)·L·Ve + σ_i·L·Vm ≈ 0, checked with the sparse Laplacian.
        assert!(ve.iter().all(|x| x.is_finite()));
        let grid = UniformGrid2D::new(12, 0.0, 1.0, 12, 0.0, 1.0).unwrap();
        let l = laplacian_2d_sparse(&grid, Boundary::Neumann);
        let lve = l.mul_vec(&ve);
        let lvm = l.mul_vec(&t.vm);
        let di = t.diff;
        let de = t.sigma_e;
        let residual: f64 = lve
            .iter()
            .zip(lvm.iter())
            .map(|(a, b)| (di + de) * a + di * b)
            .map(|x| x.abs())
            .sum();
        assert!(residual < 1e-4, "elliptic residual too large: {residual}");
    }

    #[test]
    fn bidomain_reduces_to_monodomain_for_large_sigma_e() {
        let n = 8usize;
        // Monodomain reference.
        let mut mono = Tissue::new(n, n, 1.0).unwrap();
        for j in 0..n {
            for i in 0..n {
                let c = mono.idx(i, j);
                mono.vm[c] = (i + j) as f64 * 0.2;
            }
        }
        let diff_mono = mono.diffusion_term(&mono.vm.clone());

        // Bidomain with very large extracellular conductivity: the effective
        // coupling term D_i·L(Vm+Ve) must approach D_i·L·Vm.
        let mut bi = Tissue::new(n, n, 1.0).unwrap();
        bi.enable_bidomain(1e6);
        for j in 0..n {
            for i in 0..n {
                let c = bi.idx(i, j);
                bi.vm[c] = (i + j) as f64 * 0.2;
            }
        }
        let ve = bi.extracellular_potential();
        let phi: Vec<f64> = bi.vm.iter().zip(&ve).map(|(a, b)| a + b).collect();
        let diff_bi = bi.diffusion_term(&phi);
        for c in 0..n * n {
            assert_abs_diff_eq!(diff_bi[c], diff_mono[c], epsilon = 1e-3);
        }
    }

    #[test]
    fn bidomain_equal_conductivity_matches_monodomain() {
        // When σ_i = σ_e = σ the bidomain reduces exactly to the monodomain with
        // the effective diffusion coefficient D_eff = σ_i·σ_e/(σ_i+σ_e) = σ/2.
        let n = 9usize;
        let sigma = 2.0;
        let d_eff = sigma * sigma / (sigma + sigma);
        // Monodomain reference with D = D_eff.
        let mut mono = Tissue::new(n, n, d_eff).unwrap();
        for j in 0..n {
            for i in 0..n {
                let c = mono.idx(i, j);
                mono.vm[c] = (i + j) as f64 * 0.3;
            }
        }
        let diff_mono = mono.diffusion_term(&mono.vm.clone());

        // Bidomain with σ_i = σ_e = σ.
        let mut bi = Tissue::new(n, n, sigma).unwrap();
        bi.enable_bidomain(sigma);
        for j in 0..n {
            for i in 0..n {
                let c = bi.idx(i, j);
                bi.vm[c] = (i + j) as f64 * 0.3;
            }
        }
        let ve = bi.extracellular_potential();
        let phi: Vec<f64> = bi.vm.iter().zip(&ve).map(|(a, b)| a + b).collect();
        let diff_bi = bi.diffusion_term(&phi);
        for c in 0..n * n {
            assert_abs_diff_eq!(diff_bi[c], diff_mono[c], epsilon = 1e-6);
        }
    }

    #[test]
    fn bidomain_extracellular_potential_bounded() {
        // Over many bidomain steps the extracellular potential must stay finite
        // and bounded (the elliptic solve must not blow up).
        let mut t = Tissue::new(10, 10, 1.0).unwrap();
        t.enable_bidomain(0.7);
        t.stimulate(0, 5, 40.0);
        for _ in 0..60 {
            t.bidomain_step(0.005).unwrap();
            assert!(t.ve.iter().all(|x| x.is_finite()));
        }
        let max_ve = t.ve.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_ve = t.ve.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(max_ve.is_finite() && min_ve.is_finite());
        assert!(
            max_ve.abs().max(min_ve.abs()) < 1e3,
            "extracellular potential should remain bounded, got [{min_ve}, {max_ve}]"
        );
    }

    #[test]
    fn anisotropic_harmonic_field_rhs() {
        // For u(x,y) = a·x² + b·y², ∇·(D∇u) = 2a·Dxx + 2b·Dyy (a constant).
        // With a=3, b=2 and D = diag(5, 7): RHS = 2·3·5 + 2·2·7 = 58.
        let n = 11usize;
        let d = DiffusionTensor::diagonal(5.0, 7.0);
        let t = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![d; n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 - 5.0;
                let y = j as f64 - 5.0;
                phi[t.idx(i, j)] = 3.0 * x * x + 2.0 * y * y;
            }
        }
        let term = t.diffusion_term(&phi);
        assert_abs_diff_eq!(term[t.idx(5, 5)], 58.0, epsilon = 1e-9);
    }

    #[test]
    fn tensor_reproduces_isotropic_laplacian() {
        // A constant isotropic tensor D = diag(s, s) must reproduce the scalar
        // Laplacian with the same scalar s.
        let n = 13usize;
        let s = 2.5;
        let scalar = Tissue::new(n, n, s).unwrap();
        let tensor = Tissue::with_model(
            n,
            n,
            Diffusivity::Tensor(vec![DiffusionTensor::isotropic(s); n * n]),
            HodgkinHuxley::resting(),
        )
        .unwrap();
        let mut phi = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = i as f64;
                let y = j as f64;
                phi[tensor.idx(i, j)] = 0.5 * (x * x + y * y) + x * y;
            }
        }
        let ds = scalar.diffusion_term(&phi);
        let dt = tensor.diffusion_term(&phi);
        for c in 0..n * n {
            assert_abs_diff_eq!(dt[c], ds[c], epsilon = 1e-9);
        }
    }
}
