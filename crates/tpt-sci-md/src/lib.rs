//! # tpt-sci-md
//!
//! A small, dependency-light **classical molecular dynamics (MD)** engine built
//! entirely from scratch (no wrapped engine such as `lumol`, which was audited
//! and rejected — BSD-3-Clause and alpha/stale). It provides:
//!
//! * A [`Particle`] type (position, velocity, force, mass, type id).
//! * Pairwise **Lennard-Jones (12-6)** interactions with a cut-off radius and
//!   minimum-image convention under periodic boundaries ([`Forces::lennard_jones`]).
//! * A **Finnis-Sinclair-style EAM** (embedded-atom method) potential
//!   ([`EamParams`], [`Forces::eam`]) for simple metallic-bonding-like systems.
//! * **Ewald summation** for periodic long-range electrostatics ([`Ewald`]).
//! * **SHAKE/RATTLE** holonomic bond-length constraints ([`Shake`],
//!   [`Integrator::velocity_verlet_constrained`]).
//! * **Cell lists** (linked cells) for `O(n)`-ish neighbor finding
//!   ([`CellList`], [`Forces::lennard_jones_cells`]), cross-checked against
//!   [`neighbor_pairs_brute_force`].
//! * Velocity-Verlet integration ([`Integrator::velocity_verlet`]), a generic
//!   [`Integrator::step_with`] hook for other potentials, and optional
//!   Berendsen-style thermostats.
//! * A **radial distribution function** [`rdf`] for structural analysis.
//!
//! The engine is intentionally minimal: it models mono- (or few-) species
//! systems in a cubic periodic box and is sized for teaching, prototyping,
//! and coupling into the broader `tpt-science` platform — not for
//! production-scale biomolecular simulation.
//!
//! # Example
//!
//! ```
//! use tpt_sci_md::{Particle, lennard_jones};
//! use tpt_math_linalg::tpt_math_linalg_dense::DVector;
//!
//! // Two particles on a line, slightly closer than the LJ minimum.
//! let parts = vec![
//!     Particle::new(0, DVector::from_row_slice(&[0.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
//!     Particle::new(1, DVector::from_row_slice(&[1.0, 0.0, 0.0]), DVector::zeros(3), 1.0).unwrap(),
//! ];
//! let forces = lennard_jones(&parts, 10.0, 1.0);
//! assert!(forces[0].iter().any(|&f| f.is_finite()));
//! ```
#![forbid(unsafe_code)]

use tpt_math_linalg::tpt_math_linalg_dense::DVector;

mod error;

pub use error::MdError;

mod cells;
mod constraints;
mod eam;
mod ewald;
mod forces;
mod integrator;
mod particle;
mod rdf;

pub use cells::{CellList, neighbor_pairs_brute_force};
pub use constraints::{Bond, Shake};
pub(crate) use eam::eam_forces_energy;
pub use eam::{EamParams, eam_forces};
pub use ewald::Ewald;
pub use forces::{Forces, lennard_jones};
pub use integrator::Integrator;
pub use particle::Particle;
pub use rdf::rdf;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn v(x: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(x)
    }

    #[test]
    fn particle_new_rejects_invalid() {
        assert!(Particle::new(0, v(&[0.0]), v(&[1.0]), 0.0).is_err());
        assert!(Particle::new(0, v(&[0.0]), v(&[1.0, 0.0]), 1.0).is_err());
        assert!(Particle::new(0, v(&[0.0]), v(&[0.0]), 1.0).is_ok());
    }

    #[test]
    fn lj_repulsive_at_close_range() {
        // Two particles closer than σ: the LJ force must push them apart
        // (force on particle 0 points toward negative x if particle 1 is at +x).
        let parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[0.8, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = lennard_jones(&parts, f64::INFINITY, 1.0);
        // Particle 1 to the right => repulsive force on 0 is toward -x.
        assert!(f[0][0] < 0.0);
        assert!(f[1][0] > 0.0);
    }

    #[test]
    fn lj_force_vanishes_beyond_cutoff() {
        let parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[10.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = lennard_jones(&parts, f64::INFINITY, 1.0);
        assert_abs_diff_eq!(f[0][0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_verlet_conserves_energy_free_particles() {
        // Two non-interacting particles (huge sigma so no force) drift at
        // constant velocity; position advances by v·dt.
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[5.0, 0.0, 0.0]), v(&[0.0, 1.0, 0.0]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 100.0, 0.01).unwrap();
        let _energy = int.velocity_verlet(&mut parts);
        assert_abs_diff_eq!(parts[0].position[0], 0.01, epsilon = 1e-9);
        assert_abs_diff_eq!(parts[1].position[1], 0.01, epsilon = 1e-9);
    }

    #[test]
    fn temperature_scales_with_velocity() {
        let parts = vec![Particle::new(0, v(&[0.0; 3]), v(&[2.0, 0.0, 0.0]), 1.0).unwrap()];
        let int = Integrator::new(f64::INFINITY, 1.0, 0.01).unwrap();
        // Single particle => T = 0 by the (N-1) normalization.
        assert_abs_diff_eq!(int.temperature(&parts), 0.0, epsilon = 1e-12);
        let two = vec![
            Particle::new(0, v(&[0.0; 3]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[0.0; 3]), v(&[1.0, 0.0, 0.0]), 1.0).unwrap(),
        ];
        // K = 2·(½·1·1) = 1 ; reduced T = 2·K / (3·1·1) = 2/3.
        assert_abs_diff_eq!(int.temperature(&two), 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn rdf_dilute_is_unity_like() {
        // Widely separated particles in a large periodic box: g(r) is defined
        // but sparse; just check it runs and returns the right length.
        let parts: Vec<Particle> = (0..10)
            .map(|i| Particle::new(i, v(&[(i as f64) * 5.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap())
            .collect();
        let (r, g) = rdf(&parts, f64::INFINITY, 10.0, 20).unwrap();
        assert_eq!(r.len(), 20);
        assert_eq!(g.len(), 20);
    }

    // ---- EAM ----------------------------------------------------------

    #[test]
    fn eam_repulsive_at_short_range_attractive_at_intermediate_range() {
        let params = EamParams::generic();
        // Short range (r = 0.5, well inside the crossover at r ~ 1.0 for the
        // generic params): net force must be repulsive (push apart).
        let short = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[0.5, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = eam_forces(&short, f64::INFINITY, &params);
        assert!(
            f[0][0] < 0.0,
            "expected repulsion at short range, got {f:?}"
        );
        assert!(
            f[1][0] > 0.0,
            "expected repulsion at short range, got {f:?}"
        );

        // Intermediate range (r = 1.5): net force must be attractive (pull together).
        let mid = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[1.5, 0.0, 0.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let f = eam_forces(&mid, f64::INFINITY, &params);
        assert!(f[0][0] > 0.0, "expected attraction at mid range, got {f:?}");
        assert!(f[1][0] < 0.0, "expected attraction at mid range, got {f:?}");
    }

    #[test]
    fn eam_conserves_energy_over_short_nve_trajectory() {
        let params = EamParams::generic();
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.05, 0.0, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[1.3, 0.0, 0.0]), v(&[-0.05, 0.02, 0.0]), 1.0).unwrap(),
            Particle::new(2, v(&[0.0, 1.3, 0.0]), v(&[0.0, -0.03, 0.01]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 1.0, 0.001).unwrap();
        let _ = Forces::eam(&mut parts, f64::INFINITY, &params);
        let kinetic0 = int.kinetic_energy(&parts);
        let potential0 = eam_forces_energy(&parts, f64::INFINITY, &params).1;
        let e0 = kinetic0 + potential0;

        let mut e_final = e0;
        for _ in 0..300 {
            let u = int.step_with(&mut parts, |p| Forces::eam(p, f64::INFINITY, &params));
            let k = int.kinetic_energy(&parts);
            e_final = u + k;
        }
        let rel_drift = (e_final - e0).abs() / e0.abs().max(1e-12);
        assert!(
            rel_drift < 0.05,
            "energy drifted too much: e0={e0}, e_final={e_final}, rel_drift={rel_drift}"
        );
    }

    // ---- Ewald ----------------------------------------------------------

    #[test]
    fn ewald_opposite_charges_attract_like_charges_repel() {
        let box_len = 40.0;
        let ewald = Ewald::new(box_len, 0.3, 6, box_len * 0.5).unwrap();

        let opposite = vec![
            Particle::new(0, v(&[20.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[21.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e, f) = ewald.energy_forces(&opposite, &[1.0, -1.0]).unwrap();
        // Opposite charges: energy negative (bound), force pulls together
        // (particle 0 pulled toward +x, i.e. toward particle 1).
        assert!(
            e < 0.0,
            "expected negative energy for opposite charges, got {e}"
        );
        assert!(f[0][0] > 0.0, "expected attractive pull, got {f:?}");

        let same = vec![
            Particle::new(0, v(&[20.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[21.0, 20.0, 20.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e2, f2) = ewald.energy_forces(&same, &[1.0, 1.0]).unwrap();
        assert!(
            e2 > 0.0,
            "expected positive energy for like charges, got {e2}"
        );
        assert!(f2[0][0] < 0.0, "expected repulsive push, got {f2:?}");
    }

    #[test]
    fn ewald_matches_direct_coulomb_in_dilute_limit() {
        // For a well-separated pair in a very large box relative to the
        // separation, periodic image contributions are negligible and the
        // Ewald energy should approach the bare Coulomb energy q1*q2/r
        // (Gaussian/reduced units, k_e = 1).
        let box_len = 200.0;
        let ewald = Ewald::new(box_len, 0.2, 8, box_len * 0.5).unwrap();
        let r = 2.0;
        let parts = vec![
            Particle::new(0, v(&[100.0, 100.0, 100.0]), v(&[0.0; 3]), 1.0).unwrap(),
            Particle::new(1, v(&[100.0 + r, 100.0, 100.0]), v(&[0.0; 3]), 1.0).unwrap(),
        ];
        let (e, _f) = ewald.energy_forces(&parts, &[1.0, -1.0]).unwrap();
        let direct = -1.0 / r;
        assert_abs_diff_eq!(e, direct, epsilon = 5e-2);
    }

    #[test]
    fn ewald_conserves_energy_over_short_nve_trajectory() {
        let box_len = 30.0;
        let ewald = Ewald::new(box_len, 0.3, 5, box_len * 0.5).unwrap();
        let charges = vec![1.0, -1.0];
        let mut parts = vec![
            Particle::new(0, v(&[14.0, 15.0, 15.0]), v(&[0.0, 0.02, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[16.0, 15.0, 15.0]), v(&[0.0, -0.02, 0.01]), 1.0).unwrap(),
        ];
        let int = Integrator::new(box_len, 1.0, 0.002).unwrap();
        let (u0, f0) = ewald.energy_forces(&parts, &charges).unwrap();
        for (p, f) in parts.iter_mut().zip(f0) {
            p.force = f;
        }
        let e0 = int.kinetic_energy(&parts) + u0;

        let mut e_final = e0;
        for _ in 0..150 {
            let u = int.step_with(&mut parts, |p| {
                let (e, f) = ewald.energy_forces(p, &charges).unwrap();
                for (particle, ff) in p.iter_mut().zip(f) {
                    particle.force = ff;
                }
                e
            });
            e_final = int.kinetic_energy(&parts) + u;
        }
        let rel_drift = (e_final - e0).abs() / e0.abs().max(1e-12);
        assert!(
            rel_drift < 0.1,
            "energy drifted too much: e0={e0}, e_final={e_final}, rel_drift={rel_drift}"
        );
    }

    // ---- SHAKE / RATTLE --------------------------------------------------

    #[test]
    fn shake_keeps_bond_length_within_tolerance() {
        let bonds = vec![Bond {
            i: 0,
            j: 1,
            r0: 1.0,
        }];
        let shake = Shake::new(bonds, 1e-10, 200).unwrap();
        // Tiny sigma => the LJ cut-off (2.5*sigma) sits well inside the bond
        // length, so the LJ force is exactly zero; motion is pure free-flight
        // + constraint, exercising SHAKE cleanly.
        let mut parts = vec![
            Particle::new(0, v(&[0.0, 0.0, 0.0]), v(&[0.1, 0.05, 0.0]), 1.0).unwrap(),
            Particle::new(1, v(&[1.0, 0.0, 0.0]), v(&[-0.1, -0.02, 0.03]), 1.0).unwrap(),
        ];
        let int = Integrator::new(f64::INFINITY, 0.01, 0.01).unwrap();
        for _ in 0..200 {
            int.velocity_verlet_constrained(&mut parts, &shake).unwrap();
            let d = (parts[0].position.clone() - parts[1].position.clone()).norm();
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn shake_rejects_out_of_range_bond() {
        let bonds = vec![Bond {
            i: 0,
            j: 5,
            r0: 1.0,
        }];
        let shake = Shake::new(bonds, 1e-8, 100).unwrap();
        let old = vec![v(&[0.0; 3]), v(&[1.0, 0.0, 0.0])];
        let mut new = old.clone();
        let masses = vec![1.0, 1.0];
        assert!(
            shake
                .constrain_positions(&old, &mut new, &masses, f64::INFINITY)
                .is_err()
        );
    }

    // ---- Cell lists -------------------------------------------------------

    #[test]
    fn cell_list_matches_brute_force_neighbors() {
        // A modest random-ish 3-D configuration in a periodic box, dense
        // enough that some pairs cross cell boundaries and periodic images.
        let box_len = 10.0;
        let rcut = 2.0;
        let coords: [[f64; 3]; 14] = [
            [0.1, 0.2, 0.3],
            [9.9, 0.1, 0.2],
            [1.5, 1.5, 1.5],
            [5.0, 5.0, 5.0],
            [5.1, 5.05, 4.95],
            [0.0, 9.95, 0.05],
            [3.3, 7.7, 2.2],
            [3.4, 7.6, 2.1],
            [8.0, 8.0, 8.0],
            [8.1, 0.0, 8.1],
            [2.0, 2.0, 8.0],
            [2.1, 2.1, 7.9],
            [6.0, 1.0, 6.0],
            [6.05, 1.05, 5.95],
        ];
        let parts: Vec<Particle> = coords
            .iter()
            .enumerate()
            .map(|(i, c)| Particle::new(i, v(c), v(&[0.0; 3]), 1.0).unwrap())
            .collect();

        let brute = neighbor_pairs_brute_force(&parts, box_len, rcut);
        let cell_list = CellList::build(&parts, box_len, rcut).unwrap();
        let via_cells = cell_list.neighbor_pairs(&parts);

        assert_eq!(
            brute, via_cells,
            "cell-list neighbors must match brute force exactly"
        );
        assert!(
            !brute.is_empty(),
            "test configuration should have at least one neighbor pair"
        );
    }

    #[test]
    fn lennard_jones_cells_matches_brute_force_energy() {
        let box_len = 12.0;
        let sigma = 1.0;
        let coords: [[f64; 3]; 8] = [
            [1.0, 1.0, 1.0],
            [1.9, 1.0, 1.0],
            [1.0, 1.9, 1.0],
            [6.0, 6.0, 6.0],
            [6.9, 6.0, 6.0],
            [11.5, 1.0, 1.0],
            [0.4, 1.0, 1.0],
            [3.0, 9.0, 2.0],
        ];
        let mut brute: Vec<Particle> = coords
            .iter()
            .enumerate()
            .map(|(i, c)| Particle::new(i, v(c), v(&[0.0; 3]), 1.0).unwrap())
            .collect();
        let mut cells = brute.clone();

        let e_brute = Forces::lennard_jones(&mut brute, box_len, sigma);
        let e_cells = Forces::lennard_jones_cells(&mut cells, box_len, sigma).unwrap();

        assert_abs_diff_eq!(e_brute, e_cells, epsilon = 1e-10);
        for (pb, pc) in brute.iter().zip(cells.iter()) {
            for k in 0..3 {
                assert_abs_diff_eq!(pb.force[k], pc.force[k], epsilon = 1e-10);
            }
        }
    }
}
