//! Molecular descriptors computed from a [`Molecule`].
//!
//! The logP and TPSA values are *approximations* in the spirit of Wildman–Crippen
//! and Ertl, with a reduced atom-type table. They are fit for ranking and for use
//! as QSPR features, not as a substitute for the published reference values.

use crate::mol::Molecule;
use std::collections::VecDeque;

/// Names of the descriptors, in the fixed order of [`DescriptorSet::as_slice`].
pub const DESCRIPTOR_NAMES: [&str; 18] = [
    "mw",
    "heavy_atoms",
    "hbd",
    "hba",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "rings",
    "aromatic_rings",
    "frac_sp3",
    "formal_charge",
    "cc_double_bonds",
    "amides",
    "phenol_oh",
    "carbonyls",
    "halogens",
    "wiener",
    "polar_fraction",
];

/// A fixed-order vector of descriptor values for one molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorSet {
    values: Vec<f64>,
}

impl DescriptorSet {
    /// Compute all descriptors for `mol`.
    pub fn compute(mol: &Molecule) -> DescriptorSet {
        let tpsa = tpsa(mol);
        let mw = mol.molecular_weight();
        let aromatic_rings = mol
            .rings
            .iter()
            .filter(|r| r.iter().all(|&i| mol.atoms[i].aromatic))
            .count();
        let values = vec![
            mw,
            mol.num_heavy_atoms() as f64,
            hbd(mol) as f64,
            hba(mol) as f64,
            rotatable_bonds(mol) as f64,
            tpsa,
            logp(mol),
            mol.rings.len() as f64,
            aromatic_rings as f64,
            frac_sp3(mol),
            mol.atoms.iter().map(|a| f64::from(a.charge)).sum(),
            cc_double_bonds(mol) as f64,
            amides(mol) as f64,
            phenol_oh(mol) as f64,
            carbonyls(mol) as f64,
            mol.atoms
                .iter()
                .filter(|a| matches!(a.z, 9 | 17 | 35 | 53))
                .count() as f64,
            wiener(mol),
            if mw > 0.0 { tpsa / mw } else { 0.0 },
        ];
        DescriptorSet { values }
    }

    /// Descriptor values in the order of [`DESCRIPTOR_NAMES`].
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Look up a descriptor by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        DESCRIPTOR_NAMES
            .iter()
            .position(|n| *n == name)
            .map(|i| self.values[i])
    }
}

fn is_no(z: u8) -> bool {
    matches!(z, 7 | 8)
}

/// Hydrogen-bond donors: N/O atoms carrying at least one hydrogen.
pub fn hbd(mol: &Molecule) -> usize {
    mol.atoms
        .iter()
        .filter(|a| is_no(a.z) && a.total_h() > 0)
        .count()
}

/// Hydrogen-bond acceptors (Lipinski definition): all N and O atoms.
pub fn hba(mol: &Molecule) -> usize {
    mol.atoms.iter().filter(|a| is_no(a.z)).count()
}

/// Rotatable bonds: acyclic single bonds between two non-terminal heavy atoms,
/// neither of which is part of a triple bond.
pub fn rotatable_bonds(mol: &Molecule) -> usize {
    let has_triple = |i: usize| mol.adj[i].iter().any(|&(_, b)| mol.bonds[b].order == 3);
    mol.bonds
        .iter()
        .enumerate()
        .filter(|(bi, b)| {
            b.order == 1
                && !b.aromatic
                && !mol.bond_in_ring(*bi)
                && mol.heavy_degree(b.a) > 1
                && mol.heavy_degree(b.b) > 1
                && !has_triple(b.a)
                && !has_triple(b.b)
        })
        .count()
}

fn double_to(mol: &Molecule, i: usize, z: u8) -> bool {
    mol.adj[i]
        .iter()
        .any(|&(n, b)| mol.bonds[b].order == 2 && !mol.bonds[b].aromatic && mol.atoms[n].z == z)
}

fn has_double(mol: &Molecule, i: usize) -> bool {
    mol.adj[i]
        .iter()
        .any(|&(_, b)| mol.bonds[b].order == 2 && !mol.bonds[b].aromatic)
}

fn has_aromatic_bond(mol: &Molecule, i: usize) -> bool {
    mol.adj[i].iter().any(|&(_, b)| mol.bonds[b].aromatic)
}

/// Approximate topological polar surface area (Å²), Ertl-style N/O/S/P
/// fragment contributions with a reduced type table.
pub fn tpsa(mol: &Molecule) -> f64 {
    let mut total = 0.0;
    for (i, a) in mol.atoms.iter().enumerate() {
        let h = a.total_h();
        let deg = mol.heavy_degree(i);
        let triple = mol.adj[i].iter().any(|&(_, b)| mol.bonds[b].order == 3);
        total += match a.z {
            7 if a.aromatic => {
                if h > 0 {
                    15.79
                } else if deg == 3 {
                    4.93
                } else {
                    12.89
                }
            }
            7 if triple => 23.79,
            7 if has_double(mol, i) => {
                if h > 0 {
                    23.85
                } else {
                    12.36
                }
            }
            7 => match h {
                2.. => 26.02,
                1 => 12.03,
                _ => 3.24,
            },
            8 if a.aromatic => 13.14,
            8 if has_double(mol, i) => 17.07,
            8 => {
                if h > 0 {
                    20.23
                } else {
                    9.23
                }
            }
            16 if h > 0 && !a.aromatic => 38.80,
            15 => 13.59,
            _ => 0.0,
        };
    }
    total
}

/// Approximate Crippen-style octanol/water logP (atom contributions incl. H).
pub fn logp(mol: &Molecule) -> f64 {
    let mut total = 0.0;
    for (i, a) in mol.atoms.iter().enumerate() {
        let h = f64::from(a.total_h());
        let hetero_nb = mol.adj[i]
            .iter()
            .filter(|&&(n, _)| is_no(mol.atoms[n].z))
            .count();
        let (base, h_contrib) = match a.z {
            6 if a.aromatic => (0.2955, 0.1230),
            6 if double_to(mol, i, 8) => (-0.2783, 0.1230),
            6 if has_double(mol, i) => (0.2640, 0.1230),
            6 if mol.adj[i].iter().any(|&(_, b)| mol.bonds[b].order == 3) => (0.3000, 0.1230),
            6 => (0.1441 - 0.20 * hetero_nb as f64, 0.1230),
            7 if a.aromatic => (-0.3239, -0.2677),
            7 if has_double(mol, i) || has_aromatic_bond(mol, i) => (-0.3187, -0.2677),
            7 if mol.adj[i].iter().any(|&(n, _)| double_to(mol, n, 8)) => (-0.4458, -0.2677),
            7 => (if a.total_h() >= 2 { -1.0190 } else { -0.7096 }, -0.2677),
            8 if a.aromatic => (0.1552, 0.0),
            8 if has_double(mol, i) => (-0.1526, 0.0),
            8 if a.total_h() > 0 => (-0.2893, -0.2677),
            8 => (-0.0684, 0.0),
            9 => (0.4202, 0.0),
            15 => (0.8612, 0.0),
            16 => (0.6482, -0.2677),
            17 => (0.6895, 0.0),
            35 => (0.8456, 0.0),
            53 => (0.8857, 0.0),
            1 => (0.0, 0.0),
            _ => (0.0, 0.0),
        };
        total += base + h * h_contrib;
    }
    total
}

/// Fraction of carbons that are sp3 (no double/triple/aromatic bonds).
pub fn frac_sp3(mol: &Molecule) -> f64 {
    let carbons: Vec<usize> = (0..mol.atoms.len())
        .filter(|&i| mol.atoms[i].z == 6)
        .collect();
    if carbons.is_empty() {
        return 0.0;
    }
    let sp3 = carbons
        .iter()
        .filter(|&&i| {
            !mol.atoms[i].aromatic && mol.adj[i].iter().all(|&(_, b)| mol.bonds[b].order == 1)
        })
        .count();
    sp3 as f64 / carbons.len() as f64
}

/// Non-aromatic C=C double bonds (the polyene motif of the sanshools).
pub fn cc_double_bonds(mol: &Molecule) -> usize {
    mol.bonds
        .iter()
        .filter(|b| b.order == 2 && !b.aromatic && mol.atoms[b.a].z == 6 && mol.atoms[b.b].z == 6)
        .count()
}

/// Carbonyl groups (C=O).
pub fn carbonyls(mol: &Molecule) -> usize {
    (0..mol.atoms.len())
        .filter(|&i| mol.atoms[i].z == 6 && double_to(mol, i, 8))
        .count()
}

/// Amide groups: C(=O)–N.
pub fn amides(mol: &Molecule) -> usize {
    (0..mol.atoms.len())
        .filter(|&i| {
            mol.atoms[i].z == 6
                && double_to(mol, i, 8)
                && mol.adj[i].iter().any(|&(n, b)| {
                    mol.atoms[n].z == 7 && mol.bonds[b].order == 1 && !mol.bonds[b].aromatic
                })
        })
        .count()
}

/// Phenolic hydroxyls: OH attached to an aromatic carbon.
pub fn phenol_oh(mol: &Molecule) -> usize {
    (0..mol.atoms.len())
        .filter(|&i| {
            mol.atoms[i].z == 8
                && mol.atoms[i].total_h() > 0
                && mol.adj[i]
                    .iter()
                    .any(|&(n, _)| mol.atoms[n].z == 6 && mol.atoms[n].aromatic)
        })
        .count()
}

/// Wiener index: sum of shortest-path lengths over all heavy-atom pairs of the
/// same connected component.
pub fn wiener(mol: &Molecule) -> f64 {
    let n = mol.atoms.len();
    let mut total = 0u64;
    for s in 0..n {
        if mol.atoms[s].z == 1 {
            continue;
        }
        let mut dist = vec![u32::MAX; n];
        dist[s] = 0;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &mol.adj[u] {
                if dist[v] == u32::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        total += (s + 1..n)
            .filter(|&t| mol.atoms[t].z != 1 && dist[t] != u32::MAX)
            .map(|t| u64::from(dist[t]))
            .sum::<u64>();
    }
    total as f64
}
