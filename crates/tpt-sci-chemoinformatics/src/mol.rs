//! Molecular graph: atoms, bonds, implicit hydrogens, rings and aromaticity.

use crate::elements::{self, ElementData, H_MASS};
use crate::error::{ChemoinformaticsError, Result};
use std::collections::{BTreeMap, VecDeque};

/// Tetrahedral chirality tag as written in SMILES (`@` / `@@`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chirality {
    /// `@` — counter-clockwise.
    Ccw,
    /// `@@` — clockwise.
    Cw,
}

/// Directional bond marker (`/` or `\`), used for E/Z double-bond geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondDir {
    /// `/`
    Up,
    /// `\`
    Down,
}

/// One atom of a [`Molecule`].
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Atomic number.
    pub z: u8,
    /// Whether the atom is aromatic.
    pub aromatic: bool,
    /// Formal charge.
    pub charge: i8,
    /// Isotope mass number, if specified.
    pub isotope: Option<u16>,
    /// Explicit hydrogen count from a bracket atom (`[nH]`), if bracketed.
    pub explicit_h: Option<u8>,
    /// Implicit hydrogens inferred from valence (0 for bracket atoms).
    pub implicit_h: u8,
    /// Tetrahedral chirality tag, if any.
    pub chirality: Option<Chirality>,
    /// Atom-map class from a bracket atom (`[C:1]`), if any.
    pub class: Option<u32>,
    pub(crate) bracket: bool,
    pub(crate) lowercase: bool,
}

impl Atom {
    pub(crate) fn new(z: u8) -> Self {
        Atom {
            z,
            aromatic: false,
            charge: 0,
            isotope: None,
            explicit_h: None,
            implicit_h: 0,
            chirality: None,
            class: None,
            bracket: false,
            lowercase: false,
        }
    }

    /// Element data for this atom.
    ///
    /// # Panics
    /// Never for atoms produced by this crate, whose `z` always comes from the
    /// element table.
    pub fn element(&self) -> &'static ElementData {
        elements::by_z(self.z).expect("atoms are only created from the element table")
    }

    /// Total attached hydrogens (explicit + implicit).
    pub fn total_h(&self) -> u8 {
        self.explicit_h.unwrap_or(0) + self.implicit_h
    }
}

/// One bond of a [`Molecule`].
#[derive(Debug, Clone, PartialEq)]
pub struct Bond {
    /// First atom index.
    pub a: usize,
    /// Second atom index.
    pub b: usize,
    /// Kekulé-style order (1, 2 or 3). Aromatic bonds from lowercase SMILES
    /// atoms carry order 1 with [`Bond::aromatic`] set.
    pub order: u8,
    /// Whether the bond is aromatic.
    pub aromatic: bool,
    /// Directional marker (`/`, `\`), if any.
    pub dir: Option<BondDir>,
    pub(crate) implicit: bool,
}

impl Bond {
    /// The atom at the other end of the bond from `x`.
    pub fn other(&self, x: usize) -> usize {
        if self.a == x { self.b } else { self.a }
    }
}

/// A molecular graph built from a SMILES string.
#[derive(Debug, Clone, PartialEq)]
pub struct Molecule {
    /// Atoms in SMILES input order.
    pub atoms: Vec<Atom>,
    /// Bonds in creation order.
    pub bonds: Vec<Bond>,
    /// `adj[i]` lists `(neighbour, bond index)` pairs.
    pub adj: Vec<Vec<(usize, usize)>>,
    /// Smallest set of smallest rings, each as an ordered atom cycle.
    pub rings: Vec<Vec<usize>>,
    pub(crate) bond_in_ring: Vec<bool>,
}

impl Molecule {
    /// Parse a SMILES string. See [`crate::smiles`] for the supported subset.
    ///
    /// # Errors
    /// Returns an error for malformed, over-valent or unsupported SMILES.
    pub fn from_smiles(smiles: &str) -> Result<Molecule> {
        crate::smiles::parse(smiles)
    }

    pub(crate) fn from_parts(atoms: Vec<Atom>, bonds: Vec<Bond>) -> Result<Molecule> {
        let mut adj = vec![Vec::new(); atoms.len()];
        for (bi, b) in bonds.iter().enumerate() {
            adj[b.a].push((b.b, bi));
            adj[b.b].push((b.a, bi));
        }
        let mut m = Molecule {
            atoms,
            bonds,
            adj,
            rings: Vec::new(),
            bond_in_ring: Vec::new(),
        };
        m.perceive_rings();
        // An implicit bond between two aromatic atoms that is not in a ring
        // (e.g. biphenyl's linking bond) is a single bond.
        for bi in 0..m.bonds.len() {
            if m.bonds[bi].implicit && m.bonds[bi].aromatic && !m.bond_in_ring[bi] {
                m.bonds[bi].aromatic = false;
            }
        }
        m.assign_implicit_h()?;
        m.perceive_aromaticity();
        Ok(m)
    }

    /// Number of atoms including explicit `[H]` atoms.
    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Number of heavy (non-hydrogen) atoms.
    pub fn num_heavy_atoms(&self) -> usize {
        self.atoms.iter().filter(|a| a.z != 1).count()
    }

    /// Number of heavy-atom neighbours of atom `i`.
    pub fn heavy_degree(&self, i: usize) -> usize {
        self.adj[i]
            .iter()
            .filter(|(n, _)| self.atoms[*n].z != 1)
            .count()
    }

    /// Whether atom `i` belongs to at least one ring.
    pub fn atom_in_ring(&self, i: usize) -> bool {
        self.adj[i].iter().any(|&(_, b)| self.bond_in_ring[b])
    }

    /// Whether bond `b` belongs to at least one ring.
    pub fn bond_in_ring(&self, b: usize) -> bool {
        self.bond_in_ring[b]
    }

    /// Total hydrogens in the molecule: implicit, bracket and explicit `[H]` atoms.
    pub fn total_hydrogens(&self) -> u32 {
        self.atoms
            .iter()
            .map(|a| u32::from(a.total_h()) + u32::from(a.z == 1))
            .sum()
    }

    /// Hill-order molecular formula, e.g. `C16H25NO2`.
    pub fn formula(&self) -> String {
        let mut counts: BTreeMap<&'static str, u32> = BTreeMap::new();
        for a in &self.atoms {
            if a.z != 1 {
                *counts.entry(a.element().symbol).or_insert(0) += 1;
            }
        }
        let h = self.total_hydrogens();
        let mut out = String::new();
        let mut push = |s: &str, n: u32| {
            out.push_str(s);
            if n > 1 {
                out.push_str(&n.to_string());
            }
        };
        if let Some(c) = counts.remove("C") {
            push("C", c);
            if h > 0 {
                push("H", h);
            }
            for (s, n) in counts {
                push(s, n);
            }
        } else {
            let mut all: Vec<(&str, u32)> = counts.into_iter().collect();
            if h > 0 {
                all.push(("H", h));
            }
            all.sort_by_key(|(s, _)| *s);
            for (s, n) in all {
                push(s, n);
            }
        }
        out
    }

    /// Average molecular weight (g/mol), including hydrogens.
    pub fn molecular_weight(&self) -> f64 {
        let heavy: f64 = self.atoms.iter().map(|a| a.element().mass).sum();
        // `[H]` atoms are already counted in `heavy` via their element mass.
        let hyd: f64 = self
            .atoms
            .iter()
            .map(|a| f64::from(a.total_h()))
            .sum::<f64>()
            * H_MASS;
        heavy + hyd
    }

    /// Bond-order sum around atom `i`, counting aromatic bonds as 1.
    fn bond_order_sum(&self, i: usize) -> u32 {
        self.adj[i]
            .iter()
            .map(|&(_, b)| {
                let bd = &self.bonds[b];
                if bd.aromatic { 1 } else { u32::from(bd.order) }
            })
            .sum()
    }

    fn assign_implicit_h(&mut self) -> Result<()> {
        for i in 0..self.atoms.len() {
            if self.atoms[i].bracket {
                self.atoms[i].implicit_h = 0;
                continue;
            }
            let el = self.atoms[i].element();
            let mut sum = self.bond_order_sum(i);
            let aromatic = self.atoms[i].lowercase;
            // An aromatic atom's Kekulé double bond is implicit, so it adds one
            // valence unit. Carbon and boron/phosphorus always need it; nitrogen
            // needs it only when pyridine-like (two neighbours). Pyrrole-type
            // nitrogen with three connections carries no extra valence.
            if aromatic {
                // Count all neighbours, not just aromatic bonds: an N-methyl bond
                // is a plain single bond but still makes the nitrogen pyrrole-type.
                let degree = self.adj[i].len();
                let needs_double = match self.atoms[i].z {
                    7 => degree == 2,
                    5 | 6 | 15 => true,
                    _ => false,
                };
                if needs_double {
                    sum += 1;
                }
            }
            match el.valences.iter().find(|&&v| u32::from(v) >= sum) {
                Some(&v) => self.atoms[i].implicit_h = (u32::from(v) - sum) as u8,
                None if aromatic => self.atoms[i].implicit_h = 0,
                None => {
                    return Err(ChemoinformaticsError::Valence {
                        index: i,
                        symbol: el.symbol.to_string(),
                        valence: sum,
                    });
                }
            }
        }
        Ok(())
    }

    /// Smallest set of smallest rings: shortest cycle through each ring bond,
    /// then a GF(2)-independent subset in order of size.
    fn perceive_rings(&mut self) {
        let nb = self.bonds.len();
        self.bond_in_ring = vec![false; nb];
        let mut candidates: Vec<(Vec<usize>, Vec<usize>)> = Vec::new(); // (atoms, bonds)
        for bi in 0..nb {
            let (a, b) = (self.bonds[bi].a, self.bonds[bi].b);
            if let Some((atoms, bonds)) = self.shortest_path_avoiding(a, b, bi) {
                self.bond_in_ring[bi] = true;
                let mut bonds = bonds;
                bonds.push(bi);
                bonds.sort_unstable();
                if !candidates.iter().any(|(_, cb)| *cb == bonds) {
                    candidates.push((atoms, bonds));
                }
            }
        }
        candidates.sort_by_key(|(atoms, _)| atoms.len());
        let words = nb.div_ceil(64).max(1);
        let mut basis: Vec<Vec<u64>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        for (atoms, bonds) in candidates {
            let mut v = vec![0u64; words];
            for &b in &bonds {
                v[b / 64] |= 1 << (b % 64);
            }
            for (bv, &p) in basis.iter().zip(&pivots) {
                if v[p / 64] >> (p % 64) & 1 == 1 {
                    for (x, y) in v.iter_mut().zip(bv) {
                        *x ^= y;
                    }
                }
            }
            if let Some(p) = (0..nb).find(|&b| v[b / 64] >> (b % 64) & 1 == 1) {
                basis.push(v);
                pivots.push(p);
                self.rings.push(atoms);
            }
        }
    }

    /// BFS path from `a` to `b` that does not use bond `skip`; returns the
    /// atoms along the path and the bonds used.
    fn shortest_path_avoiding(
        &self,
        a: usize,
        b: usize,
        skip: usize,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let n = self.atoms.len();
        let mut prev: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut seen = vec![false; n];
        seen[a] = true;
        let mut q = VecDeque::from([a]);
        while let Some(u) = q.pop_front() {
            if u == b {
                break;
            }
            for &(v, bi) in &self.adj[u] {
                if bi != skip && !seen[v] {
                    seen[v] = true;
                    prev[v] = Some((u, bi));
                    q.push_back(v);
                }
            }
        }
        if !seen[b] {
            return None;
        }
        let (mut atoms, mut bonds) = (vec![b], Vec::new());
        let mut cur = b;
        while let Some((p, bi)) = prev[cur] {
            atoms.push(p);
            bonds.push(bi);
            cur = p;
        }
        atoms.reverse();
        Some((atoms, bonds))
    }

    /// Hückel (4n+2) aromaticity for 5/6-membered rings written in Kekulé form.
    ///
    /// Rings that already contain a lowercase aromatic atom are left as parsed.
    /// This is an approximation: a ring-atom double bond may lie in a fused
    /// neighbour ring, and exocyclic C=X (pyridones etc.) is not treated as
    /// aromatic.
    fn perceive_aromaticity(&mut self) {
        let rings = self.rings.clone();
        for ring in rings.iter().filter(|r| matches!(r.len(), 5 | 6)) {
            if ring
                .iter()
                .any(|&i| self.atoms[i].aromatic || self.atoms[i].lowercase)
            {
                continue;
            }
            let mut electrons = 0u32;
            let mut ok = true;
            for &i in ring {
                match self.pi_electrons(i) {
                    Some(e) => electrons += e,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && electrons % 4 == 2 {
                for (k, &i) in ring.iter().enumerate() {
                    self.atoms[i].aromatic = true;
                    let j = ring[(k + 1) % ring.len()];
                    if let Some(&(_, bi)) = self.adj[i].iter().find(|&&(n, _)| n == j) {
                        self.bonds[bi].aromatic = true;
                    }
                }
            }
        }
    }

    fn pi_electrons(&self, i: usize) -> Option<u32> {
        let a = &self.atoms[i];
        let mut ring_double = false;
        let mut other_double = false;
        for &(n, bi) in &self.adj[i] {
            if self.bonds[bi].order >= 2 && !self.bonds[bi].aromatic {
                if self.atom_in_ring(n) {
                    ring_double = true;
                } else {
                    other_double = true;
                }
            }
        }
        match a.z {
            6 if ring_double && !other_double => Some(1),
            6 if a.charge == -1 && !ring_double && !other_double => Some(2),
            7 if ring_double && !other_double => Some(1),
            7 if a.charge == 0 && !ring_double && !other_double => Some(2),
            8 | 16 | 34 if a.charge == 0 && !ring_double && !other_double => Some(2),
            _ => None,
        }
    }
}
