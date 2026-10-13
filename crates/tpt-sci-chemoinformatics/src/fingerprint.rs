//! Morgan/ECFP-style circular fingerprints and Tanimoto similarity.

use crate::mol::Molecule;

/// Number of bits in a [`Fingerprint`].
pub const FP_BITS: usize = 1024;
const WORDS: usize = FP_BITS / 64;

/// A fixed-length hashed bit-vector fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    words: [u64; WORDS],
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

impl Fingerprint {
    /// Compute a Morgan fingerprint of the given `radius` (2 ≈ ECFP4).
    pub fn morgan(mol: &Molecule, radius: usize) -> Fingerprint {
        let n = mol.atoms.len();
        let mut ids: Vec<u64> = (0..n)
            .map(|i| {
                let a = &mol.atoms[i];
                let mut h = mix(u64::from(a.z));
                h = mix(h ^ mol.heavy_degree(i) as u64);
                h = mix(h ^ u64::from(a.total_h()) << 8);
                h = mix(h ^ ((i64::from(a.charge) + 8) as u64) << 16);
                h = mix(h ^ u64::from(a.aromatic) << 24);
                mix(h ^ u64::from(mol.atom_in_ring(i)) << 32)
            })
            .collect();
        let mut fp = Fingerprint { words: [0; WORDS] };
        for &id in &ids {
            fp.set(id);
        }
        for _ in 0..radius {
            let mut next = ids.clone();
            for i in 0..n {
                let mut nb: Vec<u64> = mol.adj[i]
                    .iter()
                    .map(|&(j, b)| {
                        let bd = &mol.bonds[b];
                        let kind = if bd.aromatic { 4 } else { u64::from(bd.order) };
                        mix(ids[j] ^ kind << 56)
                    })
                    .collect();
                nb.sort_unstable();
                let mut h = mix(ids[i] ^ 0x9e37_79b9_7f4a_7c15);
                for v in nb {
                    h = mix(h ^ v);
                }
                next[i] = h;
                fp.set(h);
            }
            ids = next;
        }
        fp
    }

    fn set(&mut self, hash: u64) {
        let bit = (hash % FP_BITS as u64) as usize;
        self.words[bit / 64] |= 1 << (bit % 64);
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Tanimoto similarity in `[0, 1]`; two empty fingerprints give 0.
    pub fn tanimoto(&self, other: &Fingerprint) -> f64 {
        let (mut inter, mut union) = (0u32, 0u32);
        for (a, b) in self.words.iter().zip(&other.words) {
            inter += (a & b).count_ones();
            union += (a | b).count_ones();
        }
        if union == 0 {
            0.0
        } else {
            f64::from(inter) / f64::from(union)
        }
    }
}
