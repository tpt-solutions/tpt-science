//! From-scratch SMILES parser.
//!
//! Supported: the organic subset (`B C N O P S F Cl Br I`), aromatic
//! lowercase atoms (`b c n o p s`, `se`, `as` in brackets), bracket atoms with
//! isotope, chirality (`@`/`@@`), hydrogen count, charge and atom class,
//! branches, ring closures (`1`-`9`, `%nn`), bond symbols `- = # : / \`, and
//! disconnected components (`.`).
//!
//! Not supported (returns [`ChemoinformaticsError::UnsupportedSyntax`]):
//! wildcard `*`, quadruple bond `$`, extended chirality (`@TH1`, `@SP1`, ...),
//! reaction arrows and SMARTS.

use crate::elements;
use crate::error::{ChemoinformaticsError, Result};
use crate::mol::{Atom, Bond, BondDir, Chirality, Molecule};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Spec {
    Order(u8),
    Aromatic,
    Dir(BondDir),
}

fn err(pos: usize, msg: &str) -> ChemoinformaticsError {
    ChemoinformaticsError::Smiles {
        pos,
        msg: msg.to_string(),
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    prev: Option<usize>,
    pending: Option<Spec>,
    stack: Vec<Option<usize>>,
    rings: HashMap<u32, (usize, Option<Spec>, usize)>,
}

/// Parse a SMILES string into a [`Molecule`].
///
/// # Errors
/// Returns [`ChemoinformaticsError::Smiles`] for malformed input,
/// [`ChemoinformaticsError::UnknownElement`] for unsupported elements,
/// [`ChemoinformaticsError::Valence`] for over-valent organic-subset atoms and
/// [`ChemoinformaticsError::UnsupportedSyntax`] for valid but unsupported syntax.
pub fn parse(smiles: &str) -> Result<Molecule> {
    if !smiles.is_ascii() {
        return Err(err(0, "non-ASCII character"));
    }
    let trimmed = smiles.trim();
    if trimmed.is_empty() {
        return Err(err(0, "empty SMILES"));
    }
    let mut p = Parser {
        s: trimmed.as_bytes(),
        i: 0,
        atoms: Vec::new(),
        bonds: Vec::new(),
        prev: None,
        pending: None,
        stack: Vec::new(),
        rings: HashMap::new(),
    };
    p.run()?;
    Molecule::from_parts(p.atoms, p.bonds)
}

impl Parser<'_> {
    fn run(&mut self) -> Result<()> {
        while self.i < self.s.len() {
            let c = self.s[self.i];
            match c {
                b'(' => {
                    if self.prev.is_none() {
                        return Err(err(self.i, "branch with no preceding atom"));
                    }
                    self.stack.push(self.prev);
                    self.i += 1;
                }
                b')' => {
                    if self.pending.is_some() {
                        return Err(err(self.i, "bond symbol before ')'"));
                    }
                    self.prev = self
                        .stack
                        .pop()
                        .ok_or_else(|| err(self.i, "unmatched ')'"))?;
                    self.i += 1;
                }
                b'.' => {
                    if self.pending.is_some() {
                        return Err(err(self.i, "bond symbol before '.'"));
                    }
                    self.prev = None;
                    self.i += 1;
                }
                b'-' | b'=' | b'#' | b':' | b'/' | b'\\' => {
                    if self.pending.is_some() {
                        return Err(err(self.i, "two consecutive bond symbols"));
                    }
                    self.pending = Some(match c {
                        b'-' => Spec::Order(1),
                        b'=' => Spec::Order(2),
                        b'#' => Spec::Order(3),
                        b':' => Spec::Aromatic,
                        b'/' => Spec::Dir(BondDir::Up),
                        _ => Spec::Dir(BondDir::Down),
                    });
                    self.i += 1;
                }
                b'$' => {
                    return Err(ChemoinformaticsError::UnsupportedSyntax(
                        "quadruple bond '$'".into(),
                    ));
                }
                b'*' => {
                    return Err(ChemoinformaticsError::UnsupportedSyntax(
                        "wildcard atom '*'".into(),
                    ));
                }
                b'0'..=b'9' => {
                    let n = u32::from(c - b'0');
                    self.i += 1;
                    self.ring_closure(n)?;
                }
                b'%' => {
                    let start = self.i;
                    let d = self
                        .s
                        .get(self.i + 1..self.i + 3)
                        .filter(|d| d.iter().all(u8::is_ascii_digit));
                    let d = d.ok_or_else(|| err(start, "'%' must be followed by two digits"))?;
                    let n = u32::from(d[0] - b'0') * 10 + u32::from(d[1] - b'0');
                    self.i += 3;
                    self.ring_closure(n)?;
                }
                b'[' => self.bracket_atom()?,
                _ => self.organic_atom()?,
            }
        }
        if !self.stack.is_empty() {
            return Err(err(self.s.len(), "unclosed branch '('"));
        }
        if let Some((_, _, pos)) = self.rings.values().next() {
            return Err(err(*pos, "unclosed ring bond"));
        }
        if self.pending.is_some() {
            return Err(err(self.s.len(), "dangling bond symbol"));
        }
        Ok(())
    }

    fn organic_atom(&mut self) -> Result<()> {
        let pos = self.i;
        let c = self.s[self.i];
        let two = self.s.get(self.i + 1).copied();
        let (symbol, len, lower) = match (c, two) {
            (b'C', Some(b'l')) => ("Cl", 2, false),
            (b'B', Some(b'r')) => ("Br", 2, false),
            (b'B', _) => ("B", 1, false),
            (b'C', _) => ("C", 1, false),
            (b'N', _) => ("N", 1, false),
            (b'O', _) => ("O", 1, false),
            (b'P', _) => ("P", 1, false),
            (b'S', _) => ("S", 1, false),
            (b'F', _) => ("F", 1, false),
            (b'I', _) => ("I", 1, false),
            (b'b', _) => ("B", 1, true),
            (b'c', _) => ("C", 1, true),
            (b'n', _) => ("N", 1, true),
            (b'o', _) => ("O", 1, true),
            (b'p', _) => ("P", 1, true),
            (b's', _) => ("S", 1, true),
            _ => {
                let ch = c as char;
                return if ch.is_ascii_alphabetic() {
                    Err(ChemoinformaticsError::UnknownElement(format!(
                        "{ch} (outside the organic subset; use brackets) at position {pos}"
                    )))
                } else {
                    Err(err(pos, &format!("unexpected character '{ch}'")))
                };
            }
        };
        self.i += len;
        let mut atom = Atom::new(elements::by_symbol(symbol).expect("organic subset").z);
        atom.lowercase = lower;
        atom.aromatic = lower;
        self.push_atom(atom)
    }

    fn bracket_atom(&mut self) -> Result<()> {
        let open = self.i;
        self.i += 1;
        let isotope = self.number()?;
        let isotope = isotope.map(|v| v as u16);

        // Element symbol (or aromatic lowercase symbol).
        let c = *self
            .s
            .get(self.i)
            .ok_or_else(|| err(open, "unterminated bracket atom"))?;
        let (z, lowercase) = if c.is_ascii_uppercase() {
            let mut sym = String::from(c as char);
            if let Some(&n) = self.s.get(self.i + 1) {
                if n.is_ascii_lowercase() {
                    let two = format!("{}{}", c as char, n as char);
                    if elements::by_symbol(&two).is_some() {
                        sym = two;
                    }
                }
            }
            self.i += sym.len();
            let el = elements::by_symbol(&sym)
                .ok_or_else(|| ChemoinformaticsError::UnknownElement(sym.clone()))?;
            (el.z, false)
        } else if c.is_ascii_lowercase() {
            let two = self
                .s
                .get(self.i..self.i + 2)
                .map(|b| String::from_utf8_lossy(b).to_string());
            let (sym, len) = match two.as_deref() {
                Some("se") => ("Se", 2),
                Some("as") => {
                    return Err(ChemoinformaticsError::UnknownElement("As".into()));
                }
                _ => match c {
                    b'b' => ("B", 1),
                    b'c' => ("C", 1),
                    b'n' => ("N", 1),
                    b'o' => ("O", 1),
                    b'p' => ("P", 1),
                    b's' => ("S", 1),
                    _ => {
                        return Err(ChemoinformaticsError::UnknownElement(
                            (c as char).to_string(),
                        ));
                    }
                },
            };
            self.i += len;
            (elements::by_symbol(sym).expect("table").z, true)
        } else {
            return Err(err(self.i, "expected element symbol in bracket atom"));
        };
        let mut atom = Atom::new(z);
        atom.bracket = true;
        atom.lowercase = lowercase;
        atom.aromatic = lowercase;
        atom.isotope = isotope;

        // Chirality.
        if self.peek() == Some(b'@') {
            self.i += 1;
            if self.peek() == Some(b'@') {
                self.i += 1;
                atom.chirality = Some(Chirality::Cw);
            } else {
                atom.chirality = Some(Chirality::Ccw);
            }
            if matches!(self.peek(), Some(b'T' | b'A' | b'S' | b'O')) {
                return Err(ChemoinformaticsError::UnsupportedSyntax(
                    "extended chirality classes (@TH, @SP, ...)".into(),
                ));
            }
        }
        // Hydrogen count.
        let mut h = 0u8;
        if self.peek() == Some(b'H') && z != 1 {
            self.i += 1;
            h = self.number()?.map_or(1, |n| n as u8);
        }
        atom.explicit_h = Some(h);
        // Charge.
        if let Some(sign @ (b'+' | b'-')) = self.peek() {
            self.i += 1;
            let mut mag: i32 = 1;
            if let Some(n) = self.number()? {
                mag = n as i32;
            } else {
                while self.peek() == Some(sign) {
                    self.i += 1;
                    mag += 1;
                }
            }
            let signed = if sign == b'+' { mag } else { -mag };
            atom.charge = i8::try_from(signed).map_err(|_| err(self.i, "charge too large"))?;
        }
        // Atom class.
        if self.peek() == Some(b':') {
            self.i += 1;
            atom.class = Some(
                self.number()?
                    .ok_or_else(|| err(self.i, "expected atom class number"))?,
            );
        }
        if self.peek() != Some(b']') {
            return Err(err(self.i, "expected ']' to close bracket atom"));
        }
        self.i += 1;
        self.push_atom(atom)
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn number(&mut self) -> Result<Option<u32>> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        if start == self.i {
            return Ok(None);
        }
        std::str::from_utf8(&self.s[start..self.i])
            .ok()
            .and_then(|t| t.parse::<u32>().ok())
            .map(Some)
            .ok_or_else(|| err(start, "number out of range"))
    }

    fn push_atom(&mut self, atom: Atom) -> Result<()> {
        let idx = self.atoms.len();
        let aromatic = atom.aromatic;
        self.atoms.push(atom);
        if let Some(p) = self.prev {
            let spec = self.pending.take();
            let both_aromatic = aromatic && self.atoms[p].aromatic;
            self.add_bond(p, idx, spec, both_aromatic);
        } else if self.pending.is_some() {
            return Err(err(self.i, "bond symbol with no preceding atom"));
        }
        self.prev = Some(idx);
        Ok(())
    }

    fn add_bond(&mut self, a: usize, b: usize, spec: Option<Spec>, both_aromatic: bool) {
        let (order, aromatic, dir, implicit) = match spec {
            Some(Spec::Order(o)) => (o, false, None, false),
            Some(Spec::Aromatic) => (1, true, None, false),
            Some(Spec::Dir(d)) => (1, false, Some(d), false),
            None => (1, both_aromatic, None, true),
        };
        self.bonds.push(Bond {
            a,
            b,
            order,
            aromatic,
            dir,
            implicit,
        });
    }

    fn ring_closure(&mut self, n: u32) -> Result<()> {
        let at = self.i;
        let cur = self
            .prev
            .ok_or_else(|| err(at, "ring closure with no preceding atom"))?;
        let spec = self.pending.take();
        match self.rings.remove(&n) {
            None => {
                self.rings.insert(n, (cur, spec, at));
            }
            Some((other, open_spec, _)) => {
                if other == cur {
                    return Err(err(at, "ring closure bonds an atom to itself"));
                }
                if self
                    .bonds
                    .iter()
                    .any(|b| (b.a == other && b.b == cur) || (b.a == cur && b.b == other))
                {
                    return Err(err(at, "duplicate bond between atoms"));
                }
                let merged = match (open_spec, spec) {
                    (Some(a), Some(b))
                        if a != b && !matches!((a, b), (Spec::Dir(_), Spec::Dir(_))) =>
                    {
                        return Err(err(at, "conflicting ring-closure bond symbols"));
                    }
                    (a, b) => a.or(b),
                };
                let both = self.atoms[other].aromatic && self.atoms[cur].aromatic;
                self.add_bond(other, cur, merged, both);
            }
        }
        Ok(())
    }
}
