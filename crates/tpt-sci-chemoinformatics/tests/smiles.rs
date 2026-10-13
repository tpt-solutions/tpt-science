use tpt_sci_chemoinformatics::{ChemoinformaticsError, DescriptorSet, Molecule};

fn mol(s: &str) -> Molecule {
    Molecule::from_smiles(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

#[test]
fn water_and_methane_formulae() {
    assert_eq!(mol("O").formula(), "H2O");
    assert_eq!(mol("C").formula(), "CH4");
}

#[test]
fn benzene_is_aromatic_six_ring() {
    let m = mol("c1ccccc1");
    assert_eq!(m.formula(), "C6H6");
    assert_eq!(m.rings.len(), 1);
    assert_eq!(m.rings[0].len(), 6);
    assert!(m.atoms.iter().all(|a| a.aromatic));
}

#[test]
fn kekule_benzene_is_detected_aromatic() {
    let m = mol("C1=CC=CC=C1");
    assert!(m.atoms.iter().all(|a| a.aromatic));
    assert_eq!(m.formula(), "C6H6");
}

#[test]
fn caffeine_formula_and_mw() {
    let m = mol("Cn1cnc2c1c(=O)n(C)c(=O)n2C");
    assert_eq!(m.formula(), "C8H10N4O2");
    assert!((m.molecular_weight() - 194.19).abs() < 0.05);
    assert_eq!(m.rings.len(), 2);
}

#[test]
fn capsaicin_formula_and_mw() {
    let m = mol("COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O");
    assert_eq!(m.formula(), "C18H27NO3");
    assert!((m.molecular_weight() - 305.41).abs() < 0.05);
}

#[test]
fn sanshool_formula_and_polyene_count() {
    let m = mol("C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O");
    assert_eq!(m.formula(), "C16H25NO2");
    let d = DescriptorSet::compute(&m);
    assert_eq!(d.get("cc_double_bonds"), Some(4.0));
    assert_eq!(d.get("amides"), Some(1.0));
}

#[test]
fn egcg_has_phenolic_hydroxyls_and_four_rings() {
    let m = mol("Oc1cc(O)c2C[C@@H](OC(=O)c3cc(O)c(O)c(O)c3)[C@H](Oc2c1)c1cc(O)c(O)c(O)c1");
    assert_eq!(m.formula(), "C22H18O11");
    let d = DescriptorSet::compute(&m);
    assert_eq!(d.get("phenol_oh"), Some(8.0));
    assert_eq!(d.get("rings"), Some(4.0)); // A, C (pyran), B and galloyl rings
}

#[test]
fn disconnected_components_are_kept() {
    let m = mol("[Na+].[Cl-]");
    assert_eq!(m.num_atoms(), 2);
    assert!(m.bonds.is_empty());
    assert_eq!(m.atoms[0].charge, 1);
    assert_eq!(m.atoms[1].charge, -1);
}

#[test]
fn bracket_hydrogen_count_and_isotope() {
    let m = mol("[13CH4]");
    assert_eq!(m.atoms[0].isotope, Some(13));
    assert_eq!(m.atoms[0].total_h(), 4);
}

#[test]
fn ring_closure_with_two_digit_label() {
    let m = mol("C%10CCCCC%10");
    assert_eq!(m.rings.len(), 1);
    assert_eq!(m.formula(), "C6H12");
}

#[test]
fn malformed_smiles_are_rejected() {
    for bad in ["C(C", "C1CC", "C)", "C==C", "", "=C", "C1CC1)", "C(=O"] {
        assert!(
            matches!(
                Molecule::from_smiles(bad),
                Err(ChemoinformaticsError::Smiles { .. })
            ),
            "expected Smiles error for {bad:?}"
        );
    }
}

#[test]
fn over_valent_organic_atom_is_rejected() {
    assert!(matches!(
        Molecule::from_smiles("C(C)(C)(C)(C)C"),
        Err(ChemoinformaticsError::Valence { .. })
    ));
}

#[test]
fn unknown_element_is_rejected() {
    assert!(matches!(
        Molecule::from_smiles("[Xx]"),
        Err(ChemoinformaticsError::UnknownElement(_))
    ));
    assert!(matches!(
        Molecule::from_smiles("Q"),
        Err(ChemoinformaticsError::UnknownElement(_))
    ));
}

#[test]
fn unsupported_syntax_is_reported_not_guessed() {
    assert!(matches!(
        Molecule::from_smiles("*C"),
        Err(ChemoinformaticsError::UnsupportedSyntax(_))
    ));
}

#[test]
fn ring_and_rotatable_counts() {
    let benzene = DescriptorSet::compute(&mol("c1ccccc1"));
    assert_eq!(benzene.get("rotatable_bonds"), Some(0.0));
    assert_eq!(benzene.get("aromatic_rings"), Some(1.0));
    // Butane has one internal rotatable bond.
    assert_eq!(
        DescriptorSet::compute(&mol("CCCC")).get("rotatable_bonds"),
        Some(1.0)
    );
}

#[test]
fn hbd_hba_counts() {
    let d = DescriptorSet::compute(&mol("OCC(N)=O"));
    assert_eq!(d.get("hbd"), Some(2.0)); // OH and NH2
    assert_eq!(d.get("hba"), Some(3.0)); // hydroxyl O, amide N, carbonyl O
}

#[test]
fn logp_orders_polar_below_apolar() {
    let hexane = DescriptorSet::compute(&mol("CCCCCC")).get("logp").unwrap();
    let glycerol = DescriptorSet::compute(&mol("OCC(O)CO"))
        .get("logp")
        .unwrap();
    assert!(
        hexane > glycerol + 2.0,
        "hexane {hexane} vs glycerol {glycerol}"
    );
}

#[test]
fn tpsa_zero_for_hydrocarbon_and_positive_for_polar() {
    assert_eq!(
        DescriptorSet::compute(&mol("CCCCCC")).get("tpsa"),
        Some(0.0)
    );
    assert!(
        DescriptorSet::compute(&mol("OCC(O)CO"))
            .get("tpsa")
            .unwrap()
            > 40.0
    );
}
