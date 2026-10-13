use tpt_sci_chemoinformatics::{
    Dataset, Fingerprint, Molecule, Ridge, Target, TrpConfig, TrpModel, cross_validate,
    model::Metrics,
    trp::{botanical_smiles, demo_set},
};

#[test]
fn ridge_recovers_linear_relation() {
    let x: Vec<Vec<f64>> = (0..40)
        .map(|i| vec![f64::from(i), f64::from(i % 7)])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] - 2.0 * r[1] + 5.0).collect();
    let m = Ridge::fit(&x, &y, 1e-6).unwrap();
    for (r, t) in x.iter().zip(&y) {
        assert!((m.predict(r).unwrap() - t).abs() < 1e-3);
    }
}

#[test]
fn ridge_rejects_bad_input() {
    let x = vec![vec![1.0], vec![2.0]];
    assert!(Ridge::fit(&x, &[1.0, 2.0], 0.0).is_err());
    assert!(Ridge::fit(&x, &[1.0], 1.0).is_err());
    let m = Ridge::fit(&x, &[1.0, 2.0], 1.0).unwrap();
    assert!(m.predict(&[1.0, 2.0]).is_err());
}

#[test]
fn metrics_perfect_and_mean_predictor() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let perfect = Metrics::compute(&y, &y).unwrap();
    assert!((perfect.r2 - 1.0).abs() < 1e-12);
    assert_eq!(perfect.rmse, 0.0);
    let mean = [2.5; 4];
    assert!((Metrics::compute(&y, &mean).unwrap().r2).abs() < 1e-12);
}

#[test]
fn tanimoto_identity_and_range() {
    let a = Fingerprint::morgan(&Molecule::from_smiles("c1ccccc1O").unwrap(), 2);
    let b = Fingerprint::morgan(&Molecule::from_smiles("CCCCCCCC").unwrap(), 2);
    assert!((a.tanimoto(&a) - 1.0).abs() < 1e-12);
    let s = a.tanimoto(&b);
    assert!((0.0..1.0).contains(&s));
}

#[test]
fn csv_loader_parses_and_reports_errors() {
    let ok = Dataset::from_csv("name,smiles,target\n# comment\nwater,O,1.5\n\nethanol,CCO,2.0\n")
        .unwrap();
    assert_eq!(ok.len(), 2);
    assert_eq!(ok.records[0].name.as_deref(), Some("water"));
    assert!(Dataset::from_csv("smiles,activity\nO,1").is_err());
    assert!(Dataset::from_csv("smiles,target\nO,abc").is_err());
    assert!(Dataset::from_csv("smiles,target\nO,inf").is_err());
    assert!(Dataset::from_csv("smiles,target\n").is_err());
    assert!(Dataset::from_csv("smiles,target\nO").is_err());
}

#[test]
fn cross_validation_is_deterministic_for_seed() {
    let data = demo_set(Target::Trpv1);
    let cfg = TrpConfig::default();
    let a = TrpModel::cross_validate(&data, &cfg, 2, 7).unwrap();
    let b = TrpModel::cross_validate(&data, &cfg, 2, 7).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.n, data.len());
}

#[test]
fn leave_one_out_covers_every_sample_once() {
    let y: Vec<f64> = (0..6).map(f64::from).collect();
    let mut seen = Vec::new();
    let m = cross_validate(&y, 0, 1, |train, test| {
        assert_eq!(train.len(), 5);
        seen.extend_from_slice(test);
        Ok(test.iter().map(|&i| y[i]).collect())
    })
    .unwrap();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    assert!((m.r2 - 1.0).abs() < 1e-12);
}

#[test]
fn trp_model_predicts_demo_and_flags_domain() {
    let cfg = TrpConfig::default();
    for target in [Target::Trpa1, Target::Trpv1] {
        let model = TrpModel::fit(target, &demo_set(target), &cfg).unwrap();
        assert_eq!(model.target(), target);
        // A training compound is in-domain with similarity 1.
        let train = &demo_set(target).records[0].smiles;
        let p = model.predict_smiles(train).unwrap();
        assert!((p.max_similarity - 1.0).abs() < 1e-12);
        assert!(p.in_domain);
        for (_, smi) in botanical_smiles() {
            let p = model.predict_smiles(smi).unwrap();
            assert!(p.consensus.is_finite());
            assert!((0.0..=1.0).contains(&p.max_similarity));
        }
    }
}

#[test]
fn no_pka_term_means_ionisation_state_alone_does_not_change_features() {
    // The descriptor set is structural only: no pH/pKa feature exists, so
    // ionisation enters solely through formal charge.
    let names = tpt_sci_chemoinformatics::DESCRIPTOR_NAMES;
    assert!(
        names
            .iter()
            .all(|n| *n != "pka" && *n != "ph" && *n != "pka_ph")
    );
}
