//! Fit TRPA1 and TRPV1 QSPR models on the illustrative demo set, report
//! cross-validated metrics, and screen a few botanical compounds.
//!
//! The demo labels are approximate and illustrative only; supply a curated
//! dataset with `Dataset::from_csv` for real work.

use tpt_sci_chemoinformatics::trp::{self, Target, TrpConfig, TrpModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "WARNING: the demo set has 4 compounds per target. LOO metrics below are \
         not a validation, and predictions for compounds with in_domain=false are \
         extrapolations. Use a curated dataset (Dataset::from_csv) for real work.\n"
    );
    let config = TrpConfig::default();
    for target in [Target::Trpv1, Target::Trpa1] {
        let data = trp::demo_set(target);
        // Four compounds per target: leave-one-out is the only sensible CV here.
        let cv = TrpModel::cross_validate(&data, &config, 0, 42)?;
        println!(
            "{target:?}: n={} LOO R2={:.3} RMSE={:.3} MAE={:.3}",
            cv.n, cv.r2, cv.rmse, cv.mae
        );
        let model = TrpModel::fit(target, &data, &config)?;
        for (name, smiles) in trp::botanical_smiles() {
            let p = model.predict_smiles(smiles)?;
            println!(
                "  {name:<24} consensus={:+.2} sim={:.2} in_domain={}",
                p.consensus, p.max_similarity, p.in_domain
            );
        }
    }
    Ok(())
}
