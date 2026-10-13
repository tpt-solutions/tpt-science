use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_sci_chemoinformatics::{DescriptorSet, Fingerprint, Molecule, Ridge};

const SANSHOOL: &str = "C/C=C/C=C/C=C\\CC/C=C/C(=O)NCC(C)(C)O";
const EGCG: &str = "Oc1cc(O)c2C[C@@H](OC(=O)c3cc(O)c(O)c(O)c3)[C@H](Oc2c1)c1cc(O)c(O)c(O)c1";

fn bench_parse(c: &mut Criterion) {
    c.bench_function("smiles_parse_egcg", |b| {
        b.iter(|| Molecule::from_smiles(black_box(EGCG)).unwrap());
    });
}

fn bench_descriptors(c: &mut Criterion) {
    let mol = Molecule::from_smiles(SANSHOOL).unwrap();
    c.bench_function("descriptors_sanshool", |b| {
        b.iter(|| DescriptorSet::compute(black_box(&mol)));
    });
}

fn bench_fingerprint(c: &mut Criterion) {
    let mol = Molecule::from_smiles(EGCG).unwrap();
    c.bench_function("morgan_r2_egcg", |b| {
        b.iter(|| Fingerprint::morgan(black_box(&mol), 2));
    });
}

fn bench_ridge(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..200)
        .map(|i| {
            (0..18)
                .map(|j| ((i * 31 + j * 7) % 97) as f64 / 97.0)
                .collect()
        })
        .collect();
    let y: Vec<f64> = x.iter().map(|r| r[0] * 2.0 - r[3] + 0.5).collect();
    c.bench_function("ridge_fit_200x18", |b| {
        b.iter(|| Ridge::fit(black_box(&x), black_box(&y), 1.0).unwrap());
    });
}

criterion_group!(
    benches,
    bench_parse,
    bench_descriptors,
    bench_fingerprint,
    bench_ridge
);
criterion_main!(benches);
