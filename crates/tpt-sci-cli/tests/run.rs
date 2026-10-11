//! End-to-end checks for the declarative runner.

use tpt_sci_cli::{Spec, parse_method, run, run_to_csv};

const SIR: &str = r#"{
  "network": "beta, S + I --> 2 I\ngamma, I --> R",
  "parameters": { "beta": 0.0004, "gamma": 0.2 },
  "initial": { "S": 990, "I": 10, "R": 0 },
  "t_final": 60,
  "points": 60
}"#;

#[test]
fn sir_csv_has_header_rows_and_conserves_population() {
    let csv = run_to_csv(SIR).unwrap();
    let mut lines = csv.lines();
    assert_eq!(lines.next().unwrap(), "t,S,I,R");
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse::<f64>().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), 60);
    for row in &rows {
        let total = row[1] + row[2] + row[3];
        assert!(
            (total - 1000.0).abs() < 1e-3,
            "population {total} at t = {}",
            row[0]
        );
    }
    let last = rows.last().unwrap();
    assert!((last[0] - 60.0).abs() < 1e-12);
    assert!(
        last[2] < 10.0,
        "epidemic should have burned out, I = {}",
        last[2]
    );
}

#[test]
fn spec_defaults_apply_when_fields_are_omitted() {
    let spec: Spec = serde_json::from_str(
        r#"{ "network": "k, A --> B", "parameters": {"k": 1.0},
             "initial": {"A": 1.0}, "t_final": 1.0 }"#,
    )
    .unwrap();
    assert_eq!(spec.points, 100);
    assert_eq!(spec.method, "bdf");
    let traj = run(&spec).unwrap();
    assert_eq!(traj.len(), 100);
    let a_end = traj.column("A").unwrap().last().copied().unwrap();
    assert!((a_end - (-1.0f64).exp()).abs() < 1e-5, "A(1) = {a_end}");
}

#[test]
fn unknown_fields_are_rejected() {
    let bad = SIR.replace("\"points\": 60", "\"point\": 60");
    assert!(run_to_csv(&bad).is_err());
}

#[test]
fn bad_inputs_produce_clear_errors() {
    assert!(parse_method("rk4").is_err());
    let backwards = SIR.replace("\"t_final\": 60", "\"t_final\": -1");
    let err = run_to_csv(&backwards).unwrap_err().to_string();
    assert!(err.contains("t_final"), "{err}");
    let undeclared = SIR.replace("\"gamma\": 0.2", "\"delta\": 0.2");
    assert!(run_to_csv(&undeclared).is_err());
}
