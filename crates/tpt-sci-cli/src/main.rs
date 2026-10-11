//! The `tpt-sci` command: run a reaction-network spec and write its trajectory.
//!
//! ```text
//! tpt-sci run <spec.json>                 # CSV to stdout
//! tpt-sci run <spec.json> --out run.csv   # CSV to a file
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};

const USAGE: &str = "usage: tpt-sci run <spec.json> [--out <file.csv>]";

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("tpt-sci: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (spec_path, out_path) = parse_args(&args)?;
    let json = std::fs::read_to_string(&spec_path)
        .with_context(|| format!("reading {}", spec_path.display()))?;
    let csv = tpt_sci_cli::run_to_csv(&json)?;
    match out_path {
        Some(path) => {
            std::fs::write(&path, csv).with_context(|| format!("writing {}", path.display()))?
        }
        None => print!("{csv}"),
    }
    Ok(())
}

/// Parse `run <spec> [--out <file>]`.
fn parse_args(args: &[String]) -> Result<(PathBuf, Option<PathBuf>)> {
    match args {
        [cmd, spec] if cmd == "run" => Ok((PathBuf::from(spec), None)),
        [cmd, spec, flag, out] if cmd == "run" && flag == "--out" => {
            Ok((PathBuf::from(spec), Some(PathBuf::from(out))))
        }
        _ => bail!("{USAGE}"),
    }
}
