#!/usr/bin/env python3
"""Compare criterion benchmark means against a stored baseline.

Reads `target/criterion/**/new/estimates.json` (written by `cargo bench`) and
compares each benchmark's mean time with `benchmarks/baseline.json`.

    python scripts/bench_check.py                 # compare; exit 1 on regression
    python scripts/bench_check.py --update        # record the current run as baseline

Timings are machine-dependent. Record the baseline on the machine that will
run the comparison (for CI, the runner), and treat small regressions as noise
with the default 50 % tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_current(criterion_dir: Path) -> dict[str, float]:
    """Map `group/bench` to its mean time in nanoseconds."""
    results: dict[str, float] = {}
    for estimates in criterion_dir.glob("**/new/estimates.json"):
        bench = estimates.parent.parent.relative_to(criterion_dir).as_posix()
        data = json.loads(estimates.read_text(encoding="utf-8"))
        results[bench] = float(data["mean"]["point_estimate"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baseline.json"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="allowed slowdown as a fraction of the baseline (default 0.5 = 50%%)",
    )
    parser.add_argument("--update", action="store_true", help="write the current run as the baseline")
    args = parser.parse_args()

    current = read_current(args.criterion_dir)
    if not current:
        print(f"no criterion results under {args.criterion_dir}; run `cargo bench` first", file=sys.stderr)
        return 2

    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        ordered = {k: current[k] for k in sorted(current)}
        args.baseline.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
        print(f"recorded {len(ordered)} benchmarks in {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    regressions: list[str] = []
    missing = sorted(set(baseline) - set(current))
    for name in sorted(set(baseline) & set(current)):
        ratio = current[name] / baseline[name]
        marker = ""
        if ratio > 1.0 + args.tolerance:
            regressions.append(name)
            marker = "  <-- REGRESSION"
        print(f"{ratio:6.2f}x  {name}{marker}")
    for name in missing:
        print(f"  missing  {name} (in baseline, not in this run)")
    new = sorted(set(current) - set(baseline))
    for name in new:
        print(f"   new     {name} (not in baseline)")

    if regressions:
        print(f"\n{len(regressions)} benchmark(s) slower than {1 + args.tolerance:.2f}x baseline")
        return 1
    print("\nno regressions beyond tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
