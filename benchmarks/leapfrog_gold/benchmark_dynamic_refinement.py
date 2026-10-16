"""Measure an offline domain-refinement candidate on owner-supplied NPZ files.

No Leapfrog executable is used. Predictions never see the 'reference' labels.
Each file must contain: points (N,3), tensors (N,3,3), initial_labels (N,),
reference_labels (N,). Labels are consulted ONLY after predictions finish.

Usage:
 python benchmark_dynamic_refinement.py --inputs "/path/to/cases/*.npz" \
   --out dynamic_domain_results.csv

Development cases: S2_R50, S2_R100, S3_R100, S3_R300, S5_R100, S5_R300.
All other case names are marked holdout. Do not tune on holdouts.
"""
from __future__ import annotations

import argparse
import csv
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from dynamic_domain_refinement import refine_domains

DEVELOPMENT = {"S2_R50", "S2_R100", "S3_R100",
               "S3_R300", "S5_R100", "S5_R300"}


def benchmark(paths):
    results = []
    for path in sorted(paths):
        source = Path(path)
        with np.load(source, allow_pickle=False) as obj:
            points = obj["points"].copy()
            tensors = obj["tensors"].copy()
            initial = obj["initial_labels"].copy()
            reference = obj["reference_labels"].copy()

        # Reference labels are not passed into the predictor.
        prediction, diagnostics = refine_domains(points, tensors, initial)
        results.append({
            "case": source.stem,
            "group": "development" if source.stem in DEVELOPMENT else "holdout",
            "baseline_ari": float(adjusted_rand_score(reference, initial)),
            "refined_ari": float(adjusted_rand_score(reference, prediction)),
            "initial_domains": int(len(np.unique(initial))),
            "refined_domains": int(len(np.unique(prediction))),
            **diagnostics,
        })
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, help="Glob for locally held NPZ cases")
    p.add_argument("--out", type=Path, default=Path("dynamic_domain_results.csv"))
    args = p.parse_args()
    paths = glob(args.inputs)
    if not paths:
        p.error("No local input files match the supplied glob")
    rows = benchmark(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group in ("development", "holdout"):
        selected = [row for row in rows if row["group"] == group]
        if not selected:
            continue
        base = np.mean([row["baseline_ari"] for row in selected])
        refined = np.mean([row["refined_ari"] for row in selected])
        print(f"{group}: baseline={base:.6f} candidate={refined:.6f} delta={refined-base:+.6f}")


if __name__ == "__main__":
    main()
