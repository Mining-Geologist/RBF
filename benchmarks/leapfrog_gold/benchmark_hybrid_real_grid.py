"""Benchmark hybrid SubDomainer against independently saved reference memberships.

Input directory must contain NPZ files with keys:
  points (N,3), tensors (N,3,3), baseline (N,), truth (N,).
The prediction never reads 'truth'. It is read solely after fit_hybrid()
returns, to compute adjusted Rand index (ARI).

Usage: python benchmark_hybrid_real_grid.py --case-dir local_case_cache \
  --out hybrid_results.csv

First six cases are development; last four are held-out (one independent
geology). The candidate Settings() were selected on development only.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from hybrid_real_grid_domainer import Settings, fit_hybrid

DEVELOPMENT = ("S2_R50", "S2_R100", "S3_R100",
               "S3_R300", "S5_R100", "S5_R300")
HOLDOUT = ("S3_R500", "S5_R50", "S5_R500", "Asymmetric_Fold")


def benchmark(directory: Path):
    results = []
    for name in (*DEVELOPMENT, *HOLDOUT):
        file = directory / (name + ".npz")
        if not file.is_file():
            raise FileNotFoundError(file)
        with np.load(file, allow_pickle=False) as dataset:
            x, mats, initial = (dataset[key].copy()
                                for key in ("points", "tensors", "baseline"))
            # 'truth' intentionally not accessed until after prediction.
            predicted, info = fit_hybrid(x, mats, initial, Settings())
            reference = dataset["truth"].copy()
        results.append(dict(
            case=name,
            split="development" if name in DEVELOPMENT else "heldout",
            baseline_ari=float(adjusted_rand_score(reference, initial)),
            hybrid_ari=float(adjusted_rand_score(reference, predicted)),
            reference_domains=int(np.unique(reference).size),
            initial_domains=info["initial_domains"],
            hybrid_domains=info["output_domains"],
            accepted_refinement=info["stage2"]["accepted"],
            merge_count=info["merges"],
            birth_count=info["births"]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("hybrid_results.csv"))
    args = p.parse_args()
    rows = benchmark(args.case_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label, data in [
        ("development", rows[:6]), ("heldout", rows[6:]),
        ("all", rows),
    ]:
        a = np.mean([r["baseline_ari"] for r in data])
        b = np.mean([r["hybrid_ari"] for r in data])
        print(f"{label}: baseline={a:.6f} hybrid={b:.6f} delta={b-a:+.6f}")


if __name__ == "__main__":
    main()
