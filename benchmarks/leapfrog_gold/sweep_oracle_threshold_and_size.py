"""Sweep consistency threshold and maximum domain fraction against Leapfrog labels.

This diagnostic keeps the recovered structural field, centroid grid, adjacency, merge
matrix, heap implementation, and point assignment fixed.  It changes only the
consistency threshold and maximum centroid-domain fraction for the unmodified
production reciprocal-determinant control.

The purpose is diagnostic, not parameter fitting.  If some parameter pair approaches
the decoded Leapfrog labels, the remaining mismatch is probably score scaling or size
semantics.  If no pair improves materially, the likely error is earlier in mini-cluster
construction, adjacency, stale-heap/tie handling, or point-to-grid assignment.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing this wrapper patches the oracle CSV loader used by comparison.
import compare_automatic_domains_to_oracle_robust as robust  # noqa: E402,F401

comparison = robust.comparison


def _parse_float_list(text: str, *, name: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} contains a non-numeric value: {token!r}"
            ) from exc
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root",
        type=Path,
        default=Path("Leapfrog_LVA_decoded_benchmark"),
    )
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        default="0.15,0.30,0.45,0.60,0.75,0.90",
        help="Comma-separated consistency thresholds.",
    )
    parser.add_argument(
        "--maximum-fractions",
        default="0.10,0.15,0.20,0.30",
        help="Comma-separated maximum centroid-domain fractions.",
    )
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/oracle-threshold-size-sweep.json"),
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    if not case_name:
        parser.error("--case must not be empty")
    if not args.decoded_root.is_dir():
        parser.error(f"Decoded benchmark root was not found: {args.decoded_root}")
    if args.centroid_count <= 0:
        parser.error("--centroid-count must be positive")
    if not 0.0 < args.minimum_fraction <= 1.0:
        parser.error("--minimum-fraction must be in (0, 1]")
    if args.top <= 0:
        parser.error("--top must be positive")

    try:
        thresholds = _parse_float_list(args.thresholds, name="--thresholds")
        maximum_fractions = _parse_float_list(
            args.maximum_fractions,
            name="--maximum-fractions",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if any(not 0.0 < value <= 1.0 for value in thresholds):
        parser.error("every threshold must be in (0, 1]")
    if any(
        not args.minimum_fraction <= value <= 1.0
        for value in maximum_fractions
    ):
        parser.error(
            "every maximum fraction must satisfy minimum_fraction <= value <= 1"
        )

    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory  # noqa: E402
    import run_selected_exact_leapfrog_lva as exact  # noqa: E402

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(  # noqa: SLF001
        args.decoded_root,
        case_name,
    )
    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = comparison._parse_case_parameters(case_name)  # noqa: SLF001
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        strength,
        trend_range,
    )

    builder_class = polatory.AutomaticStructuralDomainBuilder3
    builder_module = importlib.import_module(builder_class.__module__)
    original_merge_function = getattr(
        builder_module,
        "_merged_matrix_and_consistency",
        None,
    )
    determinant_function = getattr(builder_module, "_symmetric_determinant", None)
    if original_merge_function is None or determinant_function is None:
        raise RuntimeError(
            f"{builder_class.__module__} does not expose the recovered merge helpers."
        )

    preparation_builder = builder_class(
        centroid_count=args.centroid_count,
        minimum_cluster_fraction=args.minimum_fraction,
        maximum_cluster_fraction=max(maximum_fractions),
        consistency_threshold=min(thresholds),
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, _, shape, centroids = preparation_builder._prepare_grid(  # noqa: SLF001
        points
    )
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids,
        trend_input,
        non_decaying=False,
    )

    print(
        f"case={case_name} points={len(points)} oracle_domains="
        f"{len(np.unique(oracle_labels))} centroids={len(centroids)} grid={shape}",
        flush=True,
    )
    print(
        f"sweeping {len(thresholds)} thresholds x "
        f"{len(maximum_fractions)} maximum fractions = "
        f"{len(thresholds) * len(maximum_fractions)} runs",
        flush=True,
    )

    results = []
    try:
        for maximum_fraction in maximum_fractions:
            for threshold in thresholds:
                result = comparison._compare_formula(  # noqa: SLF001
                    case_name=case_name,
                    formula_name="reciprocal_det_control",
                    transform=None,
                    builder_class=builder_class,
                    builder_module=builder_module,
                    original_merge_function=original_merge_function,
                    determinant_function=determinant_function,
                    points=np.asarray(points, dtype=np.float64),
                    oracle_labels=np.asarray(oracle_labels, dtype=np.int64),
                    centroid_anisotropies=np.asarray(
                        centroid_anisotropies,
                        dtype=np.float64,
                    ),
                    minimum=np.asarray(minimum, dtype=np.float64),
                    maximum=np.asarray(maximum, dtype=np.float64),
                    shape=tuple(int(value) for value in shape),
                    threshold=float(threshold),
                    centroid_count=args.centroid_count,
                    minimum_fraction=args.minimum_fraction,
                    maximum_fraction=float(maximum_fraction),
                )
                row = asdict(result)
                row["maximum_fraction"] = float(maximum_fraction)
                results.append(row)
                print(
                    f"threshold={threshold:>5.2f} max_fraction={maximum_fraction:>5.2f} "
                    f"pred={result.predicted_domains:>3d} "
                    f"ARI={result.adjusted_rand_index:>8.5f} "
                    f"match={result.optimal_label_accuracy:>8.5f}",
                    flush=True,
                )
    finally:
        setattr(
            builder_module,
            "_merged_matrix_and_consistency",
            original_merge_function,
        )

    results.sort(
        key=lambda item: (
            -item["adjusted_rand_index"],
            -item["optimal_label_accuracy"],
            abs(item["predicted_domains"] - item["oracle_domains"]),
            item["maximum_fraction"],
            item["threshold"],
        )
    )

    print("\nBest parameter pairs:", flush=True)
    print(
        f"{'threshold':>9} {'max_frac':>9} {'oracle':>7} {'pred':>7} "
        f"{'ARI':>9} {'AMI':>9} {'match_acc':>10} {'merges':>8}",
        flush=True,
    )
    for item in results[: min(args.top, len(results))]:
        print(
            f"{item['threshold']:>9.2f} {item['maximum_fraction']:>9.2f} "
            f"{item['oracle_domains']:>7d} {item['predicted_domains']:>7d} "
            f"{item['adjusted_rand_index']:>9.5f} "
            f"{item['adjusted_mutual_information']:>9.5f} "
            f"{item['optimal_label_accuracy']:>10.5f} "
            f"{item['merge_count']:>8d}",
            flush=True,
        )

    payload = {
        "case": case_name,
        "point_count": int(len(points)),
        "oracle_domains": int(len(np.unique(oracle_labels))),
        "centroid_count_requested": int(args.centroid_count),
        "centroid_count_actual": int(len(centroids)),
        "centroid_grid_shape": [int(value) for value in shape],
        "minimum_fraction": float(args.minimum_fraction),
        "thresholds": thresholds,
        "maximum_fractions": maximum_fractions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
