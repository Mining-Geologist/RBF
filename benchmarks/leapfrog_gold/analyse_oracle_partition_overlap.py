"""Analyse how corrected automatic domains overlap decoded Leapfrog domains.

This diagnostic determines whether the missing Leapfrog real-location SubDomainer stage
could be represented as a simple split or merge of the current grid-derived point labels.
It reports the full oracle/predicted contingency matrix, per-domain purity, and pairwise
false-split/false-merge counts. No RBF fitting or surface meshing is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing this module patches the strict coordinate/label loader in comparison.
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402


def _choose2(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    return int(np.sum(values * (values - 1) // 2))


def _contingency(
    oracle: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oracle_values, oracle_inverse = np.unique(oracle, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    matrix = np.zeros((len(oracle_values), len(predicted_values)), dtype=np.int64)
    np.add.at(matrix, (oracle_inverse, predicted_inverse), 1)
    return oracle_values, predicted_values, matrix


def _print_matrix(
    oracle_values: np.ndarray,
    predicted_values: np.ndarray,
    matrix: np.ndarray,
) -> None:
    width = max(6, max(len(str(int(value))) for value in predicted_values) + 2)
    header = "oracle\\pred".rjust(12) + "".join(
        f"P{int(value)}".rjust(width) for value in predicted_values
    ) + " | total"
    print(header)
    print("-" * len(header))
    for row_index, oracle_value in enumerate(oracle_values):
        row = matrix[row_index]
        print(
            f"O{int(oracle_value)}".rjust(12)
            + "".join(f"{int(value):>{width}d}" for value in row)
            + f" | {int(row.sum())}"
        )
    totals = matrix.sum(axis=0)
    print("-" * len(header))
    print(
        "pred total".rjust(12)
        + "".join(f"{int(value):>{width}d}" for value in totals)
        + f" | {int(matrix.sum())}"
    )


def _domain_rows(
    source_values: np.ndarray,
    target_values: np.ndarray,
    matrix: np.ndarray,
    source_name: str,
    target_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, source_value in enumerate(source_values):
        counts = matrix[index]
        total = int(counts.sum())
        nonzero = np.flatnonzero(counts)
        dominant_index = int(np.argmax(counts))
        dominant_count = int(counts[dominant_index])
        rows.append(
            {
                f"{source_name}_domain": int(source_value),
                "point_count": total,
                f"dominant_{target_name}_domain": int(target_values[dominant_index]),
                "dominant_count": dominant_count,
                "purity": dominant_count / float(total),
                f"overlapping_{target_name}_domains": int(len(nonzero)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark")
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark-results")
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory  # noqa: E402
    import run_selected_exact_leapfrog_lva as exact  # noqa: E402
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # noqa: E402

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(  # noqa: SLF001
        args.decoded_root, case_name
    )
    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = comparison._parse_case_parameters(case_name)  # noqa: SLF001
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        strength,
        trend_range,
    )

    builder = polatory.AutomaticStructuralDomainBuilder3(
        centroid_count=args.centroid_count,
        minimum_cluster_fraction=args.minimum_fraction,
        maximum_cluster_fraction=args.maximum_fraction,
        consistency_threshold=args.threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, _, shape, centroids = builder._prepare_grid(points)  # noqa: SLF001
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    predicted_labels, _, minimum_points, maximum_points, merge_count = (
        builder._automatic_labels(  # noqa: SLF001
            np.asarray(points, dtype=np.float64),
            np.asarray(centroid_anisotropies, dtype=np.float64),
            np.asarray(minimum, dtype=np.float64),
            np.asarray(maximum, dtype=np.float64),
            tuple(int(value) for value in shape),
        )
    )
    oracle_labels = np.asarray(oracle_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)

    oracle_values, predicted_values, matrix = _contingency(
        oracle_labels, predicted_labels
    )
    oracle_rows = _domain_rows(
        oracle_values, predicted_values, matrix, "oracle", "predicted"
    )
    predicted_rows = _domain_rows(
        predicted_values, oracle_values, matrix.T, "predicted", "oracle"
    )

    oracle_sizes = matrix.sum(axis=1)
    predicted_sizes = matrix.sum(axis=0)
    intersection_same_pairs = _choose2(matrix.ravel())
    oracle_same_pairs = _choose2(oracle_sizes)
    predicted_same_pairs = _choose2(predicted_sizes)
    false_split_pairs = oracle_same_pairs - intersection_same_pairs
    false_merge_pairs = predicted_same_pairs - intersection_same_pairs

    oracle_refines_predicted = all(
        int(row["overlapping_predicted_domains"]) == 1 for row in oracle_rows
    )
    predicted_refines_oracle = all(
        int(row["overlapping_oracle_domains"]) == 1 for row in predicted_rows
    )
    weighted_oracle_purity = sum(
        int(row["dominant_count"]) for row in oracle_rows
    ) / float(len(points))
    weighted_predicted_purity = sum(
        int(row["dominant_count"]) for row in predicted_rows
    ) / float(len(points))
    optimal_accuracy, matched_points = comparison._optimal_label_accuracy(  # noqa: SLF001
        oracle_labels, predicted_labels
    )

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(oracle_values)} predicted_domains={len(predicted_values)}"
    )
    print(
        f"grid_internal_limits: min={minimum_points} max={maximum_points}; "
        f"merges={merge_count}"
    )
    print(
        f"ARI={adjusted_rand_score(oracle_labels, predicted_labels):.5f} "
        f"AMI={adjusted_mutual_info_score(oracle_labels, predicted_labels):.5f} "
        f"optimal_match={optimal_accuracy:.5f} ({matched_points}/{len(points)})"
    )
    print()
    _print_matrix(oracle_values, predicted_values, matrix)

    print("\nOracle domains split across predicted domains:")
    print(f"{'oracle':>8} {'size':>7} {'best_pred':>10} {'best_n':>8} {'purity':>9} {'pred_overlap':>13}")
    for row in oracle_rows:
        print(
            f"{int(row['oracle_domain']):>8d} {int(row['point_count']):>7d} "
            f"{int(row['dominant_predicted_domain']):>10d} "
            f"{int(row['dominant_count']):>8d} {float(row['purity']):>9.4f} "
            f"{int(row['overlapping_predicted_domains']):>13d}"
        )

    print("\nPredicted domains mixing oracle domains:")
    print(f"{'pred':>8} {'size':>7} {'best_oracle':>11} {'best_n':>8} {'purity':>9} {'oracle_overlap':>14}")
    for row in predicted_rows:
        print(
            f"{int(row['predicted_domain']):>8d} {int(row['point_count']):>7d} "
            f"{int(row['dominant_oracle_domain']):>11d} "
            f"{int(row['dominant_count']):>8d} {float(row['purity']):>9.4f} "
            f"{int(row['overlapping_oracle_domains']):>14d}"
        )

    print("\nPartition diagnosis:")
    print(f"oracle_is_refinement_of_predicted={oracle_refines_predicted}")
    print(f"predicted_is_refinement_of_oracle={predicted_refines_oracle}")
    print(f"weighted_oracle_to_predicted_purity={weighted_oracle_purity:.5f}")
    print(f"weighted_predicted_to_oracle_purity={weighted_predicted_purity:.5f}")
    print(
        f"false_split_pairs={false_split_pairs} "
        f"({false_split_pairs / float(max(oracle_same_pairs, 1)):.5f} of oracle-same pairs)"
    )
    print(
        f"false_merge_pairs={false_merge_pairs} "
        f"({false_merge_pairs / float(max(predicted_same_pairs, 1)):.5f} of predicted-same pairs)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"oracle-partition-overlap-{case_name.lower()}"
    matrix_path = args.output_dir / f"{stem}-matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle_domain", *[f"predicted_{int(v)}" for v in predicted_values], "total"])
        for index, oracle_value in enumerate(oracle_values):
            writer.writerow(
                [int(oracle_value), *[int(v) for v in matrix[index]], int(matrix[index].sum())]
            )
        writer.writerow(["total", *[int(v) for v in predicted_sizes], int(matrix.sum())])

    summary = {
        "case": case_name,
        "point_count": int(len(points)),
        "grid_shape": [int(value) for value in shape],
        "oracle_domain_count": int(len(oracle_values)),
        "predicted_domain_count": int(len(predicted_values)),
        "oracle_domain_sizes": [int(value) for value in oracle_sizes],
        "predicted_domain_sizes": [int(value) for value in predicted_sizes],
        "adjusted_rand_index": float(adjusted_rand_score(oracle_labels, predicted_labels)),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(oracle_labels, predicted_labels)
        ),
        "optimal_label_accuracy": float(optimal_accuracy),
        "matched_points": int(matched_points),
        "oracle_is_refinement_of_predicted": bool(oracle_refines_predicted),
        "predicted_is_refinement_of_oracle": bool(predicted_refines_oracle),
        "weighted_oracle_to_predicted_purity": float(weighted_oracle_purity),
        "weighted_predicted_to_oracle_purity": float(weighted_predicted_purity),
        "oracle_same_pairs": int(oracle_same_pairs),
        "predicted_same_pairs": int(predicted_same_pairs),
        "intersection_same_pairs": int(intersection_same_pairs),
        "false_split_pairs": int(false_split_pairs),
        "false_merge_pairs": int(false_merge_pairs),
        "oracle_domains": oracle_rows,
        "predicted_domains": predicted_rows,
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {matrix_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
