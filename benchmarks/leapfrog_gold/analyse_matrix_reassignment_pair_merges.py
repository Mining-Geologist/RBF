"""Analyse one-step matrix reassignment followed by a single domain merge.

The best standalone diagnostic so far is one reassignment iteration using the two
nearest coarse-domain centroids and no spatial penalty.  It improves S3_R100 from
ARI 0.47915 to about 0.50083 but preserves ten domains while Leapfrog has nine.
This script tests every possible pair merge after that reassignment and reports
whether the missing count correction could plausibly be one final merge.  Oracle
metrics are diagnostic only; model-side adjacency and matrix consistency are also
reported so the result is not interpreted as a production tuning rule.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_matrix_reassignment as reassignment  # noqa: E402


def _merge_pair(labels: np.ndarray, first: int, second: int) -> np.ndarray:
    merged = np.asarray(labels, dtype=np.int64).copy()
    merged[merged == second] = first
    return reassignment._relabel(merged)


def _delaunay_domain_pairs(points: np.ndarray, labels: np.ndarray) -> set[tuple[int, int]]:
    from scipy.spatial import Delaunay, QhullError

    pairs: set[tuple[int, int]] = set()
    try:
        simplices = Delaunay(points, qhull_options="QJ").simplices
    except QhullError:
        return pairs
    for simplex in simplices:
        simplex_labels = sorted(set(int(labels[int(index)]) for index in simplex))
        for first, second in combinations(simplex_labels, 2):
            pairs.add((first, second))
    return pairs


def _knn_domain_pairs(points: np.ndarray, labels: np.ndarray, k: int) -> set[tuple[int, int]]:
    from scipy.spatial import cKDTree

    count = len(points)
    actual_k = min(max(int(k), 1), max(count - 1, 1))
    neighbours = np.asarray(cKDTree(points).query(points, k=actual_k + 1)[1])[:, 1:]
    pairs: set[tuple[int, int]] = set()
    for first_index, row in enumerate(neighbours):
        first_label = int(labels[first_index])
        for second_index in row:
            second_label = int(labels[int(second_index)])
            if first_label != second_label:
                pairs.add(tuple(sorted((first_label, second_label))))
    return pairs


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
    parser.add_argument("--nearest-domains", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--spatial-penalty", type=float, default=0.0)
    parser.add_argument("--knn", type=int, default=6)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
        _normalise_determinant,
    )
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(
        args.decoded_root, case_name
    )
    points = np.asarray(points, dtype=np.float64)
    oracle_labels = np.asarray(oracle_labels, dtype=np.int64)

    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = comparison._parse_case_parameters(case_name)
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
    minimum, maximum, _, shape, centroids = builder._prepare_grid(points)
    centroid_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    point_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        points, trend_input, non_decaying=False
    )
    point_matrices = np.asarray(
        [_normalise_determinant(matrix) for matrix in point_matrices], dtype=float
    )
    coarse_labels, _, _, _, coarse_merges = builder._automatic_labels(
        points,
        np.asarray(centroid_matrices, dtype=np.float64),
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        tuple(int(value) for value in shape),
    )

    reassigned, changes, history = reassignment._run_reassignment(
        points,
        point_matrices,
        coarse_labels,
        nearest_domains=args.nearest_domains,
        spatial_penalty=args.spatial_penalty,
        iterations=args.iterations,
        threshold=args.threshold,
        normalise=_normalise_determinant,
        merged_score=_merged_matrix_and_consistency,
    )
    reassigned = reassignment._relabel(reassigned)

    def metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(accuracy),
            int(matched),
        )

    coarse_metrics = metrics(coarse_labels)
    reassigned_metrics = metrics(reassigned)
    values, matrices, domain_centroids, sizes = reassignment._domain_statistics(
        points, point_matrices, reassigned, _normalise_determinant
    )
    value_to_position = {int(value): index for index, value in enumerate(values)}

    delaunay_pairs = _delaunay_domain_pairs(points, reassigned)
    knn_pairs = _knn_domain_pairs(points, reassigned, args.knn)
    spatial_extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), np.finfo(float).eps)

    rows: list[dict[str, object]] = []
    for first_value, second_value in combinations((int(value) for value in values), 2):
        first_position = value_to_position[first_value]
        second_position = value_to_position[second_value]
        _, consistency = _merged_matrix_and_consistency(
            matrices[first_position],
            int(sizes[first_position]),
            matrices[second_position],
            int(sizes[second_position]),
        )
        distance = float(
            np.linalg.norm(domain_centroids[first_position] - domain_centroids[second_position])
        )
        merged_labels = _merge_pair(reassigned, first_value, second_value)
        ari, ami, match, matched = metrics(merged_labels)
        pair = tuple(sorted((first_value, second_value)))
        rows.append(
            {
                "pair": pair,
                "sizes": (int(sizes[first_position]), int(sizes[second_position])),
                "combined": int(sizes[first_position] + sizes[second_position]),
                "consistency": float(consistency),
                "distance": distance,
                "distance_fraction": distance / spatial_extent,
                "delaunay": pair in delaunay_pairs,
                "knn": pair in knn_pairs,
                "ari": ari,
                "ami": ami,
                "match": match,
                "matched": matched,
            }
        )

    def format_row(row: dict[str, object]) -> str:
        first, second = row["pair"]
        first_size, second_size = row["sizes"]
        return (
            f"merge=({first},{second}) sizes=({first_size},{second_size}) "
            f"combined={int(row['combined']):3d} consistency={float(row['consistency']):.6f} "
            f"distance={float(row['distance']):.3f} "
            f"adj[delaunay={bool(row['delaunay'])},knn{args.knn}={bool(row['knn'])}] "
            f"ARI={float(row['ari']):.5f} AMI={float(row['ami']):.5f} "
            f"match={float(row['match']):.5f} ({int(row['matched'])}/{len(points)})"
        )

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} coarse_domains={len(np.unique(coarse_labels))}"
    )
    print(
        f"coarse: merges={coarse_merges} ARI={coarse_metrics[0]:.5f} "
        f"AMI={coarse_metrics[1]:.5f} match={coarse_metrics[2]:.5f} "
        f"({coarse_metrics[3]}/{len(points)})"
    )
    print(
        f"reassigned: near={args.nearest_domains} penalty={args.spatial_penalty:.3f} "
        f"iterations={args.iterations} changes={changes} history={history} "
        f"domains={len(values)} ARI={reassigned_metrics[0]:.5f} "
        f"AMI={reassigned_metrics[1]:.5f} match={reassigned_metrics[2]:.5f} "
        f"({reassigned_metrics[3]}/{len(points)})"
    )

    top = max(args.top, 1)
    print(f"\nTop {top} pair merges by ARI:")
    for row in sorted(rows, key=lambda item: float(item["ari"]), reverse=True)[:top]:
        print(format_row(row))

    adjacent_rows = [row for row in rows if bool(row["delaunay"]) or bool(row["knn"])]
    print(f"\nTop {top} adjacent pair merges by ARI:")
    for row in sorted(adjacent_rows, key=lambda item: float(item["ari"]), reverse=True)[:top]:
        print(format_row(row))

    print(f"\nTop {top} adjacent pair merges by model consistency:")
    for row in sorted(
        adjacent_rows,
        key=lambda item: (float(item["consistency"]), -float(item["distance_fraction"])),
        reverse=True,
    )[:top]:
        print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
