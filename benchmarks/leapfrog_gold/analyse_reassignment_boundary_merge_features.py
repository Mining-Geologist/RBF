"""Analyse boundary-level features for the post-reassignment domain merge.

The best oracle merge after one local matrix reassignment is not consistently the pair
with the highest whole-domain matrix consistency.  This diagnostic measures the
actual interface between adjacent reassigned domains using Delaunay and k-nearest
point graphs: cross-edge counts, boundary support, edge lengths, and boundary-only
matrix consistency.  It reports where the oracle-best merge ranks under each
model-side feature without changing production code.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse_matrix_reassignment_pair_merges as pair_diag  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_matrix_reassignment as reassignment  # noqa: E402


def _unique_delaunay_edges(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay, QhullError

    try:
        simplices = np.asarray(Delaunay(points, qhull_options="QJ").simplices, dtype=np.int64)
    except QhullError:
        return np.empty((0, 2), dtype=np.int64)
    edges: set[tuple[int, int]] = set()
    for simplex in simplices:
        for first, second in combinations((int(value) for value in simplex), 2):
            edges.add((first, second) if first < second else (second, first))
    return np.asarray(sorted(edges), dtype=np.int64)


def _unique_knn_edges(points: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    count = len(points)
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)
    actual_k = min(max(int(k), 1), count - 1)
    neighbours = np.asarray(cKDTree(points).query(points, k=actual_k + 1)[1])[:, 1:]
    edges: set[tuple[int, int]] = set()
    for first, row in enumerate(neighbours):
        for second in row:
            second = int(second)
            edges.add((first, second) if first < second else (second, first))
    return np.asarray(sorted(edges), dtype=np.int64)


def _interface_records(
    points: np.ndarray,
    labels: np.ndarray,
    matrices: np.ndarray,
    edges: np.ndarray,
    *,
    normalise,
    merged_score,
) -> dict[tuple[int, int], dict[str, object]]:
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for first_index, second_index in np.asarray(edges, dtype=np.int64):
        first_label = int(labels[int(first_index)])
        second_label = int(labels[int(second_index)])
        if first_label == second_label:
            continue
        pair = tuple(sorted((first_label, second_label)))
        if first_label == pair[0]:
            grouped[pair].append((int(first_index), int(second_index)))
        else:
            grouped[pair].append((int(second_index), int(first_index)))

    output: dict[tuple[int, int], dict[str, object]] = {}
    for pair, pair_edges in grouped.items():
        edge_array = np.asarray(pair_edges, dtype=np.int64)
        first_points = np.unique(edge_array[:, 0])
        second_points = np.unique(edge_array[:, 1])
        lengths = np.linalg.norm(
            points[edge_array[:, 0]] - points[edge_array[:, 1]], axis=1
        )
        first_matrix = normalise(np.asarray(matrices[first_points]).mean(axis=0))
        second_matrix = normalise(np.asarray(matrices[second_points]).mean(axis=0))
        _, boundary_consistency = merged_score(
            first_matrix,
            len(first_points),
            second_matrix,
            len(second_points),
        )
        edge_consistencies: list[float] = []
        for first_index, second_index in edge_array:
            _, value = merged_score(
                matrices[int(first_index)], 1, matrices[int(second_index)], 1
            )
            if np.isfinite(value):
                edge_consistencies.append(float(value))
        output[pair] = {
            "edges": int(len(edge_array)),
            "first_boundary_points": int(len(first_points)),
            "second_boundary_points": int(len(second_points)),
            "boundary_points": int(len(first_points) + len(second_points)),
            "length_min": float(np.min(lengths)),
            "length_median": float(np.median(lengths)),
            "length_mean": float(np.mean(lengths)),
            "boundary_consistency": float(boundary_consistency),
            "edge_consistency_mean": float(np.mean(edge_consistencies))
            if edge_consistencies
            else float("-inf"),
            "edge_consistency_min": float(np.min(edge_consistencies))
            if edge_consistencies
            else float("-inf"),
        }
    return output


def _rank(rows: list[dict[str, object]], key: str, pair: tuple[int, int], reverse: bool) -> int:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    for index, row in enumerate(ordered, start=1):
        if row["pair"] == pair:
            return index
    return -1


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
    parser.add_argument("--top", type=int, default=12)
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
        match, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(match),
            int(matched),
        )

    values, domain_matrices, domain_centroids, sizes = reassignment._domain_statistics(
        points, point_matrices, reassigned, _normalise_determinant
    )
    position = {int(value): index for index, value in enumerate(values)}
    delaunay = _interface_records(
        points,
        reassigned,
        point_matrices,
        _unique_delaunay_edges(points),
        normalise=_normalise_determinant,
        merged_score=_merged_matrix_and_consistency,
    )
    knn = _interface_records(
        points,
        reassigned,
        point_matrices,
        _unique_knn_edges(points, args.knn),
        normalise=_normalise_determinant,
        merged_score=_merged_matrix_and_consistency,
    )

    rows: list[dict[str, object]] = []
    for pair in sorted(set(delaunay) | set(knn)):
        first, second = pair
        first_position = position[first]
        second_position = position[second]
        _, whole_consistency = _merged_matrix_and_consistency(
            domain_matrices[first_position],
            int(sizes[first_position]),
            domain_matrices[second_position],
            int(sizes[second_position]),
        )
        merged_labels = pair_diag._merge_pair(reassigned, first, second)
        ari, ami, match, matched = metrics(merged_labels)
        d = delaunay.get(pair, {})
        k = knn.get(pair, {})
        rows.append(
            {
                "pair": pair,
                "sizes": (int(sizes[first_position]), int(sizes[second_position])),
                "whole_consistency": float(whole_consistency),
                "centroid_distance": float(
                    np.linalg.norm(
                        domain_centroids[first_position] - domain_centroids[second_position]
                    )
                ),
                "d_edges": int(d.get("edges", 0)),
                "d_boundary_points": int(d.get("boundary_points", 0)),
                "d_length_median": float(d.get("length_median", float("inf"))),
                "d_boundary_consistency": float(
                    d.get("boundary_consistency", float("-inf"))
                ),
                "d_edge_consistency_mean": float(
                    d.get("edge_consistency_mean", float("-inf"))
                ),
                "k_edges": int(k.get("edges", 0)),
                "k_boundary_points": int(k.get("boundary_points", 0)),
                "k_length_median": float(k.get("length_median", float("inf"))),
                "k_boundary_consistency": float(
                    k.get("boundary_consistency", float("-inf"))
                ),
                "k_edge_consistency_mean": float(
                    k.get("edge_consistency_mean", float("-inf"))
                ),
                "ari": ari,
                "ami": ami,
                "match": match,
                "matched": matched,
            }
        )

    coarse_metrics = metrics(coarse_labels)
    reassigned_metrics = metrics(reassigned)
    best = max(rows, key=lambda row: float(row["ari"]))
    best_pair = best["pair"]

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
        f"reassigned: changes={changes} history={history} domains={len(values)} "
        f"ARI={reassigned_metrics[0]:.5f} AMI={reassigned_metrics[1]:.5f} "
        f"match={reassigned_metrics[2]:.5f} ({reassigned_metrics[3]}/{len(points)})"
    )

    print("\nOracle-best adjacent merge and model-side feature ranks:")
    print(
        f"pair={best_pair} sizes={best['sizes']} ARI={best['ari']:.5f} "
        f"AMI={best['ami']:.5f} match={best['match']:.5f} ({best['matched']}/{len(points)})"
    )
    feature_specs = (
        ("whole_consistency", True),
        ("centroid_distance", False),
        ("d_edges", True),
        ("d_boundary_points", True),
        ("d_length_median", False),
        ("d_boundary_consistency", True),
        ("d_edge_consistency_mean", True),
        ("k_edges", True),
        ("k_boundary_points", True),
        ("k_length_median", False),
        ("k_boundary_consistency", True),
        ("k_edge_consistency_mean", True),
    )
    for key, reverse in feature_specs:
        print(
            f"  {key:24s} value={best[key]!s:>12s} "
            f"rank={_rank(rows, key, best_pair, reverse)}/{len(rows)}"
        )

    def format_row(row: dict[str, object]) -> str:
        return (
            f"pair={row['pair']} sizes={row['sizes']} whole={float(row['whole_consistency']):.6f} "
            f"D[edges={int(row['d_edges']):4d},pts={int(row['d_boundary_points']):3d},"
            f"med={float(row['d_length_median']):7.2f},bc={float(row['d_boundary_consistency']):.6f},"
            f"ec={float(row['d_edge_consistency_mean']):.6f}] "
            f"K[edges={int(row['k_edges']):4d},pts={int(row['k_boundary_points']):3d},"
            f"med={float(row['k_length_median']):7.2f},bc={float(row['k_boundary_consistency']):.6f},"
            f"ec={float(row['k_edge_consistency_mean']):.6f}] "
            f"ARI={float(row['ari']):.5f} match={float(row['match']):.5f}"
        )

    top = max(args.top, 1)
    print(f"\nTop {top} adjacent pairs by Delaunay cross-edge count:")
    for row in sorted(rows, key=lambda item: int(item["d_edges"]), reverse=True)[:top]:
        print(format_row(row))

    print(f"\nTop {top} adjacent pairs by Delaunay boundary consistency:")
    for row in sorted(
        rows, key=lambda item: float(item["d_boundary_consistency"]), reverse=True
    )[:top]:
        print(format_row(row))

    print(f"\nTop {top} adjacent pairs by oracle ARI:")
    for row in sorted(rows, key=lambda item: float(item["ari"]), reverse=True)[:top]:
        print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
