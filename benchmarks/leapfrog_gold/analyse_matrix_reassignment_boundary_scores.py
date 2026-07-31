"""Analyse boundary-level evidence for merges after one real-point reassignment.

Whole-domain matrix consistency does not select the oracle-improving merge in the
S3_R100 and S3_R300 diagnostics.  This script therefore measures the shared boundary
between every Delaunay-adjacent reassigned-domain pair: cross-boundary edge count,
unique boundary-point support, edge lengths, and point-pair matrix consistency.  It
reports oracle metrics only as diagnostics and ranks several model-only scores to see
whether the same rule can identify the useful merge across cases.
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

import analyse_matrix_reassignment_pair_merges as pair_analysis  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_matrix_reassignment as reassignment  # noqa: E402


def _delaunay_point_edges(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    simplices = np.asarray(Delaunay(points, qhull_options="QJ").simplices, dtype=np.int64)
    edges: list[tuple[int, int]] = []
    for simplex in simplices:
        edges.extend(combinations((int(value) for value in simplex), 2))
    array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    array.sort(axis=1)
    array = array[array[:, 0] != array[:, 1]]
    return np.unique(array, axis=0)


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

    labels, changes, history = reassignment._run_reassignment(
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
    labels = reassignment._relabel(labels)

    def metrics(candidate: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, candidate)
        return (
            float(adjusted_rand_score(oracle_labels, candidate)),
            float(adjusted_mutual_info_score(oracle_labels, candidate)),
            float(accuracy),
            int(matched),
        )

    coarse_metrics = metrics(coarse_labels)
    reassigned_metrics = metrics(labels)
    values, domain_matrices, domain_centroids, sizes = reassignment._domain_statistics(
        points, point_matrices, labels, _normalise_determinant
    )
    positions = {int(value): index for index, value in enumerate(values)}

    boundary_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for first_index, second_index in _delaunay_point_edges(points):
        first_label = int(labels[first_index])
        second_label = int(labels[second_index])
        if first_label == second_label:
            continue
        pair = tuple(sorted((first_label, second_label)))
        boundary_edges[pair].append((int(first_index), int(second_index)))

    rows: list[dict[str, object]] = []
    for pair, edges in boundary_edges.items():
        first_label, second_label = pair
        first_position = positions[first_label]
        second_position = positions[second_label]
        first_size = int(sizes[first_position])
        second_size = int(sizes[second_position])

        whole_merged, whole_consistency = _merged_matrix_and_consistency(
            domain_matrices[first_position],
            first_size,
            domain_matrices[second_position],
            second_size,
        )
        del whole_merged

        lengths: list[float] = []
        local_consistencies: list[float] = []
        first_boundary: set[int] = set()
        second_boundary: set[int] = set()
        for first_index, second_index in edges:
            if int(labels[first_index]) == second_label:
                first_index, second_index = second_index, first_index
            first_boundary.add(first_index)
            second_boundary.add(second_index)
            lengths.append(float(np.linalg.norm(points[first_index] - points[second_index])))
            _, local_consistency = _merged_matrix_and_consistency(
                point_matrices[first_index], 1, point_matrices[second_index], 1
            )
            local_consistencies.append(float(local_consistency))

        local = np.asarray(local_consistencies, dtype=float)
        edge_lengths = np.asarray(lengths, dtype=float)
        edge_count = len(edges)
        first_support = len(first_boundary)
        second_support = len(second_boundary)
        support_min_fraction = min(
            first_support / max(first_size, 1),
            second_support / max(second_size, 1),
        )
        support_geomean_fraction = float(
            np.sqrt(
                (first_support / max(first_size, 1))
                * (second_support / max(second_size, 1))
            )
        )
        edge_density = edge_count / max(np.sqrt(first_size * second_size), 1.0)
        mean_local = float(np.mean(local))
        median_local = float(np.median(local))
        p10_local = float(np.quantile(local, 0.10))
        excess_mean = float(np.mean(np.maximum(local - args.threshold, 0.0)))
        mean_length = float(np.mean(edge_lengths))
        centroid_distance = float(
            np.linalg.norm(domain_centroids[first_position] - domain_centroids[second_position])
        )

        merged_labels = pair_analysis._merge_pair(labels, first_label, second_label)
        ari, ami, match, matched = metrics(merged_labels)
        rows.append(
            {
                "pair": pair,
                "sizes": (first_size, second_size),
                "combined": first_size + second_size,
                "whole": float(whole_consistency),
                "edges": edge_count,
                "support": (first_support, second_support),
                "support_min_fraction": support_min_fraction,
                "support_geomean_fraction": support_geomean_fraction,
                "edge_density": edge_density,
                "mean_local": mean_local,
                "median_local": median_local,
                "p10_local": p10_local,
                "excess_mean": excess_mean,
                "mean_length": mean_length,
                "centroid_distance": centroid_distance,
                "score_contact": support_geomean_fraction * mean_local,
                "score_edges": edge_density * mean_local,
                "score_excess": support_geomean_fraction * excess_mean,
                "ari": ari,
                "ami": ami,
                "match": match,
                "matched": matched,
            }
        )

    def format_row(row: dict[str, object]) -> str:
        first, second = row["pair"]
        first_size, second_size = row["sizes"]
        first_support, second_support = row["support"]
        return (
            f"pair=({first},{second}) sizes=({first_size},{second_size}) "
            f"edges={int(row['edges']):4d} support=({first_support},{second_support}) "
            f"support_min={float(row['support_min_fraction']):.3f} "
            f"edge_density={float(row['edge_density']):.3f} "
            f"local[mean={float(row['mean_local']):.6f},p10={float(row['p10_local']):.6f}] "
            f"whole={float(row['whole']):.6f} mean_len={float(row['mean_length']):.2f} "
            f"score_contact={float(row['score_contact']):.6f} "
            f"score_edges={float(row['score_edges']):.6f} "
            f"ARI={float(row['ari']):.5f} match={float(row['match']):.5f} "
            f"({int(row['matched'])}/{len(points)})"
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
        f"reassigned: changes={changes} history={history} domains={len(values)} "
        f"ARI={reassigned_metrics[0]:.5f} AMI={reassigned_metrics[1]:.5f} "
        f"match={reassigned_metrics[2]:.5f} ({reassigned_metrics[3]}/{len(points)})"
    )

    top = max(args.top, 1)
    rankings = (
        ("oracle ARI diagnostic", "ari"),
        ("boundary contact score", "score_contact"),
        ("boundary edge-density score", "score_edges"),
        ("boundary excess-consistency score", "score_excess"),
        ("whole-domain consistency", "whole"),
        ("boundary support", "support_geomean_fraction"),
    )
    for title, key in rankings:
        print(f"\nTop {top} by {title}:")
        for row in sorted(rows, key=lambda item: float(item[key]), reverse=True)[:top]:
            print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
