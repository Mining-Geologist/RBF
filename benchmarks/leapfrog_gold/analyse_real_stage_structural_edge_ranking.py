"""Diagnose endpoint-relative structural ranking on real-stage Delaunay edges.

The preceding diagnostics established two constraints:

* the within-coarse-domain Delaunay graph preserves decoded oracle-fragment
  connectivity, but contains many cross-fragment edges; and
* replacing it with Euclidean k-NN or applying a single global matrix-similarity
  threshold destroys too much valid connectivity.

This diagnostic therefore keeps the Delaunay candidate graph and evaluates local,
endpoint-relative edge rules.  It compares structural-similarity ranks, anisotropic
edge-distance ranks, local score percentiles, tensor-direction alignment, and
Delaunay-neighbourhood agreement.  Production code is unchanged; decoded oracle
labels are used only to measure each diagnostic graph.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse_real_stage_edge_consistency as edge_base  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _normalised_rank(values: np.ndarray, descending: bool) -> np.ndarray:
    """Return endpoint-local percentile in [0, 1], where one is best."""
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(values, kind="mergesort")
    if descending:
        order = order[::-1]
    result = np.empty(count, dtype=float)
    if count == 1:
        result[order[0]] = 1.0
        return result
    result[order] = 1.0 - np.arange(count, dtype=float) / float(count - 1)
    return result


def _edge_features(
    points: np.ndarray,
    matrices: np.ndarray,
    edges: list[tuple[int, int]],
) -> list[dict[str, float | int]]:
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
    )

    adjacency: dict[int, set[int]] = defaultdict(set)
    incident: dict[int, list[int]] = defaultdict(list)
    rows: list[dict[str, float | int]] = []

    for edge_index, (first, second) in enumerate(edges):
        first_i, second_i = int(first), int(second)
        adjacency[first_i].add(second_i)
        adjacency[second_i].add(first_i)
        incident[first_i].append(edge_index)
        incident[second_i].append(edge_index)

        delta = np.asarray(points[second_i] - points[first_i], dtype=float)
        direction = _safe_unit(delta)
        euclidean = float(np.linalg.norm(delta))

        first_matrix = np.asarray(matrices[first_i], dtype=float)
        second_matrix = np.asarray(matrices[second_i], dtype=float)
        merged_matrix, consistency = _merged_matrix_and_consistency(
            first_matrix, 1, second_matrix, 1
        )
        merged_matrix = 0.5 * (
            np.asarray(merged_matrix, dtype=float)
            + np.asarray(merged_matrix, dtype=float).T
        )

        direct_q = float(max(delta @ merged_matrix @ delta, 0.0))
        direct_distance = float(np.sqrt(direct_q))
        try:
            inverse_matrix = np.linalg.inv(merged_matrix)
            inverse_q = float(max(delta @ inverse_matrix @ delta, 0.0))
            inverse_distance = float(np.sqrt(inverse_q))
        except np.linalg.LinAlgError:
            inverse_distance = float("inf")

        eigenvalues, eigenvectors = np.linalg.eigh(merged_matrix)
        least_penalised = _safe_unit(eigenvectors[:, int(np.argmin(eigenvalues))])
        most_penalised = _safe_unit(eigenvectors[:, int(np.argmax(eigenvalues))])
        least_alignment = float(abs(direction @ least_penalised))
        most_alignment = float(abs(direction @ most_penalised))

        rows.append(
            {
                "first": first_i,
                "second": second_i,
                "similarity": float(np.clip(consistency, 0.0, 1.0)),
                "euclidean": euclidean,
                "tensor_direct": direct_distance,
                "tensor_inverse": inverse_distance,
                "align_least": least_alignment,
                "align_most": most_alignment,
            }
        )

    for row in rows:
        first_i, second_i = int(row["first"]), int(row["second"])
        common = adjacency[first_i].intersection(adjacency[second_i])
        union = adjacency[first_i].union(adjacency[second_i])
        union.discard(first_i)
        union.discard(second_i)
        row["common_count"] = int(len(common))
        row["common_jaccard"] = float(len(common) / max(len(union), 1))

    rank_fields = (
        ("similarity", True, "similarity_pct"),
        ("euclidean", False, "euclidean_pct"),
        ("tensor_direct", False, "tensor_direct_pct"),
        ("tensor_inverse", False, "tensor_inverse_pct"),
        ("align_least", True, "align_least_pct"),
        ("align_most", True, "align_most_pct"),
        ("common_jaccard", True, "common_pct"),
    )
    for node, edge_indices in incident.items():
        for field, descending, output in rank_fields:
            values = np.asarray(
                [float(rows[index][field]) for index in edge_indices], dtype=float
            )
            ranks = _normalised_rank(values, descending=descending)
            for index, rank in zip(edge_indices, ranks):
                if node == int(rows[index]["first"]):
                    rows[index][f"{output}_first"] = float(rank)
                else:
                    rows[index][f"{output}_second"] = float(rank)

    for row in rows:
        for output in (
            "similarity_pct",
            "euclidean_pct",
            "tensor_direct_pct",
            "tensor_inverse_pct",
            "align_least_pct",
            "align_most_pct",
            "common_pct",
        ):
            first_value = float(row[f"{output}_first"])
            second_value = float(row[f"{output}_second"])
            row[f"{output}_either"] = max(first_value, second_value)
            row[f"{output}_both"] = min(first_value, second_value)

        row["score_similarity_alignment"] = (
            0.70 * float(row["similarity_pct_either"])
            + 0.30 * float(row["align_least_pct_either"])
        )
        row["score_similarity_tensor"] = (
            0.65 * float(row["similarity_pct_either"])
            + 0.35 * float(row["tensor_direct_pct_either"])
        )
        row["score_similarity_common"] = (
            0.70 * float(row["similarity_pct_either"])
            + 0.30 * float(row["common_pct_either"])
        )
        row["score_structural_combined"] = (
            0.50 * float(row["similarity_pct_either"])
            + 0.25 * float(row["tensor_direct_pct_either"])
            + 0.15 * float(row["align_least_pct_either"])
            + 0.10 * float(row["common_pct_either"])
        )
    return rows


def _evaluate_graph(
    groups: list[dict[str, object]],
    selected: dict[int, set[int]],
) -> dict[str, float | int]:
    edge_count = 0
    same_edges = 0
    cross_edges = 0
    fragment_total = 0
    fragment_connected = 0
    split_excess = 0
    largest_numerator = 0
    total_points = 0

    for group_index, group in enumerate(groups):
        local_oracle = np.asarray(group["oracle"], dtype=np.int64)
        rows = group["rows"]
        kept_indices = selected.get(group_index, set())
        kept_edges: list[tuple[int, int]] = []

        for edge_index in kept_indices:
            row = rows[edge_index]
            first, second = int(row["first"]), int(row["second"])
            kept_edges.append((first, second))
            edge_count += 1
            if int(local_oracle[first]) == int(local_oracle[second]):
                same_edges += 1
            else:
                cross_edges += 1

        for oracle_value in np.unique(local_oracle):
            fragment_indices = np.flatnonzero(local_oracle == oracle_value)
            fragment_edges = [
                (first, second)
                for first, second in kept_edges
                if int(local_oracle[first]) == int(oracle_value)
                and int(local_oracle[second]) == int(oracle_value)
            ]
            components = edge_base._component_count(fragment_indices, fragment_edges)
            fragment_total += 1
            fragment_connected += int(components == 1)
            split_excess += max(components - 1, 0)

            if len(fragment_indices):
                mapping = {
                    int(value): position
                    for position, value in enumerate(fragment_indices)
                }
                uf = edge_base.UnionFind(len(fragment_indices))
                for first, second in fragment_edges:
                    uf.union(mapping[first], mapping[second])
                component_sizes: dict[int, int] = defaultdict(int)
                for position in range(len(fragment_indices)):
                    component_sizes[uf.find(position)] += 1
                largest_numerator += max(component_sizes.values(), default=0)
                total_points += len(fragment_indices)

    return {
        "edges": edge_count,
        "same_edges": same_edges,
        "cross_edges": cross_edges,
        "purity": same_edges / max(edge_count, 1),
        "same_recall": 0.0,
        "fragments_connected": fragment_connected / max(fragment_total, 1),
        "split_excess": split_excess,
        "largest_fraction": largest_numerator / max(total_points, 1),
    }


def _select_all(groups: list[dict[str, object]]) -> dict[int, set[int]]:
    return {
        group_index: set(range(len(group["rows"])))
        for group_index, group in enumerate(groups)
    }


def _select_threshold(
    groups: list[dict[str, object]],
    field: str,
    threshold: float,
) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    for group_index, group in enumerate(groups):
        selected[group_index] = {
            edge_index
            for edge_index, row in enumerate(group["rows"])
            if float(row[field]) >= threshold
        }
    return selected


def _select_endpoint_topk(
    groups: list[dict[str, object]],
    field: str,
    k: int,
    descending: bool,
    mutual: bool,
) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    for group_index, group in enumerate(groups):
        rows = group["rows"]
        incident: dict[int, list[int]] = defaultdict(list)
        for edge_index, row in enumerate(rows):
            incident[int(row["first"])].append(edge_index)
            incident[int(row["second"])].append(edge_index)

        chosen_by_node: dict[int, set[int]] = {}
        for node, edge_indices in incident.items():
            ordered = sorted(
                edge_indices,
                key=lambda index: float(rows[index][field]),
                reverse=descending,
            )
            chosen_by_node[node] = set(
                ordered[: min(max(int(k), 1), len(ordered))]
            )

        kept: set[int] = set()
        for edge_index, row in enumerate(rows):
            first = int(row["first"])
            second = int(row["second"])
            selected_first = edge_index in chosen_by_node[first]
            selected_second = edge_index in chosen_by_node[second]
            if (
                selected_first and selected_second
                if mutual
                else selected_first or selected_second
            ):
                kept.add(edge_index)
        selected[group_index] = kept
    return selected


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
    parser.add_argument("--minimum-connectivity", type=float, default=0.90)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import _normalise_determinant

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
    coarse_labels = np.asarray(coarse_labels, dtype=np.int64)

    groups: list[dict[str, object]] = []
    for coarse_value in np.unique(coarse_labels):
        global_indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[global_indices]
        local_matrices = point_matrices[global_indices]
        local_oracle = oracle_labels[global_indices]
        local_edges = edge_base._delaunay_edges(local_points)
        groups.append(
            {
                "coarse": int(coarse_value),
                "oracle": local_oracle,
                "rows": _edge_features(local_points, local_matrices, local_edges),
            }
        )

    candidates: list[tuple[str, dict[int, set[int]]]] = [
        ("delaunay", _select_all(groups)),
    ]

    for k in (2, 3, 4, 5, 6, 8, 10, 12):
        candidates.append(
            (
                f"similarity_top{k}_either",
                _select_endpoint_topk(
                    groups, "similarity", k, descending=True, mutual=False
                ),
            )
        )
        candidates.append(
            (
                f"similarity_top{k}_both",
                _select_endpoint_topk(
                    groups, "similarity", k, descending=True, mutual=True
                ),
            )
        )
        candidates.append(
            (
                f"tensor_top{k}_either",
                _select_endpoint_topk(
                    groups, "tensor_direct", k, descending=False, mutual=False
                ),
            )
        )
        candidates.append(
            (
                f"tensor_top{k}_both",
                _select_endpoint_topk(
                    groups, "tensor_direct", k, descending=False, mutual=True
                ),
            )
        )

    threshold_fields = (
        "similarity_pct_either",
        "similarity_pct_both",
        "tensor_direct_pct_either",
        "tensor_direct_pct_both",
        "tensor_inverse_pct_either",
        "align_least_pct_either",
        "common_pct_either",
        "score_similarity_alignment",
        "score_similarity_tensor",
        "score_similarity_common",
        "score_structural_combined",
    )
    for field in threshold_fields:
        for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            candidates.append(
                (
                    f"{field}@{threshold:.2f}",
                    _select_threshold(groups, field, threshold),
                )
            )

    baseline = _evaluate_graph(groups, candidates[0][1])
    total_same = int(baseline["same_edges"])
    rows: list[dict[str, float | int | str]] = []
    for name, selected in candidates:
        metrics = _evaluate_graph(groups, selected)
        metrics["name"] = name
        metrics["same_recall"] = int(metrics["same_edges"]) / max(total_same, 1)
        rows.append(metrics)

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    print(
        f"Delaunay edges={int(baseline['edges'])} "
        f"same={int(baseline['same_edges'])} "
        f"cross={int(baseline['cross_edges'])} "
        f"fragments_connected={float(baseline['fragments_connected']):.4f} "
        f"largest_fraction={float(baseline['largest_fraction']):.4f}"
    )

    print("\nTop candidates retaining requested fragment connectivity:")
    print(
        " rule                                      edges purity recall connected "
        "largest split cross"
    )
    feasible = [
        row
        for row in rows
        if float(row["fragments_connected"]) >= args.minimum_connectivity
    ]
    ranking = sorted(
        feasible,
        key=lambda row: (
            int(row["cross_edges"]),
            -float(row["purity"]),
            -float(row["same_recall"]),
            int(row["split_excess"]),
        ),
    )
    for row in ranking[:30]:
        print(
            f" {str(row['name']):<41s} "
            f"{int(row['edges']):>5d} {float(row['purity']):>6.3f} "
            f"{float(row['same_recall']):>6.3f} "
            f"{float(row['fragments_connected']):>9.3f} "
            f"{float(row['largest_fraction']):>7.3f} "
            f"{int(row['split_excess']):>5d} {int(row['cross_edges']):>5d}"
        )

    print("\nBest candidate at each connectivity floor:")
    print(
        " floor rule                                      purity recall largest "
        "split cross"
    )
    for floor in (0.95, 0.90, 0.85, 0.80, 0.70):
        floor_rows = [
            row for row in rows if float(row["fragments_connected"]) >= floor
        ]
        if not floor_rows:
            print(f" {floor:>4.2f} none")
            continue
        best = min(
            floor_rows,
            key=lambda row: (
                int(row["cross_edges"]),
                -float(row["purity"]),
                -float(row["same_recall"]),
                int(row["split_excess"]),
            ),
        )
        print(
            f" {floor:>4.2f} {str(best['name']):<41s} "
            f"{float(best['purity']):>6.3f} "
            f"{float(best['same_recall']):>6.3f} "
            f"{float(best['largest_fraction']):>7.3f} "
            f"{int(best['split_excess']):>5d} "
            f"{int(best['cross_edges']):>5d}"
        )

    print("\nSignal diagnostics (ROC AUC; larger means stronger separation):")
    from sklearn.metrics import roc_auc_score

    flat_same: list[int] = []
    flat_features: dict[str, list[float]] = defaultdict(list)
    signal_fields = (
        "similarity",
        "similarity_pct_either",
        "similarity_pct_both",
        "tensor_direct_pct_either",
        "tensor_inverse_pct_either",
        "align_least",
        "align_most",
        "common_jaccard",
        "score_similarity_alignment",
        "score_similarity_tensor",
        "score_similarity_common",
        "score_structural_combined",
    )
    for group in groups:
        local_oracle = np.asarray(group["oracle"], dtype=np.int64)
        for row in group["rows"]:
            first, second = int(row["first"]), int(row["second"])
            flat_same.append(int(local_oracle[first] == local_oracle[second]))
            for field in signal_fields:
                flat_features[field].append(float(row[field]))

    labels = np.asarray(flat_same, dtype=np.int8)
    auc_rows = []
    for field in signal_fields:
        values = np.asarray(flat_features[field], dtype=float)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) == 0 or len(np.unique(labels[finite])) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(labels[finite], values[finite]))
        auc_rows.append((auc, field))
    for auc, field in sorted(auc_rows, reverse=True):
        print(f" {field:<34s} {auc:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
