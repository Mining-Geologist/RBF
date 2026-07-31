"""Measure whether the recovered point-pair consistency separates oracle fragments.

Runtime evidence and connectivity tests make a within-coarse-domain Delaunay graph the
most plausible candidate graph for Leapfrog's second SubDomainer.  This diagnostic
scores every such edge with the currently recovered matrix-consistency equation and
checks whether any threshold can simultaneously preserve connectivity inside decoded
oracle fragments and reject edges crossing between them.  Production code is unchanged.
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


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def _delaunay_edges(points: np.ndarray) -> list[tuple[int, int]]:
    from scipy.spatial import Delaunay, QhullError

    if len(points) < 2:
        return []
    if len(points) == 2:
        return [(0, 1)]
    try:
        simplices = Delaunay(points, qhull_options="QJ").simplices
    except QhullError:
        return []
    edges: set[tuple[int, int]] = set()
    for simplex in simplices:
        for first, second in combinations((int(value) for value in simplex), 2):
            if first != second:
                edges.add(tuple(sorted((first, second))))
    return sorted(edges)


def _component_count(indices: np.ndarray, edges: list[tuple[int, int]]) -> int:
    if len(indices) == 0:
        return 0
    mapping = {int(value): position for position, value in enumerate(indices)}
    uf = UnionFind(len(indices))
    for first, second in edges:
        if first in mapping and second in mapping:
            uf.union(mapping[first], mapping[second])
    return len({uf.find(index) for index in range(len(indices))})


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
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
        _normalise_determinant,
    )
    from sklearn.metrics import roc_auc_score

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

    records: list[dict[str, object]] = []
    local_groups: list[dict[str, object]] = []
    for coarse_value in np.unique(coarse_labels):
        global_indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[global_indices]
        local_oracle = oracle_labels[global_indices]
        local_edges = _delaunay_edges(local_points)
        scored_edges: list[tuple[int, int, float, bool]] = []
        for first, second in local_edges:
            _, consistency = _merged_matrix_and_consistency(
                point_matrices[global_indices[first]],
                1,
                point_matrices[global_indices[second]],
                1,
            )
            same = bool(local_oracle[first] == local_oracle[second])
            score = float(consistency)
            scored_edges.append((first, second, score, same))
            records.append(
                {
                    "coarse": int(coarse_value),
                    "first": int(first),
                    "second": int(second),
                    "score": score,
                    "same": same,
                }
            )
        local_groups.append(
            {
                "coarse": int(coarse_value),
                "indices": global_indices,
                "oracle": local_oracle,
                "edges": scored_edges,
            }
        )

    scores = np.asarray([float(record["score"]) for record in records], dtype=float)
    same_flags = np.asarray([bool(record["same"]) for record in records], dtype=bool)
    same_scores = scores[same_flags]
    cross_scores = scores[~same_flags]
    auc = float(roc_auc_score(same_flags.astype(np.int8), scores))

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    print(
        f"Delaunay edges={len(scores)} same={len(same_scores)} cross={len(cross_scores)} "
        f"same_fraction={len(same_scores) / max(len(scores), 1):.5f} ROC_AUC={auc:.5f}"
    )

    quantiles = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    print("\nConsistency quantiles:")
    print(" quantile       same       cross")
    for quantile in quantiles:
        same_value = float(np.quantile(same_scores, quantile)) if len(same_scores) else np.nan
        cross_value = float(np.quantile(cross_scores, quantile)) if len(cross_scores) else np.nan
        print(f" {quantile:>7.2f}  {same_value:>10.7f}  {cross_value:>10.7f}")

    fixed_thresholds = [
        0.60,
        0.80,
        0.90,
        0.95,
        0.97,
        0.98,
        0.99,
        0.995,
        0.997,
        0.999,
        0.9995,
        0.9999,
    ]
    data_thresholds = []
    for values in (same_scores, cross_scores):
        if len(values):
            data_thresholds.extend(float(np.quantile(values, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
    thresholds = sorted(set(fixed_thresholds + data_thresholds))

    print("\nThreshold sweep on within-coarse Delaunay edges:")
    print(
        " threshold  kept  purity  same_recall  fragments_connected  split_excess  components  cross"
    )
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        kept_total = 0
        kept_same = 0
        kept_cross = 0
        fragment_total = 0
        fragment_connected = 0
        split_excess = 0
        component_total = 0

        for group in local_groups:
            local_oracle = np.asarray(group["oracle"], dtype=np.int64)
            kept_edges = [
                (int(first), int(second))
                for first, second, score, _ in group["edges"]
                if float(score) >= threshold
            ]
            kept_total += len(kept_edges)
            for first, second, score, same in group["edges"]:
                if float(score) >= threshold:
                    if bool(same):
                        kept_same += 1
                    else:
                        kept_cross += 1

            all_indices = np.arange(len(local_oracle), dtype=np.int64)
            component_total += _component_count(all_indices, kept_edges)
            for oracle_value in np.unique(local_oracle):
                fragment_indices = np.flatnonzero(local_oracle == oracle_value)
                fragment_edges = [
                    (first, second)
                    for first, second in kept_edges
                    if local_oracle[first] == oracle_value
                    and local_oracle[second] == oracle_value
                ]
                components = _component_count(fragment_indices, fragment_edges)
                fragment_total += 1
                fragment_connected += int(components == 1)
                split_excess += max(components - 1, 0)

        purity = kept_same / max(kept_total, 1)
        recall = kept_same / max(len(same_scores), 1)
        connected_fraction = fragment_connected / max(fragment_total, 1)
        rows.append(
            {
                "threshold": threshold,
                "kept": kept_total,
                "purity": purity,
                "recall": recall,
                "connected": connected_fraction,
                "split": split_excess,
                "components": component_total,
                "cross": kept_cross,
            }
        )
        print(
            f" {threshold:>9.7f} {kept_total:>5d} {purity:>7.4f} {recall:>12.4f} "
            f"{connected_fraction:>20.4f} {split_excess:>13d} "
            f"{component_total:>11d} {kept_cross:>6d}"
        )

    feasible = [row for row in rows if float(row["connected"]) >= 0.90]
    print("\nBest edge purity while retaining at least 90% fragment connectivity:")
    if feasible:
        best = max(feasible, key=lambda row: (float(row["purity"]), -int(row["cross"])))
        print(
            f"threshold={float(best['threshold']):.7f} purity={float(best['purity']):.5f} "
            f"same_recall={float(best['recall']):.5f} "
            f"fragments_connected={float(best['connected']):.5f} "
            f"split_excess={int(best['split'])} cross={int(best['cross'])}"
        )
    else:
        print("none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
