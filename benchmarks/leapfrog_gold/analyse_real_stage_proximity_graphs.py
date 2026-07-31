"""Compare sparse proximity graphs for Leapfrog's real-location SubDomainer stage.

The within-coarse-domain Delaunay graph preserves decoded oracle fragments well but
contains many cross-fragment edges, while fixed-k nearest-neighbour graphs fragment
valid domains. This diagnostic evaluates two classical Delaunay subgraphs (Gabriel
and relative-neighbourhood graphs) plus locally scaled Delaunay edge filters. It asks
whether a sparser geometry-only graph can reduce cross-fragment candidate edges while
retaining at least 90 percent decoded-fragment connectivity. Production code is unchanged.
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

    count = len(points)
    if count < 2:
        return []
    if count == 2:
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


def _gabriel_edges(points: np.ndarray, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    tolerance = 1.0e-12
    for first, second in edges:
        midpoint = 0.5 * (points[first] + points[second])
        radius_sq = 0.25 * float(np.sum((points[first] - points[second]) ** 2))
        distances_sq = np.sum((points - midpoint) ** 2, axis=1)
        mask = np.ones(len(points), dtype=bool)
        mask[[first, second]] = False
        if not np.any(distances_sq[mask] < radius_sq - tolerance):
            kept.append((first, second))
    return kept


def _rng_edges(points: np.ndarray, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    tolerance = 1.0e-12
    for first, second in edges:
        length_sq = float(np.sum((points[first] - points[second]) ** 2))
        first_sq = np.sum((points - points[first]) ** 2, axis=1)
        second_sq = np.sum((points - points[second]) ** 2, axis=1)
        mask = np.ones(len(points), dtype=bool)
        mask[[first, second]] = False
        lune_inside = np.maximum(first_sq[mask], second_sq[mask]) < length_sq - tolerance
        if not np.any(lune_inside):
            kept.append((first, second))
    return kept


def _local_scales(points: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    count = len(points)
    if count <= 1:
        return np.ones(count, dtype=float)
    actual_k = min(max(int(k), 1), count - 1)
    distances = np.asarray(cKDTree(points).query(points, k=actual_k + 1)[0], dtype=float)
    scales = distances[:, actual_k]
    positive = scales[scales > 0.0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    return np.where(scales > 0.0, scales, fallback)


def _scaled_edges(
    points: np.ndarray,
    edges: list[tuple[int, int]],
    scales: np.ndarray,
    multiplier: float,
    mode: str,
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    for first, second in edges:
        length = float(np.linalg.norm(points[first] - points[second]))
        if mode == "max":
            reference = max(float(scales[first]), float(scales[second]))
        elif mode == "mean":
            reference = 0.5 * (float(scales[first]) + float(scales[second]))
        else:
            reference = float(np.sqrt(float(scales[first]) * float(scales[second])))
        if length <= multiplier * max(reference, np.finfo(float).eps):
            kept.append((first, second))
    return kept


def _component_count(indices: np.ndarray, edges: list[tuple[int, int]]) -> int:
    if len(indices) == 0:
        return 0
    mapping = {int(value): position for position, value in enumerate(indices)}
    uf = UnionFind(len(indices))
    for first, second in edges:
        if first in mapping and second in mapping:
            uf.union(mapping[first], mapping[second])
    return len({uf.find(index) for index in range(len(indices))})


def _evaluate(groups: list[dict[str, object]], graph_name: str) -> dict[str, float | int | str]:
    edge_total = 0
    same_total = 0
    cross_total = 0
    fragment_total = 0
    fragment_connected = 0
    split_excess = 0
    largest_weighted = 0.0
    point_weight = 0
    component_total = 0

    for group in groups:
        oracle = np.asarray(group["oracle"], dtype=np.int64)
        edges = list(group[graph_name])
        edge_total += len(edges)
        for first, second in edges:
            if oracle[first] == oracle[second]:
                same_total += 1
            else:
                cross_total += 1
        all_indices = np.arange(len(oracle), dtype=np.int64)
        component_total += _component_count(all_indices, edges)
        for oracle_value in np.unique(oracle):
            indices = np.flatnonzero(oracle == oracle_value)
            fragment_edges = [
                (first, second)
                for first, second in edges
                if oracle[first] == oracle_value and oracle[second] == oracle_value
            ]
            components = _component_count(indices, fragment_edges)
            fragment_total += 1
            fragment_connected += int(components == 1)
            split_excess += max(components - 1, 0)
            if len(indices):
                mapping = {int(value): position for position, value in enumerate(indices)}
                uf = UnionFind(len(indices))
                for first, second in fragment_edges:
                    uf.union(mapping[first], mapping[second])
                counts: dict[int, int] = {}
                for position in range(len(indices)):
                    root = uf.find(position)
                    counts[root] = counts.get(root, 0) + 1
                largest = max(counts.values(), default=0)
                largest_weighted += largest
                point_weight += len(indices)

    return {
        "graph": graph_name,
        "edges": edge_total,
        "purity": same_total / max(edge_total, 1),
        "same_recall": same_total,
        "cross": cross_total,
        "connected": fragment_connected / max(fragment_total, 1),
        "largest": largest_weighted / max(point_weight, 1),
        "split": split_excess,
        "components": component_total,
    }


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
    parser.add_argument("--scale-k", type=int, default=6)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory

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
    coarse_labels, _, _, _, coarse_merges = builder._automatic_labels(
        points,
        np.asarray(centroid_matrices, dtype=np.float64),
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        tuple(int(value) for value in shape),
    )
    coarse_labels = np.asarray(coarse_labels, dtype=np.int64)

    multipliers = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
    graph_names = ["delaunay", "gabriel", "rng"]
    for mode in ("geom", "mean", "max"):
        for multiplier in multipliers:
            graph_names.append(f"scaled_{mode}_{multiplier:g}")

    groups: list[dict[str, object]] = []
    for coarse_value in np.unique(coarse_labels):
        indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[indices]
        local_oracle = oracle_labels[indices]
        delaunay = _delaunay_edges(local_points)
        scales = _local_scales(local_points, args.scale_k)
        group: dict[str, object] = {
            "coarse": int(coarse_value),
            "oracle": local_oracle,
            "delaunay": delaunay,
            "gabriel": _gabriel_edges(local_points, delaunay),
            "rng": _rng_edges(local_points, delaunay),
        }
        for mode in ("geom", "mean", "max"):
            for multiplier in multipliers:
                group[f"scaled_{mode}_{multiplier:g}"] = _scaled_edges(
                    local_points, delaunay, scales, multiplier, mode
                )
        groups.append(group)

    rows = [_evaluate(groups, name) for name in graph_names]
    baseline_same = int(next(row for row in rows if row["graph"] == "delaunay")["same_recall"])
    for row in rows:
        row["same_recall"] = int(row["same_recall"]) / max(baseline_same, 1)

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    print(f"local scale: k={args.scale_k}")
    print("\nGraph comparison:")
    print(" graph                    edges purity same_recall connected largest split components cross")
    for row in sorted(rows, key=lambda value: (float(value["connected"]), float(value["purity"])), reverse=True):
        print(
            f" {str(row['graph']):<24s} {int(row['edges']):>5d} "
            f"{float(row['purity']):>6.3f} {float(row['same_recall']):>11.3f} "
            f"{float(row['connected']):>9.3f} {float(row['largest']):>7.3f} "
            f"{int(row['split']):>5d} {int(row['components']):>10d} {int(row['cross']):>5d}"
        )

    feasible = [row for row in rows if float(row["connected"]) >= 0.90]
    print("\nBest purity while retaining at least 90% fragment connectivity:")
    if feasible:
        best = max(feasible, key=lambda row: (float(row["purity"]), -int(row["cross"])))
        print(
            f"graph={best['graph']} purity={float(best['purity']):.5f} "
            f"same_recall={float(best['same_recall']):.5f} "
            f"fragments_connected={float(best['connected']):.5f} "
            f"largest_fraction={float(best['largest']):.5f} "
            f"split_excess={int(best['split'])} cross={int(best['cross'])}"
        )
    else:
        print("none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
