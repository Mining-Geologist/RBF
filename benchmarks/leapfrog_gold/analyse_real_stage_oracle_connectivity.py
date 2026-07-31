"""Diagnose the real-location SubDomainer graph inside each coarse domain.

Recovered runtime evidence shows Leapfrog constructs a second SubDomainer for real
locations grouped by a preceding coarse-domain id.  This diagnostic therefore avoids
post-hoc global merge tuning and asks a more direct question: for each candidate local
point graph, are the decoded oracle fragments inside every coarse domain connected,
and how many graph edges cross between different oracle fragments?

A plausible graph should make same-oracle fragments internally connected while
presenting relatively few cross-oracle candidate edges.  Production code is unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _unique_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    values = np.asarray([tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]], dtype=np.int64)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(values, axis=0)


def _delaunay_edges(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay, QhullError

    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    if len(points) == 2:
        return np.asarray([[0, 1]], dtype=np.int64)
    try:
        simplices = Delaunay(points, qhull_options="QJ").simplices
    except QhullError:
        return np.empty((0, 2), dtype=np.int64)
    edges: list[tuple[int, int]] = []
    for simplex in simplices:
        for first, second in combinations((int(value) for value in simplex), 2):
            edges.append((first, second))
    return _unique_edges(edges)


def _knn_edges(points: np.ndarray, k: int, mutual: bool) -> np.ndarray:
    from scipy.spatial import cKDTree

    count = len(points)
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)
    actual_k = min(max(int(k), 1), count - 1)
    neighbours = np.asarray(cKDTree(points).query(points, k=actual_k + 1)[1])[:, 1:]
    directed = {(int(index), int(other)) for index, row in enumerate(neighbours) for other in row}
    edges: list[tuple[int, int]] = []
    for first, second in directed:
        if mutual and (second, first) not in directed:
            continue
        edges.append((first, second))
    return _unique_edges(edges)


def _component_sizes(indices: np.ndarray, edges: np.ndarray) -> list[int]:
    if len(indices) == 0:
        return []
    allowed = set(int(value) for value in indices)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in edges:
        first_i, second_i = int(first), int(second)
        if first_i in allowed and second_i in allowed:
            adjacency[first_i].append(second_i)
            adjacency[second_i].append(first_i)
    remaining = set(allowed)
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        queue: deque[int] = deque([start])
        size = 1
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, []):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    size += 1
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _evaluate_graph(oracle: np.ndarray, edges: np.ndarray) -> dict[str, float | int]:
    same_edges = 0
    cross_edges = 0
    for first, second in edges:
        if int(oracle[int(first)]) == int(oracle[int(second)]):
            same_edges += 1
        else:
            cross_edges += 1

    fragments = 0
    fully_connected = 0
    split_excess = 0
    weighted_largest = 0
    total_fragment_points = 0
    singleton_fragments = 0
    for label in np.unique(oracle):
        indices = np.flatnonzero(oracle == label)
        fragments += 1
        total_fragment_points += len(indices)
        if len(indices) == 1:
            singleton_fragments += 1
        component_sizes = _component_sizes(indices, edges)
        if len(component_sizes) == 1:
            fully_connected += 1
        split_excess += max(len(component_sizes) - 1, 0)
        if component_sizes:
            weighted_largest += component_sizes[0]

    edge_count = len(edges)
    return {
        "edges": edge_count,
        "same_edges": same_edges,
        "cross_edges": cross_edges,
        "edge_purity": same_edges / edge_count if edge_count else 0.0,
        "fragments": fragments,
        "fully_connected": fully_connected,
        "fully_connected_fraction": fully_connected / fragments if fragments else 1.0,
        "split_excess": split_excess,
        "largest_component_fraction": weighted_largest / total_fragment_points if total_fragment_points else 1.0,
        "singleton_fragments": singleton_fragments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument("--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(args.decoded_root, case_name)
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

    graph_builders = {
        "delaunay": lambda local_points: _delaunay_edges(local_points),
        "knn4": lambda local_points: _knn_edges(local_points, 4, False),
        "knn6": lambda local_points: _knn_edges(local_points, 6, False),
        "knn8": lambda local_points: _knn_edges(local_points, 8, False),
        "knn12": lambda local_points: _knn_edges(local_points, 12, False),
        "mutual4": lambda local_points: _knn_edges(local_points, 4, True),
        "mutual6": lambda local_points: _knn_edges(local_points, 6, True),
        "mutual8": lambda local_points: _knn_edges(local_points, 8, True),
        "mutual12": lambda local_points: _knn_edges(local_points, 12, True),
    }

    aggregate: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in graph_builders
    }
    coarse_rows: list[dict[str, object]] = []

    for coarse_value in np.unique(coarse_labels):
        global_indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[global_indices]
        local_oracle = oracle_labels[global_indices]
        oracle_values, oracle_counts = np.unique(local_oracle, return_counts=True)
        purity = float(oracle_counts.max() / len(global_indices))
        row: dict[str, object] = {
            "coarse": int(coarse_value),
            "size": int(len(global_indices)),
            "oracle_fragments": int(len(oracle_values)),
            "purity": purity,
            "oracle_counts": tuple(int(value) for value in sorted(oracle_counts, reverse=True)),
        }
        for name, build_graph in graph_builders.items():
            edges = build_graph(local_points)
            metrics = _evaluate_graph(local_oracle, edges)
            row[name] = metrics
            weights = aggregate[name]
            weights["edges"] += float(metrics["edges"])
            weights["same_edges"] += float(metrics["same_edges"])
            weights["cross_edges"] += float(metrics["cross_edges"])
            weights["fragments"] += float(metrics["fragments"])
            weights["fully_connected"] += float(metrics["fully_connected"])
            weights["split_excess"] += float(metrics["split_excess"])
            weights["largest_numerator"] += float(metrics["largest_component_fraction"]) * len(global_indices)
            weights["points"] += len(global_indices)
        coarse_rows.append(row)

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} coarse_domains={len(np.unique(coarse_labels))} "
        f"coarse_merges={coarse_merges}"
    )
    print("\nAggregate candidate-graph diagnostics inside coarse domains:")
    ranking_rows: list[tuple[str, float, float, float, int, int, int]] = []
    for name, values in aggregate.items():
        edges = int(values["edges"])
        same = int(values["same_edges"])
        cross = int(values["cross_edges"])
        fragments = int(values["fragments"])
        connected = int(values["fully_connected"])
        split_excess = int(values["split_excess"])
        purity = same / edges if edges else 0.0
        connected_fraction = connected / fragments if fragments else 1.0
        largest_fraction = values["largest_numerator"] / values["points"] if values["points"] else 1.0
        ranking_rows.append((name, connected_fraction, largest_fraction, purity, split_excess, cross, edges))
    for name, connected_fraction, largest_fraction, purity, split_excess, cross, edges in sorted(
        ranking_rows,
        key=lambda item: (item[1], item[2], item[3], -item[4], -item[5]),
        reverse=True,
    ):
        print(
            f"{name:>9s}: fragments_connected={connected_fraction:.4f} "
            f"largest_fraction={largest_fraction:.4f} edge_purity={purity:.4f} "
            f"split_excess={split_excess:3d} cross_edges={cross:5d} edges={edges:5d}"
        )

    print(f"\nTop {max(args.top, 1)} mixed coarse domains by lowest oracle purity:")
    for row in sorted(coarse_rows, key=lambda item: (float(item["purity"]), -int(item["size"])))[: max(args.top, 1)]:
        print(
            f"coarse={int(row['coarse']):2d} size={int(row['size']):3d} "
            f"oracle_fragments={int(row['oracle_fragments']):2d} purity={float(row['purity']):.4f} "
            f"counts={row['oracle_counts']}"
        )
        for name in graph_builders:
            metrics = row[name]
            print(
                f"  {name:>9s}: connected={int(metrics['fully_connected'])}/{int(metrics['fragments'])} "
                f"largest={float(metrics['largest_component_fraction']):.4f} "
                f"purity={float(metrics['edge_purity']):.4f} split={int(metrics['split_excess']):2d} "
                f"cross={int(metrics['cross_edges']):4d} edges={int(metrics['edges']):4d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
