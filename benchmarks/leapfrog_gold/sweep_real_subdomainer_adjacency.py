"""Sweep plausible real-location SubDomainer adjacency graphs against Leapfrog labels.

The recovered runtime shows that Leapfrog first assigns real locations to coarse
GridSeededDomainer domains, then constructs one ``SubDomainer`` per populated
coarse domain with ``bbox=None``, the same consistency threshold, and point-count
limits computed from that coarse-domain population.  What remains unknown is how
those irregular real locations are connected.  This script tests several
reasonable graph constructions without changing the production builder.
"""
from __future__ import annotations

import argparse
import os
import sys
from heapq import heappop, heappush
from itertools import combinations
from math import floor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing this module patches the strict coordinate/label loader in comparison.
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _unique_edges(edges: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    array.sort(axis=1)
    array = array[array[:, 0] != array[:, 1]]
    if len(array) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(array, axis=0)


def _delaunay_edges(points: np.ndarray) -> np.ndarray:
    count = len(points)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64)
    if count <= 4:
        return _unique_edges(list(combinations(range(count), 2)))
    from scipy.spatial import Delaunay, QhullError

    try:
        simplices = Delaunay(points, qhull_options="QJ").simplices
    except QhullError:
        return _knn_edges(points, 6, mutual=False)
    edges: list[tuple[int, int]] = []
    for simplex in simplices:
        edges.extend(combinations((int(value) for value in simplex), 2))
    return _unique_edges(edges)


def _knn_edges(points: np.ndarray, k: int, *, mutual: bool) -> np.ndarray:
    count = len(points)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64)
    from scipy.spatial import cKDTree

    actual_k = min(max(int(k), 1), count - 1)
    neighbours = np.asarray(
        cKDTree(points).query(points, k=actual_k + 1)[1], dtype=np.int64
    )[:, 1:]
    neighbour_sets = [set(int(value) for value in row) for row in neighbours]
    edges: list[tuple[int, int]] = []
    for first, row in enumerate(neighbours):
        for second_value in row:
            second = int(second_value)
            if mutual and first not in neighbour_sets[second]:
                continue
            edges.append((first, second))
    return _unique_edges(edges)


def _radius_edges(points: np.ndarray, factor: float) -> np.ndarray:
    count = len(points)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, nearest = tree.query(points, k=2)
    positive = distances[:, 1][distances[:, 1] > 0.0]
    if len(positive) == 0:
        return _knn_edges(points, 1, mutual=False)
    radius = float(np.median(positive) * factor)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    edges = [tuple(int(value) for value in pair) for pair in np.asarray(pairs)]

    # Keep every point connected to at least its nearest distinct location.  This
    # avoids graph artefacts caused only by a locally sparse sampling pattern.
    degree = np.zeros(count, dtype=np.int64)
    for first, second in edges:
        degree[first] += 1
        degree[second] += 1
    for index in np.flatnonzero(degree == 0):
        edges.append((int(index), int(nearest[index, 1])))
    return _unique_edges(edges)


def _graph_edges(points: np.ndarray, name: str, value: float | int | None) -> np.ndarray:
    if name == "delaunay":
        return _delaunay_edges(points)
    if name == "knn":
        return _knn_edges(points, int(value), mutual=False)
    if name == "mutual-knn":
        return _knn_edges(points, int(value), mutual=True)
    if name == "radius":
        return _radius_edges(points, float(value))
    raise ValueError(f"unknown graph: {name}")


def _region_grow(
    anisotropies: np.ndarray,
    edges: np.ndarray,
    *,
    threshold: float,
    maximum_points: int,
) -> tuple[np.ndarray, int]:
    """Return component labels using the corrected Leapfrog-style greedy grower."""
    import polatory

    count = len(anisotropies)
    if count <= 1:
        return np.zeros(count, dtype=np.int64), 0

    normalise = polatory.leapfrog_automatic_domain_builder_fixed._normalise_determinant
    merged_score = (
        polatory.leapfrog_automatic_domain_builder_fixed._merged_matrix_and_consistency
    )

    capacity = max(2 * count + 1, 3)
    active = np.zeros(capacity, dtype=bool)
    active[:count] = True
    version = np.zeros(capacity, dtype=np.int64)
    sizes = np.zeros(capacity, dtype=np.int64)
    sizes[:count] = 1
    matrices = np.zeros((capacity, 3, 3), dtype=float)
    matrices[:count] = np.asarray([normalise(matrix) for matrix in anisotropies])
    leaves: list[list[int]] = [[] for _ in range(capacity)]
    for index in range(count):
        leaves[index] = [index]
    neighbours: list[set[int]] = [set() for _ in range(capacity)]
    for first_value, second_value in edges:
        first, second = int(first_value), int(second_value)
        neighbours[first].add(second)
        neighbours[second].add(first)

    heap: list[tuple[float, int, int, int, int]] = []

    def push(first: int, second: int) -> None:
        if not active[first] or not active[second] or first == second:
            return
        if int(sizes[first] + sizes[second]) > maximum_points:
            return
        _, consistency = merged_score(
            matrices[first], int(sizes[first]), matrices[second], int(sizes[second])
        )
        if not np.isfinite(consistency):
            return
        low, high = sorted((first, second))
        heappush(
            heap,
            (-float(consistency), low, high, int(version[low]), int(version[high])),
        )

    for first, second in edges:
        push(int(first), int(second))

    next_id = count
    merges = 0
    while heap:
        negative, first, second, first_version, second_version = heappop(heap)
        if not active[first] or not active[second]:
            continue
        if version[first] != first_version or version[second] != second_version:
            continue
        if second not in neighbours[first] or first not in neighbours[second]:
            continue
        consistency = -negative
        if consistency < threshold:
            break
        if int(sizes[first] + sizes[second]) > maximum_points:
            continue

        merged_matrix, _ = merged_score(
            matrices[first], int(sizes[first]), matrices[second], int(sizes[second])
        )
        merged_neighbours = (neighbours[first] | neighbours[second]) - {first, second}
        active[first] = False
        active[second] = False
        version[first] += 1
        version[second] += 1

        active[next_id] = True
        sizes[next_id] = sizes[first] + sizes[second]
        matrices[next_id] = merged_matrix
        leaves[next_id] = leaves[first] + leaves[second]
        for neighbour in sorted(merged_neighbours):
            if not active[neighbour]:
                continue
            neighbours[neighbour].discard(first)
            neighbours[neighbour].discard(second)
            neighbours[neighbour].add(next_id)
            neighbours[next_id].add(neighbour)
        for neighbour in sorted(neighbours[next_id]):
            push(next_id, neighbour)
        next_id += 1
        merges += 1

    labels = np.empty(count, dtype=np.int64)
    label = 0
    for domain_id in range(next_id):
        if not active[domain_id]:
            continue
        labels[np.asarray(leaves[domain_id], dtype=np.int64)] = label
        label += 1
    return labels, merges


def _second_stage_labels(
    points: np.ndarray,
    anisotropies: np.ndarray,
    coarse_labels: np.ndarray,
    *,
    graph_name: str,
    graph_value: float | int | None,
    threshold: float,
    maximum_fraction: float,
) -> tuple[np.ndarray, int, int]:
    labels = np.empty(len(points), dtype=np.int64)
    next_label = 0
    total_edges = 0
    total_merges = 0

    for coarse_label in np.unique(coarse_labels):
        indices = np.flatnonzero(coarse_labels == coarse_label)
        local_points = points[indices]
        local_anisotropies = anisotropies[indices]
        maximum_points = max(1, int(floor(maximum_fraction * len(indices))))
        edges = _graph_edges(local_points, graph_name, graph_value)
        local_labels, merges = _region_grow(
            local_anisotropies,
            edges,
            threshold=threshold,
            maximum_points=maximum_points,
        )
        labels[indices] = local_labels + next_label
        next_label += int(local_labels.max()) + 1 if len(local_labels) else 0
        total_edges += len(edges)
        total_merges += merges

    return labels, total_edges, total_merges


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
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    point_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        points, trend_input, non_decaying=False
    )
    coarse_labels, _, coarse_minimum, coarse_maximum, coarse_merges = (
        builder._automatic_labels(
            points,
            np.asarray(centroid_anisotropies, dtype=np.float64),
            np.asarray(minimum, dtype=np.float64),
            np.asarray(maximum, dtype=np.float64),
            tuple(int(value) for value in shape),
        )
    )

    variants: list[tuple[str, float | int | None]] = [("delaunay", None)]
    variants.extend(("knn", k) for k in (2, 4, 6, 8, 12, 16))
    variants.extend(("mutual-knn", k) for k in (4, 6, 8, 12))
    variants.extend(("radius", factor) for factor in (1.5, 2.0, 3.0))

    def metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(accuracy),
            int(matched),
        )

    coarse_ari, coarse_ami, coarse_match, coarse_matched = metrics(coarse_labels)
    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))}"
    )
    print(
        f"coarse: domains={len(np.unique(coarse_labels))} min={coarse_minimum} "
        f"max={coarse_maximum} merges={coarse_merges} ARI={coarse_ari:.5f} "
        f"AMI={coarse_ami:.5f} match={coarse_match:.5f} "
        f"({coarse_matched}/{len(points)})"
    )
    print()

    rows: list[tuple[float, float, float, str, int, int, int]] = []
    for graph_name, graph_value in variants:
        labels, edge_count, merge_count = _second_stage_labels(
            points,
            np.asarray(point_anisotropies, dtype=np.float64),
            np.asarray(coarse_labels, dtype=np.int64),
            graph_name=graph_name,
            graph_value=graph_value,
            threshold=args.threshold,
            maximum_fraction=args.maximum_fraction,
        )
        ari, ami, match, matched = metrics(labels)
        display = graph_name if graph_value is None else f"{graph_name}:{graph_value}"
        rows.append(
            (
                ari,
                ami,
                match,
                display,
                len(np.unique(labels)),
                edge_count,
                merge_count,
            )
        )
        print(
            f"{display:<16} domains={len(np.unique(labels)):>4d} "
            f"edges={edge_count:>7d} merges={merge_count:>4d} "
            f"ARI={ari:.5f} AMI={ami:.5f} match={match:.5f} "
            f"({matched}/{len(points)})"
        )

    print("\nRanked by ARI:")
    for ari, ami, match, display, domains, edge_count, merge_count in sorted(
        rows, reverse=True
    ):
        print(
            f"{display:<16} domains={domains:>4d} edges={edge_count:>7d} "
            f"merges={merge_count:>4d} ARI={ari:.5f} AMI={ami:.5f} "
            f"match={match:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
