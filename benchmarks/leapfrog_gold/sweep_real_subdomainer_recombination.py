"""Test whether Leapfrog recombines local real-location subdomains globally.

The recovered runtime shows a coarse GridSeededDomainer followed by one
SubDomainer per populated coarse domain.  The previous adjacency sweep proved
that treating those local SubDomainer outputs as final domains over-fragments
the data.  This diagnostic therefore creates local Delaunay subdomains using
the recovered 10%% local cap, then tests several plausible global recombination
graphs and point-count limits.
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

import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_adjacency as adjacency  # noqa: E402


def _parent_adjacency(centroid_labels: np.ndarray, shape: tuple[int, int, int]) -> set[tuple[int, int]]:
    import polatory

    edges = polatory.leapfrog_automatic_domain_builder_fixed._grid_edges(shape)
    pairs: set[tuple[int, int]] = set()
    for first, second in edges:
        left = int(centroid_labels[int(first)])
        right = int(centroid_labels[int(second)])
        if left < 0 or right < 0 or left == right:
            continue
        pairs.add(tuple(sorted((left, right))))
    return pairs


def _build_local_components(
    points: np.ndarray,
    anisotropies: np.ndarray,
    coarse_labels: np.ndarray,
    *,
    threshold: float,
    maximum_fraction: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, int]:
    import polatory

    normalise = polatory.leapfrog_automatic_domain_builder_fixed._normalise_determinant
    members: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    sizes: list[int] = []
    centroids: list[np.ndarray] = []
    parents: list[int] = []
    total_edges = 0
    total_merges = 0

    for parent in np.unique(coarse_labels):
        indices = np.flatnonzero(coarse_labels == parent)
        local_points = points[indices]
        local_anisotropies = anisotropies[indices]
        maximum_points = max(1, int(floor(maximum_fraction * len(indices))))
        edges = adjacency._delaunay_edges(local_points)
        local_labels, merges = adjacency._region_grow(
            local_anisotropies,
            edges,
            threshold=threshold,
            maximum_points=maximum_points,
        )
        total_edges += len(edges)
        total_merges += merges
        for label in np.unique(local_labels):
            local_members = np.flatnonzero(local_labels == label)
            global_members = indices[local_members]
            normalised = np.asarray(
                [normalise(matrix) for matrix in anisotropies[global_members]], dtype=float
            )
            members.append(global_members)
            matrices.append(normalised.mean(axis=0))
            sizes.append(len(global_members))
            centroids.append(points[global_members].mean(axis=0))
            parents.append(int(parent))

    return (
        members,
        np.asarray(matrices, dtype=float),
        np.asarray(sizes, dtype=np.int64),
        np.asarray(centroids, dtype=float),
        np.asarray(parents, dtype=np.int64),
        total_edges,
        total_merges,
    )


def _global_edges(
    centroids: np.ndarray,
    parents: np.ndarray,
    parent_pairs: set[tuple[int, int]],
    mode: str,
    value: int | None,
) -> np.ndarray:
    count = len(centroids)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64)
    if mode == "complete":
        return adjacency._unique_edges(list(combinations(range(count), 2)))
    if mode == "knn":
        return adjacency._knn_edges(centroids, int(value), mutual=False)
    if mode == "mutual-knn":
        return adjacency._knn_edges(centroids, int(value), mutual=True)
    if mode == "parent-adjacent":
        edges: list[tuple[int, int]] = []
        for first, second in combinations(range(count), 2):
            left = int(parents[first])
            right = int(parents[second])
            if left == right or tuple(sorted((left, right))) in parent_pairs:
                edges.append((first, second))
        return adjacency._unique_edges(edges)
    raise ValueError(mode)


def _merge_components(
    point_count: int,
    members: list[np.ndarray],
    matrices: np.ndarray,
    sizes: np.ndarray,
    edges: np.ndarray,
    *,
    threshold: float,
    maximum_points: int,
) -> tuple[np.ndarray, int]:
    import polatory

    merged_score = (
        polatory.leapfrog_automatic_domain_builder_fixed._merged_matrix_and_consistency
    )
    count = len(members)
    if count == 0:
        raise ValueError("no local components")

    capacity = max(2 * count + 1, 3)
    active = np.zeros(capacity, dtype=bool)
    active[:count] = True
    version = np.zeros(capacity, dtype=np.int64)
    domain_sizes = np.zeros(capacity, dtype=np.int64)
    domain_sizes[:count] = sizes
    domain_matrices = np.zeros((capacity, 3, 3), dtype=float)
    domain_matrices[:count] = matrices
    domain_members: list[list[int]] = [[] for _ in range(capacity)]
    for index, values in enumerate(members):
        domain_members[index] = [int(value) for value in values]
    neighbours: list[set[int]] = [set() for _ in range(capacity)]
    for first_value, second_value in edges:
        first, second = int(first_value), int(second_value)
        neighbours[first].add(second)
        neighbours[second].add(first)

    heap: list[tuple[float, int, int, int, int]] = []

    def push(first: int, second: int) -> None:
        if not active[first] or not active[second] or first == second:
            return
        if int(domain_sizes[first] + domain_sizes[second]) > maximum_points:
            return
        _, consistency = merged_score(
            domain_matrices[first],
            int(domain_sizes[first]),
            domain_matrices[second],
            int(domain_sizes[second]),
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
        if int(domain_sizes[first] + domain_sizes[second]) > maximum_points:
            continue

        merged_matrix, _ = merged_score(
            domain_matrices[first],
            int(domain_sizes[first]),
            domain_matrices[second],
            int(domain_sizes[second]),
        )
        merged_neighbours = (neighbours[first] | neighbours[second]) - {first, second}
        active[first] = False
        active[second] = False
        version[first] += 1
        version[second] += 1

        active[next_id] = True
        domain_sizes[next_id] = domain_sizes[first] + domain_sizes[second]
        domain_matrices[next_id] = merged_matrix
        domain_members[next_id] = domain_members[first] + domain_members[second]
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

    labels = np.empty(point_count, dtype=np.int64)
    label = 0
    for domain_id in range(next_id):
        if not active[domain_id]:
            continue
        labels[np.asarray(domain_members[domain_id], dtype=np.int64)] = label
        label += 1
    return labels, merges


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
    coarse_labels, centroid_labels, _, _, _ = builder._automatic_labels(
        points,
        np.asarray(centroid_anisotropies, dtype=np.float64),
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        tuple(int(value) for value in shape),
    )

    (
        members,
        component_matrices,
        component_sizes,
        component_centroids,
        component_parents,
        local_edges,
        local_merges,
    ) = _build_local_components(
        points,
        np.asarray(point_anisotropies, dtype=np.float64),
        np.asarray(coarse_labels, dtype=np.int64),
        threshold=args.threshold,
        maximum_fraction=args.maximum_fraction,
    )
    parent_pairs = _parent_adjacency(
        np.asarray(centroid_labels, dtype=np.int64),
        tuple(int(value) for value in shape),
    )

    def metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(accuracy),
            int(matched),
        )

    local_labels = np.empty(len(points), dtype=np.int64)
    for label, values in enumerate(members):
        local_labels[values] = label
    local_ari, local_ami, local_match, local_matched = metrics(local_labels)
    print(
        f"case={case_name} points={len(points)} coarse_domains={len(np.unique(coarse_labels))} "
        f"local_components={len(members)} local_edges={local_edges} local_merges={local_merges}"
    )
    print(
        f"local-only: ARI={local_ari:.5f} AMI={local_ami:.5f} "
        f"match={local_match:.5f} ({local_matched}/{len(points)})"
    )
    print()

    graph_variants: list[tuple[str, int | None]] = [("complete", None), ("parent-adjacent", None)]
    graph_variants.extend(("knn", k) for k in (2, 4, 6, 8, 12, 16))
    graph_variants.extend(("mutual-knn", k) for k in (4, 8, 12))
    caps = [
        ("none", len(points)),
        ("10pct", max(1, int(floor(0.10 * len(points))))),
        ("20pct", max(1, int(floor(0.20 * len(points))))),
    ]

    rows: list[tuple[float, str]] = []
    for graph_name, graph_value in graph_variants:
        edges = _global_edges(
            component_centroids,
            component_parents,
            parent_pairs,
            graph_name,
            graph_value,
        )
        graph_label = graph_name if graph_value is None else f"{graph_name}:{graph_value}"
        for cap_name, maximum_points in caps:
            labels, merges = _merge_components(
                len(points),
                members,
                component_matrices,
                component_sizes,
                edges,
                threshold=args.threshold,
                maximum_points=maximum_points,
            )
            ari, ami, match, matched = metrics(labels)
            text = (
                f"{graph_label:18s} cap={cap_name:5s} domains={len(np.unique(labels)):4d} "
                f"edges={len(edges):6d} merges={merges:4d} ARI={ari:.5f} "
                f"AMI={ami:.5f} match={match:.5f} ({matched}/{len(points)})"
            )
            print(text)
            rows.append((ari, text))

    print("\nRanked by ARI:")
    for _, text in sorted(rows, key=lambda item: item[0], reverse=True):
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
