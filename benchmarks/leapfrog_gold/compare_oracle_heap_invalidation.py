"""Compare the current heap invalidation with a corrected lazy-heap update.

The reconstructed builder increments the version of every neighbouring domain when two
other domains merge. That invalidates the neighbour's still-valid heap entries to all of
its unchanged neighbours, but only the new merged-domain edge is pushed again. The heap
therefore silently loses valid adjacency candidates and region growing can stop early.

This diagnostic reproduces the current behaviour, verifies it matches the installed
builder, then reruns the same exact WolfPass oracle comparison while preserving unaffected
heap entries. No RBF fitting or surface meshing is performed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing the robust wrapper patches the oracle CSV loader used by comparison.
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402


@dataclass(frozen=True)
class HeapComparison:
    mode: str
    oracle_domains: int
    predicted_domains: int
    surviving_grid_domains: int
    merge_count: int
    stale_heap_pops: int
    adjusted_rand_index: float
    adjusted_mutual_information: float
    optimal_label_accuracy: float
    matched_points: int
    elapsed_seconds: float


def _automatic_labels_with_heap_mode(
    *,
    builder_module: object,
    points: np.ndarray,
    centroid_anisotropies: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
    minimum_fraction: float,
    maximum_fraction: float,
    threshold: float,
    invalidate_unchanged_neighbours: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Run the recovered merge loop with either current or corrected invalidation."""
    point_cell_indices = getattr(builder_module, "_point_cell_indices")
    normalise = getattr(builder_module, "_normalise_determinant")
    grid_edges = getattr(builder_module, "_grid_edges")
    merge_matrix = getattr(builder_module, "_merged_matrix_and_consistency")

    total = int(len(centroid_anisotropies))
    point_cells = point_cell_indices(points, minimum, maximum, shape)
    minimum_points = max(1, int(np.floor(minimum_fraction * total)))
    maximum_points = max(minimum_points, int(np.floor(maximum_fraction * total)))

    capacity = 2 * total + 1
    active = np.zeros(capacity, dtype=bool)
    active[:total] = True
    version = np.zeros(capacity, dtype=np.int64)
    sizes = np.zeros(capacity, dtype=np.int64)
    sizes[:total] = 1
    matrices = np.zeros((capacity, 3, 3), dtype=np.float64)
    matrices[:total] = np.asarray(
        [normalise(matrix) for matrix in centroid_anisotropies], dtype=np.float64
    )
    leaves: list[list[int]] = [[index] for index in range(total)] + [
        [] for _ in range(total + 1)
    ]
    neighbours: list[set[int]] = [set() for _ in range(capacity)]
    edges = grid_edges(shape)
    for first, second in edges:
        first_i, second_i = int(first), int(second)
        neighbours[first_i].add(second_i)
        neighbours[second_i].add(first_i)

    heap: list[tuple[float, int, int, int, int]] = []

    def push(first: int, second: int) -> None:
        if first == second or not active[first] or not active[second]:
            return
        if int(sizes[first] + sizes[second]) > maximum_points:
            return
        _, consistency = merge_matrix(
            matrices[first], int(sizes[first]), matrices[second], int(sizes[second])
        )
        if not np.isfinite(consistency):
            return
        low, high = sorted((int(first), int(second)))
        heappush(
            heap,
            (-float(consistency), low, high, int(version[low]), int(version[high])),
        )

    for first, second in edges:
        push(int(first), int(second))

    next_id = total
    merge_count = 0
    stale_heap_pops = 0
    while heap:
        negative, first, second, first_version, second_version = heappop(heap)
        if not active[first] or not active[second]:
            stale_heap_pops += 1
            continue
        if version[first] != first_version or version[second] != second_version:
            stale_heap_pops += 1
            continue
        if second not in neighbours[first] or first not in neighbours[second]:
            stale_heap_pops += 1
            continue

        consistency = -negative
        if consistency < threshold:
            break
        if int(sizes[first] + sizes[second]) > maximum_points:
            continue

        merged_matrix, _ = merge_matrix(
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
            # Current implementation increments this version, which invalidates every
            # still-valid edge from this neighbour to unrelated active neighbours.
            if invalidate_unchanged_neighbours:
                version[neighbour] += 1
            neighbours[next_id].add(neighbour)

        for neighbour in sorted(neighbours[next_id]):
            push(next_id, neighbour)

        next_id += 1
        merge_count += 1

    owner = np.empty(total, dtype=np.int64)
    for domain_id in range(next_id):
        if active[domain_id]:
            owner[np.asarray(leaves[domain_id], dtype=np.int64)] = domain_id

    point_domains = owner[point_cells]
    populated = np.unique(point_domains)
    ordering: list[tuple[tuple[float, float, float], int]] = []
    for domain_id in populated:
        owned = points[point_domains == domain_id]
        ordering.append((tuple(float(value) for value in owned.mean(axis=0)), int(domain_id)))
    ordering.sort()
    labels_by_id = {
        domain_id: label for label, (_, domain_id) in enumerate(ordering)
    }
    labels = np.asarray(
        [labels_by_id[int(domain_id)] for domain_id in point_domains], dtype=np.int64
    )
    centroid_labels = np.full(total, -1, dtype=np.int64)
    for index, domain_id in enumerate(owner):
        label = labels_by_id.get(int(domain_id))
        if label is not None:
            centroid_labels[index] = label

    return labels, centroid_labels, merge_count, stale_heap_pops


def _metrics(
    *,
    mode: str,
    oracle_labels: np.ndarray,
    labels: np.ndarray,
    centroid_labels: np.ndarray,
    merge_count: int,
    stale_heap_pops: int,
    elapsed_seconds: float,
) -> HeapComparison:
    accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)  # noqa: SLF001
    return HeapComparison(
        mode=mode,
        oracle_domains=int(len(np.unique(oracle_labels))),
        predicted_domains=int(len(np.unique(labels))),
        surviving_grid_domains=int(len(np.unique(centroid_labels[centroid_labels >= 0]))),
        merge_count=int(merge_count),
        stale_heap_pops=int(stale_heap_pops),
        adjusted_rand_index=float(comparison.adjusted_rand_score(oracle_labels, labels)),
        adjusted_mutual_information=float(
            comparison.adjusted_mutual_info_score(oracle_labels, labels)
        ),
        optimal_label_accuracy=float(accuracy),
        matched_points=int(matched),
        elapsed_seconds=float(elapsed_seconds),
    )


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
        "--output",
        type=Path,
        default=Path("benchmark-results/oracle-heap-invalidation-comparison.json"),
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory  # noqa: E402
    import run_selected_exact_leapfrog_lva as exact  # noqa: E402

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

    builder_class = polatory.AutomaticStructuralDomainBuilder3
    builder_module = importlib.import_module(builder_class.__module__)
    builder = builder_class(
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

    production_labels, production_centroid_labels, _, _, production_merges = (
        builder._automatic_labels(  # noqa: SLF001
            points,
            centroid_anisotropies,
            minimum,
            maximum,
            shape,
        )
    )

    print(
        f"case={case_name} points={len(points)} oracle_domains={len(np.unique(oracle_labels))} "
        f"centroids={len(centroids)} grid={shape}",
        flush=True,
    )

    results: list[HeapComparison] = []
    current_labels: np.ndarray | None = None
    current_centroid_labels: np.ndarray | None = None
    for mode, invalidate in (
        ("current_neighbour_invalidation", True),
        ("preserve_unaffected_heap_edges", False),
    ):
        started = time.perf_counter()
        labels, centroid_labels, merges, stale_pops = _automatic_labels_with_heap_mode(
            builder_module=builder_module,
            points=np.asarray(points, dtype=np.float64),
            centroid_anisotropies=np.asarray(centroid_anisotropies, dtype=np.float64),
            minimum=np.asarray(minimum, dtype=np.float64),
            maximum=np.asarray(maximum, dtype=np.float64),
            shape=tuple(int(value) for value in shape),
            minimum_fraction=args.minimum_fraction,
            maximum_fraction=args.maximum_fraction,
            threshold=args.threshold,
            invalidate_unchanged_neighbours=invalidate,
        )
        if invalidate:
            current_labels = labels
            current_centroid_labels = centroid_labels
        results.append(
            _metrics(
                mode=mode,
                oracle_labels=oracle_labels,
                labels=labels,
                centroid_labels=centroid_labels,
                merge_count=merges,
                stale_heap_pops=stale_pops,
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    if current_labels is None or current_centroid_labels is None:
        raise AssertionError("Current-mode diagnostic did not execute.")
    if not np.array_equal(current_labels, np.asarray(production_labels, dtype=np.int64)):
        raise RuntimeError(
            "Diagnostic current-mode labels do not reproduce the installed production builder."
        )
    if not np.array_equal(
        current_centroid_labels, np.asarray(production_centroid_labels, dtype=np.int64)
    ):
        raise RuntimeError(
            "Diagnostic current-mode centroid labels do not reproduce the installed builder."
        )
    if results[0].merge_count != int(production_merges):
        raise RuntimeError(
            "Diagnostic current-mode merge count does not reproduce the installed builder."
        )

    print(
        f"{'mode':>33} {'oracle':>7} {'pred':>7} {'grid':>7} {'ARI':>9} "
        f"{'AMI':>9} {'match':>9} {'merges':>8} {'stale':>9} {'seconds':>9}",
        flush=True,
    )
    for item in results:
        print(
            f"{item.mode:>33} {item.oracle_domains:>7d} {item.predicted_domains:>7d} "
            f"{item.surviving_grid_domains:>7d} {item.adjusted_rand_index:>9.5f} "
            f"{item.adjusted_mutual_information:>9.5f} "
            f"{item.optimal_label_accuracy:>9.5f} {item.merge_count:>8d} "
            f"{item.stale_heap_pops:>9d} {item.elapsed_seconds:>9.3f}",
            flush=True,
        )

    payload = {
        "case": case_name,
        "point_count": int(len(points)),
        "oracle_domains": int(len(np.unique(oracle_labels))),
        "centroid_count": int(len(centroids)),
        "grid_shape": [int(value) for value in shape],
        "threshold": float(args.threshold),
        "minimum_fraction": float(args.minimum_fraction),
        "maximum_fraction": float(args.maximum_fraction),
        "production_control_validated": True,
        "results": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
