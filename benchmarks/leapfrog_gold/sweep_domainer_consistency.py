"""Fast screening sweep for candidate Leapfrog SubDomainer consistency transforms.

This script deliberately avoids fitting RBF domains and generating meshes.  It reads
inspection CSVs produced by ``run_selected_exact_leapfrog_lva.py``, reconstructs the
structured centroid graph, samples the exported LVA matrix field at the centroids, and
runs the recovered greedy six-neighbour region-growing algorithm for several candidate
transforms of the proposed merged-matrix determinant.

The exported LVA field is a regular diagnostic grid (25^3 by default), so this is a
screening tool rather than a final parity measurement.  The strongest candidates should
be validated afterward with the full mesh benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree


Matrix = np.ndarray
Consistency = Callable[[float], float]


@dataclass(frozen=True)
class SweepResult:
    formula: str
    threshold: float
    centroid_count: int
    grid_shape: tuple[int, int, int]
    merge_count: int
    surviving_grid_domains: int
    populated_domains: int
    minimum_domain_size: int
    median_domain_size: float
    maximum_domain_size: int
    elapsed_seconds: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _matrix_from_row(row: dict[str, str]) -> Matrix:
    return np.asarray(
        [
            [float(row["matrix_00"]), float(row["matrix_01"]), float(row["matrix_02"])],
            [float(row["matrix_10"]), float(row["matrix_11"]), float(row["matrix_12"])],
            [float(row["matrix_20"]), float(row["matrix_21"]), float(row["matrix_22"])],
        ],
        dtype=np.float64,
    )


def _symmetric_determinant(matrix: Matrix) -> float:
    matrix = 0.5 * (matrix + matrix.T)
    a, b, c = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    e, f, i = float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[2, 2])
    return a * e * i + 2.0 * b * c * f - a * f * f - e * c * c - i * b * b


def _normalise_determinant(matrix: Matrix) -> Matrix:
    symmetric = 0.5 * (matrix + matrix.T)
    determinant = _symmetric_determinant(symmetric)
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("LVA matrices must be finite and positive definite")
    return symmetric / determinant ** (1.0 / 3.0)


def _grid_edges(shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    edges: list[np.ndarray] = []
    for axis, size in enumerate(shape):
        if size <= 1:
            continue
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, size - 1)
        right[axis] = slice(1, size)
        edges.append(np.column_stack([grid[tuple(left)].ravel(), grid[tuple(right)].ravel()]))
    return np.vstack(edges) if edges else np.empty((0, 2), dtype=np.int64)


def _formulae() -> dict[str, Consistency]:
    tiny = np.finfo(np.float64).tiny
    return {
        "reciprocal_det": lambda det: 1.0 / max(det, tiny),
        "inverse_sqrt_det": lambda det: 1.0 / math.sqrt(max(det, tiny)),
        "exp_abs_log_det": lambda det: math.exp(-abs(math.log(max(det, tiny)))),
        "symmetric_reciprocal": lambda det: min(det, 1.0 / max(det, tiny)),
    }


def _load_inputs(directory: Path, case: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
    centroid_rows = _read_csv(directory / f"{case}_domain_centroids.csv")
    point_rows = _read_csv(directory / f"{case}_domain_points.csv")
    field_rows = _read_csv(directory / f"{case}_lva_field.csv")

    centroid_rows.sort(key=lambda row: int(row["centroid_index"]))
    centroids = np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in centroid_rows],
        dtype=np.float64,
    )
    grid_indices = np.asarray(
        [[int(row["grid_i"]), int(row["grid_j"]), int(row["grid_k"])] for row in centroid_rows],
        dtype=np.int64,
    )
    shape = tuple(int(value) for value in (grid_indices.max(axis=0) + 1))
    if int(np.prod(shape)) != len(centroids):
        raise ValueError(f"Centroid CSV is not a complete structured grid: shape={shape}")

    points = np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in point_rows],
        dtype=np.float64,
    )
    field_points = np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in field_rows],
        dtype=np.float64,
    )
    field_matrices = np.asarray([_matrix_from_row(row) for row in field_rows], dtype=np.float64)
    nearest = np.asarray(cKDTree(field_points).query(centroids, k=1)[1], dtype=np.int64)
    centroid_matrices = np.asarray(
        [_normalise_determinant(matrix) for matrix in field_matrices[nearest]],
        dtype=np.float64,
    )
    return points, centroids, shape, centroid_matrices


def _point_cells(points: np.ndarray, centroids: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    # The benchmark centroids are cell centres. Reconstruct cell bounds from their
    # regular spacing, then use the same C-order flattening as the production builder.
    minimum = np.empty(3, dtype=np.float64)
    maximum = np.empty(3, dtype=np.float64)
    volume = centroids.reshape(shape + (3,))
    for axis, size in enumerate(shape):
        line = np.take(volume, indices=range(size), axis=axis)
        coordinates = np.unique(line[..., axis])
        coordinates.sort()
        if size == 1:
            minimum[axis] = coordinates[0]
            maximum[axis] = coordinates[0]
        else:
            step = float(np.median(np.diff(coordinates)))
            minimum[axis] = coordinates[0] - 0.5 * step
            maximum[axis] = coordinates[-1] + 0.5 * step
    indices = np.zeros((len(points), 3), dtype=np.int64)
    for axis, size in enumerate(shape):
        span = maximum[axis] - minimum[axis]
        if size <= 1 or span <= 0.0:
            continue
        normalized = (points[:, axis] - minimum[axis]) / span
        indices[:, axis] = np.clip(np.floor(normalized * size).astype(np.int64), 0, size - 1)
    return (indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]


def _run_one(
    points: np.ndarray,
    centroids: np.ndarray,
    shape: tuple[int, int, int],
    initial_matrices: np.ndarray,
    formula_name: str,
    transform: Consistency,
    threshold: float,
    maximum_fraction: float,
) -> SweepResult:
    started = time.perf_counter()
    total = len(centroids)
    maximum_size = max(1, int(math.floor(maximum_fraction * total)))
    active = np.ones(2 * total + 1, dtype=bool)
    version = np.zeros(2 * total + 1, dtype=np.int64)
    sizes = np.zeros(2 * total + 1, dtype=np.int64)
    sizes[:total] = 1
    matrices = np.zeros((2 * total + 1, 3, 3), dtype=np.float64)
    matrices[:total] = initial_matrices
    leaves: list[list[int]] = [[index] for index in range(total)] + [[] for _ in range(total + 1)]
    neighbours: list[set[int]] = [set() for _ in range(2 * total + 1)]
    edges = _grid_edges(shape)
    for first, second in edges:
        neighbours[int(first)].add(int(second))
        neighbours[int(second)].add(int(first))

    heap: list[tuple[float, int, int, int, int]] = []

    def push(first: int, second: int) -> None:
        if first == second or not active[first] or not active[second]:
            return
        if sizes[first] + sizes[second] > maximum_size:
            return
        merged_size = int(sizes[first] + sizes[second])
        merged = (sizes[first] * matrices[first] + sizes[second] * matrices[second]) / merged_size
        determinant = _symmetric_determinant(merged)
        if not np.isfinite(determinant) or determinant <= 0.0:
            return
        score = float(transform(determinant))
        low, high = sorted((first, second))
        heappush(heap, (-score, low, high, int(version[low]), int(version[high])))

    for first, second in edges:
        push(int(first), int(second))

    next_id = total
    merge_count = 0
    while heap:
        negative, first, second, first_version, second_version = heappop(heap)
        if not active[first] or not active[second]:
            continue
        if version[first] != first_version or version[second] != second_version:
            continue
        if second not in neighbours[first] or first not in neighbours[second]:
            continue
        score = -negative
        if score < threshold:
            break
        if sizes[first] + sizes[second] > maximum_size:
            continue

        merged_size = int(sizes[first] + sizes[second])
        merged_matrix = (sizes[first] * matrices[first] + sizes[second] * matrices[second]) / merged_size
        merged_neighbours = (neighbours[first] | neighbours[second]) - {first, second}
        active[first] = False
        active[second] = False
        version[first] += 1
        version[second] += 1
        active[next_id] = True
        sizes[next_id] = merged_size
        matrices[next_id] = merged_matrix
        leaves[next_id] = leaves[first] + leaves[second]
        for neighbour in sorted(merged_neighbours):
            if not active[neighbour]:
                continue
            neighbours[neighbour].discard(first)
            neighbours[neighbour].discard(second)
            neighbours[neighbour].add(next_id)
            version[neighbour] += 1
            neighbours[next_id].add(neighbour)
        for neighbour in sorted(neighbours[next_id]):
            push(next_id, neighbour)
        next_id += 1
        merge_count += 1

    owner = np.empty(total, dtype=np.int64)
    survivors: list[int] = []
    for domain_id in range(next_id):
        if active[domain_id]:
            survivors.append(domain_id)
            owner[np.asarray(leaves[domain_id], dtype=np.int64)] = domain_id
    point_domains = owner[_point_cells(points, centroids, shape)]
    populated, populations = np.unique(point_domains, return_counts=True)
    populations = np.asarray(populations, dtype=np.int64)
    return SweepResult(
        formula=formula_name,
        threshold=threshold,
        centroid_count=total,
        grid_shape=shape,
        merge_count=merge_count,
        surviving_grid_domains=len(survivors),
        populated_domains=len(populated),
        minimum_domain_size=int(populations.min()) if len(populations) else 0,
        median_domain_size=float(np.median(populations)) if len(populations) else 0.0,
        maximum_domain_size=int(populations.max()) if len(populations) else 0,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--inspection-dir",
        type=Path,
        default=Path("benchmark-results/exact-leapfrog-lva-forced/inspection-csv"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/consistency-sweep.json"),
    )
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if not 0.0 < args.maximum_fraction <= 1.0:
        parser.error("--maximum-fraction must be in (0, 1]")

    points, centroids, shape, matrices = _load_inputs(args.inspection_dir, args.case.upper())
    results = [
        _run_one(
            points,
            centroids,
            shape,
            matrices,
            name,
            transform,
            args.threshold,
            args.maximum_fraction,
        )
        for name, transform in _formulae().items()
    ]
    results.sort(key=lambda item: (abs(item.populated_domains - 15), item.populated_domains, item.formula))

    print(
        f"case={args.case.upper()} centroids={len(centroids)} grid={shape} "
        f"points={len(points)} threshold={args.threshold:g}"
    )
    header = (
        "formula", "populated", "grid_domains", "merges", "min_points",
        "median_points", "max_points", "seconds",
    )
    print("  ".join(f"{item:>20}" for item in header))
    for result in results:
        print(
            f"{result.formula:>20}  {result.populated_domains:>20d}  "
            f"{result.surviving_grid_domains:>20d}  {result.merge_count:>20d}  "
            f"{result.minimum_domain_size:>20d}  {result.median_domain_size:>20.1f}  "
            f"{result.maximum_domain_size:>20d}  {result.elapsed_seconds:>20.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": args.case.upper(),
        "inspection_directory": str(args.inspection_dir),
        "screening_note": (
            "Centroid matrices are nearest-neighbour samples of the exported regular LVA "
            "diagnostic grid; validate finalists with the full mesh benchmark."
        ),
        "results": [asdict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
