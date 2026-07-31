"""Leapfrog-style automatic structural domain builder.

This implementation replaces the earlier pairwise-SPD-similarity approximation
with the grid-seeded region-growing algorithm recovered from Leapfrog Geo
2026.1 runtime profiles:

* an approximately requested structured centroid grid;
* one initial domain per centroid;
* six-connected adjacency;
* point-count-weighted arithmetic matrix means;
* determinant consistency of the proposed merged matrix;
* global greedy heap ordering with lazy invalidation;
* hard centroid-population maximum during merging;
* real interpolation points transferred to final grid domains afterward.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import floor
from typing import Sequence

import numpy as np

from ._structural import StructuralDomain3, StructuralDomainBuilder3, StructuralTrendType
from .automatic_domain_builder import AutomaticStructuralDomainDiagnostics3
from .labeled_domain_builder import (
    LabeledStructuralDomainBuilder3,
    sample_single_input_anisotropies3,
)


def _leapfrog_grid_shape(
    target_count: int,
    spans: np.ndarray,
    active_axes: np.ndarray,
) -> tuple[int, int, int]:
    """Return a near-isotropic grid with at least ``target_count`` centroids."""
    if target_count <= 0:
        raise ValueError("centroid_count must be positive")
    dimensions = np.ones(3, dtype=np.int64)
    indices = np.flatnonzero(active_axes)
    if len(indices) == 0:
        indices = np.arange(3, dtype=np.int64)
    active_spans = np.asarray(spans[indices], dtype=float)
    positive = active_spans[active_spans > 0.0]
    if len(positive) == 0:
        active_spans = np.ones(len(indices), dtype=float)
    else:
        active_spans = np.maximum(active_spans, positive.min() * 1.0e-12)

    density = (float(target_count) / float(np.prod(active_spans))) ** (1.0 / len(indices))
    counts = np.maximum(np.rint(active_spans * density).astype(np.int64), 2)
    while int(np.prod(counts)) < target_count:
        cell_sizes = active_spans / np.maximum(counts - 1, 1)
        counts[int(np.argmax(cell_sizes))] += 1
    dimensions[indices] = counts
    return tuple(int(value) for value in dimensions)


def _grid_centroids(
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    axes: list[np.ndarray] = []
    for axis, size in enumerate(shape):
        if size <= 1 or not maximum[axis] > minimum[axis]:
            axes.append(np.asarray([(minimum[axis] + maximum[axis]) * 0.5]))
        else:
            step = (maximum[axis] - minimum[axis]) / size
            axes.append(minimum[axis] + (np.arange(size, dtype=float) + 0.5) * step)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _point_cell_indices(
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    indices = np.zeros((len(points), 3), dtype=np.int64)
    for axis, size in enumerate(shape):
        span = maximum[axis] - minimum[axis]
        if size <= 1 or not span > 0.0:
            continue
        normalized = (points[:, axis] - minimum[axis]) / span
        indices[:, axis] = np.clip(
            np.floor(normalized * size).astype(np.int64), 0, size - 1
        )
    return (indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]


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
        edges.append(
            np.column_stack([grid[tuple(left)].ravel(), grid[tuple(right)].ravel()])
        )
    return np.vstack(edges) if edges else np.empty((0, 2), dtype=np.int64)


def _symmetric_determinant(matrix: np.ndarray) -> float:
    matrix = 0.5 * (matrix + matrix.T)
    a, b, c = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    e, f, i = float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[2, 2])
    return a * e * i + 2.0 * b * c * f - a * f * f - e * c * c - i * b * b


def _normalise_determinant(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    determinant = _symmetric_determinant(symmetric)
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("anisotropy matrices must be positive definite")
    return symmetric / determinant ** (1.0 / 3.0)


def _merged_matrix_and_consistency(
    first_matrix: np.ndarray,
    first_size: int,
    second_matrix: np.ndarray,
    second_size: int,
) -> tuple[np.ndarray, float]:
    total = first_size + second_size
    weight = first_size / float(total)
    merged = weight * first_matrix + (1.0 - weight) * second_matrix
    merged = 0.5 * (merged + merged.T)
    determinant = _symmetric_determinant(merged)
    consistency = float("-inf") if determinant <= 0.0 else float(1.0 / determinant)
    return merged, consistency


class AutomaticStructuralDomainBuilder3:
    """Build structural domains using Leapfrog-style grid region growing."""

    def __init__(
        self,
        centroid_count: int = 6000,
        minimum_cluster_fraction: float = 0.001,
        maximum_cluster_fraction: float = 0.10,
        consistency_threshold: float = 0.60,
        base_range: float = 0.0,
        support_multiplier: int = 5,
        minimum_support_points: int = 1,
    ) -> None:
        if centroid_count <= 0:
            raise ValueError("centroid_count must be positive")
        if not 0.0 < minimum_cluster_fraction <= maximum_cluster_fraction <= 1.0:
            raise ValueError("cluster fractions must satisfy 0 < minimum <= maximum <= 1")
        if not 0.0 < consistency_threshold <= 1.0:
            raise ValueError("consistency_threshold must be in (0, 1]")
        if base_range < 0.0:
            raise ValueError("base_range must be non-negative")
        if support_multiplier <= 0 or minimum_support_points <= 0:
            raise ValueError("support settings must be positive")
        self.centroid_count = int(centroid_count)
        self.minimum_cluster_fraction = float(minimum_cluster_fraction)
        self.maximum_cluster_fraction = float(maximum_cluster_fraction)
        self.consistency_threshold = float(consistency_threshold)
        self.base_range = float(base_range)
        self.support_multiplier = int(support_multiplier)
        self.minimum_support_points = int(minimum_support_points)
        self.labels_: np.ndarray | None = None
        self.centroid_labels_: np.ndarray | None = None
        self.centroid_points_: np.ndarray | None = None
        self.centroid_grid_shape_: tuple[int, int, int] | None = None
        self.active_axes_: np.ndarray | None = None
        self.diagnostics_: AutomaticStructuralDomainDiagnostics3 | None = None

    @staticmethod
    def _validate(
        points: np.ndarray,
        anisotropies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        anisotropies = np.asarray(anisotropies, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must have shape (n, 3) and must not be empty")
        if anisotropies.shape != (len(points), 3, 3):
            raise ValueError("anisotropies must have shape (n, 3, 3)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(anisotropies)):
            raise ValueError("points and anisotropies must be finite")
        anisotropies = np.asarray([_normalise_determinant(m) for m in anisotropies])
        return points, anisotropies

    def _automatic_labels(
        self,
        points: np.ndarray,
        centroid_anisotropies: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        total = len(centroid_anisotropies)
        point_cells = _point_cell_indices(points, minimum, maximum, shape)
        minimum_points = max(1, int(floor(self.minimum_cluster_fraction * total)))
        maximum_points = max(
            minimum_points,
            int(floor(self.maximum_cluster_fraction * total)),
        )

        active = np.ones(total, dtype=bool)
        version = np.zeros(total, dtype=np.int64)
        sizes = np.ones(total, dtype=np.int64)
        matrices = np.asarray(
            [_normalise_determinant(matrix) for matrix in centroid_anisotropies],
            dtype=float,
        )
        leaves: list[list[int]] = [[index] for index in range(total)]
        neighbours: list[set[int]] = [set() for _ in range(2 * total + 1)]
        edges = _grid_edges(shape)
        for first, second in edges:
            neighbours[int(first)].add(int(second))
            neighbours[int(second)].add(int(first))

        heap: list[tuple[float, int, int, int, int]] = []

        def push(first: int, second: int) -> None:
            if not active[first] or not active[second] or first == second:
                return
            if sizes[first] + sizes[second] > maximum_points:
                return
            _, consistency = _merged_matrix_and_consistency(
                matrices[first], int(sizes[first]), matrices[second], int(sizes[second])
            )
            if not np.isfinite(consistency):
                return
            low, high = sorted((first, second))
            heappush(
                heap,
                (-consistency, low, high, int(version[low]), int(version[high])),
            )

        for first, second in edges:
            push(int(first), int(second))

        next_id = total
        merge_count = 0
        while heap:
            negative, first, second, first_version, second_version = heappop(heap)
            if first >= len(active) or second >= len(active):
                continue
            if not active[first] or not active[second]:
                continue
            if version[first] != first_version or version[second] != second_version:
                continue
            if second not in neighbours[first] or first not in neighbours[second]:
                continue
            consistency = -negative
            if consistency < self.consistency_threshold:
                break
            if sizes[first] + sizes[second] > maximum_points:
                continue

            if next_id >= len(active):
                grow = max(total, next_id - len(active) + 1)
                active = np.pad(active, (0, grow))
                version = np.pad(version, (0, grow))
                sizes = np.pad(sizes, (0, grow))
                matrices = np.pad(matrices, ((0, grow), (0, 0), (0, 0)))
                leaves.extend([] for _ in range(grow))
                neighbours.extend(set() for _ in range(grow))

            merged_matrix, _ = _merged_matrix_and_consistency(
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
            neighbours[next_id] = set()
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
        surviving_ids: list[int] = []
        for domain_id in range(next_id):
            if not active[domain_id]:
                continue
            surviving_ids.append(domain_id)
            owner[np.asarray(leaves[domain_id], dtype=np.int64)] = domain_id

        point_domains = owner[point_cells]
        populated = np.unique(point_domains)
        ordering: list[tuple[tuple[float, float, float], int]] = []
        for domain_id in populated:
            owned = points[point_domains == domain_id]
            ordering.append((tuple(float(v) for v in owned.mean(axis=0)), int(domain_id)))
        ordering.sort()
        labels_by_id = {domain_id: label for label, (_, domain_id) in enumerate(ordering)}
        labels = np.asarray([labels_by_id[int(v)] for v in point_domains], dtype=np.int64)
        centroid_labels = np.full(total, -1, dtype=np.int64)
        for index, domain_id in enumerate(owner):
            label = labels_by_id.get(int(domain_id))
            if label is not None:
                centroid_labels[index] = label
        return labels, centroid_labels, minimum_points, maximum_points, merge_count

    def _prepare_grid(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        spans = maximum - minimum
        active_axes = spans > max(float(spans.max()), 1.0) * 1.0e-12
        if not np.any(active_axes):
            active_axes[:] = True
        shape = _leapfrog_grid_shape(self.centroid_count, spans, active_axes)
        centroids = _grid_centroids(minimum, maximum, shape)
        return minimum, maximum, active_axes, shape, centroids

    def _finish(
        self,
        points: np.ndarray,
        point_anisotropies: np.ndarray,
        centroid_anisotropies: np.ndarray,
        model_parameters: Sequence[float],
        minimum: np.ndarray,
        maximum: np.ndarray,
        active_axes: np.ndarray,
        shape: tuple[int, int, int],
        centroid_points: np.ndarray,
    ) -> list[StructuralDomain3]:
        labels, centroid_labels, minimum_points, maximum_points, merge_count = self._automatic_labels(
            points, centroid_anisotropies, minimum, maximum, shape
        )
        postcluster_builder = LabeledStructuralDomainBuilder3(
            base_range=self.base_range,
            support_multiplier=self.support_multiplier,
            minimum_support_points=self.minimum_support_points,
        )
        domains = postcluster_builder.build(
            points, labels, point_anisotropies, model_parameters
        )
        self.labels_ = labels.copy()
        self.centroid_labels_ = centroid_labels.copy()
        self.centroid_points_ = centroid_points.copy()
        self.centroid_grid_shape_ = shape
        self.active_axes_ = active_axes.copy()
        self.diagnostics_ = AutomaticStructuralDomainDiagnostics3(
            labels=labels.copy(),
            centroid_points=centroid_points.copy(),
            centroid_labels=centroid_labels.copy(),
            centroid_grid_shape=shape,
            active_axes=active_axes.copy(),
            minimum_points=minimum_points,
            maximum_points=maximum_points,
            consistency_threshold=self.consistency_threshold,
            merge_count=merge_count,
            final_domain_count=len(domains),
            postcluster=tuple(postcluster_builder.diagnostics_ or ()),
        )
        return domains

    def build(
        self,
        points: np.ndarray,
        anisotropies: np.ndarray,
        model_parameters: Sequence[float],
    ) -> list[StructuralDomain3]:
        points, anisotropies = self._validate(points, anisotropies)
        minimum, maximum, active_axes, shape, centroids = self._prepare_grid(points)
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except ImportError:
            squared = np.sum((centroids[:, None, :] - points[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(squared, axis=1)
        else:
            nearest = np.asarray(cKDTree(points).query(centroids, k=1)[1], dtype=np.int64)
        centroid_anisotropies = anisotropies[nearest]
        return self._finish(
            points, anisotropies, centroid_anisotropies, model_parameters,
            minimum, maximum, active_axes, shape, centroids,
        )

    def build_from_inputs(
        self,
        points: np.ndarray,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
        trend_type: object = StructuralTrendType.STRONGEST_ALONG_INPUTS,
    ) -> list[StructuralDomain3]:
        points = np.asarray(points, dtype=float)
        inputs = list(inputs)
        if not inputs:
            raise ValueError("inputs must not be empty")
        minimum, maximum, active_axes, shape, centroids = self._prepare_grid(points)
        if len(inputs) == 1:
            non_decaying = trend_type == StructuralTrendType.NON_DECAYING
            point_anisotropies = sample_single_input_anisotropies3(
                points, inputs[0], non_decaying=non_decaying
            )
            centroid_anisotropies = sample_single_input_anisotropies3(
                centroids, inputs[0], non_decaying=non_decaying
            )
        else:
            sampler = StructuralDomainBuilder3()
            point_anisotropies = np.asarray(
                sampler.sample(points, inputs, trend_type).anisotropies, dtype=float
            )
            centroid_anisotropies = np.asarray(
                sampler.sample(centroids, inputs, trend_type).anisotropies, dtype=float
            )
        _, point_anisotropies = self._validate(points, point_anisotropies)
        centroid_anisotropies = np.asarray(
            [_normalise_determinant(matrix) for matrix in centroid_anisotropies], dtype=float
        )
        return self._finish(
            points, point_anisotropies, centroid_anisotropies, model_parameters,
            minimum, maximum, active_axes, shape, centroids,
        )


LeapfrogAutomaticDomainBuilder3 = AutomaticStructuralDomainBuilder3
