"""Automatic structural SubDomainer reconstructed from Leapfrog evidence.

The builder is independent of decoded Leapfrog labels.  It creates a deterministic
regular cloud of structural centroids, samples the LVA matrix field on that cloud,
and performs adjacency-constrained agglomeration.  The resulting point labels are
then passed through :class:`LabeledStructuralDomainBuilder3`, so the recovered
post-cluster support, local-range and bounding-box rules remain shared with the
oracle-label path.

The public parameters mirror the controls observed in Leapfrog projects:

* ``centroid_count`` (normally 6000),
* minimum and maximum cluster fractions,
* a matrix-consistency threshold.

No benchmark coordinates, case names, point labels or domain counts are stored in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from math import ceil, exp, floor, log
from typing import Sequence

import numpy as np

from ._structural import StructuralDomain3, StructuralDomainBuilder3, StructuralTrendType
from .labeled_domain_builder import (
    LabeledStructuralDomainBuilder3,
    LabeledStructuralDomainDiagnostics3,
    sample_single_input_anisotropies3,
)


@dataclass(frozen=True)
class AutomaticStructuralDomainDiagnostics3:
    """Diagnostics from one automatic SubDomainer build."""

    labels: np.ndarray
    centroid_points: np.ndarray
    centroid_labels: np.ndarray
    centroid_grid_shape: tuple[int, int, int]
    active_axes: np.ndarray
    minimum_points: int
    maximum_points: int
    consistency_threshold: float
    merge_count: int
    final_domain_count: int
    postcluster: tuple[LabeledStructuralDomainDiagnostics3, ...]


def _positive_divisors(value: int) -> list[int]:
    result: list[int] = []
    limit = int(value**0.5)
    for candidate in range(1, limit + 1):
        if value % candidate != 0:
            continue
        result.append(candidate)
        other = value // candidate
        if other != candidate:
            result.append(other)
    return sorted(result)


def _factor_grid_shape(
    count: int,
    spans: np.ndarray,
    active_axes: np.ndarray,
) -> tuple[int, int, int]:
    """Factor ``count`` into an aspect-ratio-aware three-dimensional grid."""

    if count <= 0:
        raise ValueError("centroid_count must be positive")

    active_indices = np.flatnonzero(active_axes)
    dimensions = np.ones(3, dtype=np.int64)
    active_count = len(active_indices)
    if active_count == 0:
        active_indices = np.arange(3, dtype=np.int64)
        active_count = 3

    active_spans = np.asarray(spans[active_indices], dtype=float)
    positive = active_spans[active_spans > 0.0]
    if len(positive) == 0:
        active_spans = np.ones(active_count, dtype=float)
    else:
        active_spans = np.maximum(active_spans, positive.min() * 1e-12)

    span_log = np.log(active_spans)
    span_log -= span_log.mean()

    candidates: list[tuple[int, ...]] = []
    if active_count == 1:
        candidates = [(count,)]
    elif active_count == 2:
        for first in _positive_divisors(count):
            candidates.append((first, count // first))
    else:
        for first in _positive_divisors(count):
            remaining = count // first
            for second in _positive_divisors(remaining):
                if remaining % second == 0:
                    candidates.append((first, second, remaining // second))

    def candidate_score(candidate: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        candidate_log = np.log(np.asarray(candidate, dtype=float))
        candidate_log -= candidate_log.mean()
        score = float(np.sum((candidate_log - span_log) ** 2))
        return score, candidate

    best = min(candidates, key=candidate_score)
    dimensions[active_indices] = np.asarray(best, dtype=np.int64)
    return tuple(int(value) for value in dimensions)


def _grid_centroids(
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    axes: list[np.ndarray] = []
    for axis, size in enumerate(shape):
        if size <= 1 or not maximum[axis] > minimum[axis]:
            axes.append(np.asarray([(minimum[axis] + maximum[axis]) * 0.5]))
        else:
            step = (maximum[axis] - minimum[axis]) / size
            axes.append(minimum[axis] + (np.arange(size, dtype=float) + 0.5) * step)

    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, (axes[0], axes[1], axes[2])


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
            np.floor(normalized * size).astype(np.int64),
            0,
            size - 1,
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
            np.column_stack(
                [grid[tuple(left)].ravel(), grid[tuple(right)].ravel()]
            )
        )
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(edges)


def _determinant_normalized(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if not np.all(eigenvalues > 0.0):
        raise ValueError("anisotropy matrices must be positive definite")
    determinant = float(np.prod(eigenvalues))
    return symmetric / determinant ** (1.0 / 3.0)


def _matrix_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Affine-invariant SPD similarity in the interval ``(0, 1]``."""

    first = _determinant_normalized(first)
    second = _determinant_normalized(second)
    eigenvalues, eigenvectors = np.linalg.eigh(first)
    inverse_sqrt = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    relative = inverse_sqrt @ second @ inverse_sqrt
    relative_eigenvalues = np.linalg.eigvalsh(0.5 * (relative + relative.T))
    relative_eigenvalues = np.maximum(relative_eigenvalues, np.finfo(float).tiny)
    distance = float(np.linalg.norm(np.log(relative_eigenvalues)) / np.sqrt(3.0))
    return exp(-distance)


class AutomaticStructuralDomainBuilder3:
    """Build structural domains without external or decoded cluster labels.

    Parameters
    ----------
    centroid_count:
        Number of deterministic structural centroids.  The observed Leapfrog
        default is 6000, but any positive factorable count is supported.
    minimum_cluster_fraction, maximum_cluster_fraction:
        Minimum and maximum final core populations as fractions of the input
        interpolation points.  Values are resolved with ``ceil`` and ``floor``.
    consistency_threshold:
        Minimum affine-invariant similarity for an adjacent merge.
    base_range:
        Base CovSpheroidal3 range used by the exact post-cluster builder.
    support_multiplier:
        Maximum support-to-core population used by the exact post-cluster rule.
    minimum_support_points:
        Minimum support population for every local solve.
    """

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
        if not 0.0 < minimum_cluster_fraction <= 1.0:
            raise ValueError("minimum_cluster_fraction must be in (0, 1]")
        if not 0.0 < maximum_cluster_fraction <= 1.0:
            raise ValueError("maximum_cluster_fraction must be in (0, 1]")
        if minimum_cluster_fraction > maximum_cluster_fraction:
            raise ValueError(
                "minimum_cluster_fraction must not exceed maximum_cluster_fraction"
            )
        if not 0.0 < consistency_threshold <= 1.0:
            raise ValueError("consistency_threshold must be in (0, 1]")
        if base_range < 0.0:
            raise ValueError("base_range must be non-negative")
        if support_multiplier <= 0:
            raise ValueError("support_multiplier must be positive")
        if minimum_support_points <= 0:
            raise ValueError("minimum_support_points must be positive")

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
        for matrix in anisotropies:
            _determinant_normalized(matrix)
        return points, anisotropies

    def _automatic_labels(
        self,
        points: np.ndarray,
        centroid_anisotropies: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        centroid_total = len(centroid_anisotropies)
        point_cells = _point_cell_indices(points, minimum, maximum, shape)
        point_counts = np.bincount(point_cells, minlength=centroid_total).astype(np.int64)

        minimum_points = max(1, int(ceil(self.minimum_cluster_fraction * len(points))))
        maximum_points = max(
            minimum_points,
            int(floor(self.maximum_cluster_fraction * len(points))),
        )

        parent = np.arange(centroid_total, dtype=np.int64)
        version = np.zeros(centroid_total, dtype=np.int64)
        centroid_counts = np.ones(centroid_total, dtype=np.int64)
        matrix_sums = centroid_anisotropies.copy()
        component_points = point_counts.copy()
        active = np.ones(centroid_total, dtype=bool)
        neighbours: list[set[int]] = [set() for _ in range(centroid_total)]

        def root(value: int) -> int:
            current = value
            while parent[current] != current:
                parent[current] = parent[parent[current]]
                current = int(parent[current])
            return current

        def mean_matrix(value: int) -> np.ndarray:
            return matrix_sums[value] / centroid_counts[value]

        def similarity(first: int, second: int) -> float:
            return _matrix_similarity(mean_matrix(first), mean_matrix(second))

        edges = _grid_edges(shape)
        for first, second in edges:
            neighbours[int(first)].add(int(second))
            neighbours[int(second)].add(int(first))

        heap: list[tuple[float, int, int, int, int, int]] = []

        def push_edge(first: int, second: int) -> None:
            first = root(first)
            second = root(second)
            if first == second or not active[first] or not active[second]:
                return
            if first > second:
                first, second = second, first
            score = similarity(first, second)
            combined_points = int(component_points[first] + component_points[second])
            heappush(
                heap,
                (
                    -score,
                    combined_points,
                    first,
                    second,
                    int(version[first]),
                    int(version[second]),
                ),
            )

        for first, second in edges:
            push_edge(int(first), int(second))

        merge_count = 0

        def merge(first: int, second: int) -> int:
            nonlocal merge_count
            first = root(first)
            second = root(second)
            if first == second:
                return first
            # The smaller persistent ID wins, making ties reproducible.
            if first > second:
                first, second = second, first
            parent[second] = first
            active[second] = False
            centroid_counts[first] += centroid_counts[second]
            matrix_sums[first] += matrix_sums[second]
            component_points[first] += component_points[second]
            version[first] += 1
            version[second] += 1

            combined_neighbours = neighbours[first] | neighbours[second]
            combined_neighbours.discard(first)
            combined_neighbours.discard(second)
            neighbours[first] = set()
            neighbours[second] = set()
            for neighbour in combined_neighbours:
                neighbour = root(neighbour)
                if neighbour == first or not active[neighbour]:
                    continue
                neighbours[first].add(neighbour)
                neighbours[neighbour].discard(second)
                neighbours[neighbour].discard(first)
                neighbours[neighbour].add(first)
            merge_count += 1
            for neighbour in sorted(neighbours[first]):
                push_edge(first, neighbour)
            return first

        # Main adjacency-constrained merge pass.  Empty geometric cells may merge
        # freely, but a populated component may never exceed the requested maximum.
        while heap:
            negative_score, _, first, second, first_version, second_version = heappop(heap)
            first = root(first)
            second = root(second)
            if first == second or not active[first] or not active[second]:
                continue
            if version[first] != first_version or version[second] != second_version:
                push_edge(first, second)
                continue
            score = -negative_score
            if score < self.consistency_threshold:
                break
            combined_points = int(component_points[first] + component_points[second])
            if combined_points > maximum_points:
                continue
            merge(first, second)

        # Absorb undersized populated components into their most compatible
        # neighbour.  This mirrors the UI minimum-population control while keeping
        # the operation deterministic and spatially local.
        changed = True
        while changed:
            changed = False
            roots = [index for index in range(centroid_total) if active[index]]
            roots.sort(key=lambda item: (component_points[item], item))
            for first in roots:
                if not active[first] or component_points[first] == 0:
                    continue
                if component_points[first] >= minimum_points:
                    continue
                candidates: list[tuple[int, float, int]] = []
                for neighbour in sorted(neighbours[first]):
                    neighbour = root(neighbour)
                    if neighbour == first or not active[neighbour]:
                        continue
                    combined = int(component_points[first] + component_points[neighbour])
                    overflow = max(0, combined - maximum_points)
                    candidates.append((overflow, -similarity(first, neighbour), neighbour))
                if not candidates:
                    continue
                _, _, second = min(candidates)
                merge(first, second)
                changed = True
                break

        roots_for_cells = np.asarray([root(index) for index in range(centroid_total)])
        roots_for_points = roots_for_cells[point_cells]
        populated_roots = np.unique(roots_for_points)

        # Canonical domain numbering is based on the spatial centroid of owned
        # interpolation points, not on transient merge IDs.
        ordering: list[tuple[tuple[float, float, float], int]] = []
        for component in populated_roots:
            owned = points[roots_for_points == component]
            centre = tuple(float(value) for value in owned.mean(axis=0))
            ordering.append((centre, int(component)))
        ordering.sort()
        label_for_root = {component: label for label, (_, component) in enumerate(ordering)}

        labels = np.asarray([label_for_root[int(value)] for value in roots_for_points], dtype=np.int64)
        centroid_labels = np.full(centroid_total, -1, dtype=np.int64)
        for index, component in enumerate(roots_for_cells):
            if int(component) in label_for_root:
                centroid_labels[index] = label_for_root[int(component)]

        return labels, centroid_labels, minimum_points, maximum_points, merge_count

    def build(
        self,
        points: np.ndarray,
        anisotropies: np.ndarray,
        model_parameters: Sequence[float],
    ) -> list[StructuralDomain3]:
        """Build automatic domains from already sampled point anisotropies."""

        points, anisotropies = self._validate(points, anisotropies)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        spans = maximum - minimum
        active_axes = spans > max(float(spans.max()), 1.0) * 1e-12
        if not np.any(active_axes):
            active_axes[:] = True

        shape = _factor_grid_shape(self.centroid_count, spans, active_axes)
        centroid_points, _ = _grid_centroids(minimum, maximum, shape)

        # Interpolate point matrices to centroids only when callers provide no
        # structural inputs.  Nearest input-point sampling preserves SPD matrices
        # and keeps this lower-level API dependency-free.
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except ImportError:
            nearest = np.empty(len(centroid_points), dtype=np.int64)
            best = np.full(len(centroid_points), np.inf)
            for start in range(0, len(points), 1024):
                stop = min(start + 1024, len(points))
                difference = centroid_points[:, None, :] - points[None, start:stop, :]
                squared = np.einsum("cpi,cpi->cp", difference, difference, optimize=True)
                local = np.argmin(squared, axis=1)
                local_squared = squared[np.arange(len(centroid_points)), local]
                replace = local_squared < best
                best[replace] = local_squared[replace]
                nearest[replace] = start + local[replace]
        else:
            nearest = np.asarray(cKDTree(points).query(centroid_points, k=1)[1], dtype=np.int64)
        centroid_anisotropies = anisotropies[nearest]

        labels, centroid_labels, minimum_points, maximum_points, merge_count = self._automatic_labels(
            points,
            centroid_anisotropies,
            minimum,
            maximum,
            shape,
        )

        postcluster_builder = LabeledStructuralDomainBuilder3(
            base_range=self.base_range,
            support_multiplier=self.support_multiplier,
            minimum_support_points=self.minimum_support_points,
        )
        domains = postcluster_builder.build(
            points,
            labels,
            anisotropies,
            model_parameters,
        )
        postcluster = tuple(postcluster_builder.diagnostics_ or ())

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
            postcluster=postcluster,
        )
        return domains

    def build_from_inputs(
        self,
        points: np.ndarray,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
        trend_type: object = StructuralTrendType.STRONGEST_ALONG_INPUTS,
    ) -> list[StructuralDomain3]:
        """Sample the LVA field and build automatic structural domains."""

        points = np.asarray(points, dtype=float)
        inputs = list(inputs)
        if not inputs:
            raise ValueError("inputs must not be empty")

        if len(inputs) == 1:
            non_decaying = trend_type == StructuralTrendType.NON_DECAYING
            point_anisotropies = sample_single_input_anisotropies3(
                points,
                inputs[0],
                non_decaying=non_decaying,
            )
        else:
            samples = StructuralDomainBuilder3().sample(points, inputs, trend_type)
            point_anisotropies = np.asarray(samples.anisotropies, dtype=float)

        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        spans = maximum - minimum
        active_axes = spans > max(float(spans.max()), 1.0) * 1e-12
        if not np.any(active_axes):
            active_axes[:] = True
        shape = _factor_grid_shape(self.centroid_count, spans, active_axes)
        centroid_points, _ = _grid_centroids(minimum, maximum, shape)

        if len(inputs) == 1:
            centroid_anisotropies = sample_single_input_anisotropies3(
                centroid_points,
                inputs[0],
                non_decaying=trend_type == StructuralTrendType.NON_DECAYING,
            )
        else:
            centroid_samples = StructuralDomainBuilder3().sample(
                centroid_points,
                inputs,
                trend_type,
            )
            centroid_anisotropies = np.asarray(
                centroid_samples.anisotropies,
                dtype=float,
            )

        labels, centroid_labels, minimum_points, maximum_points, merge_count = self._automatic_labels(
            points,
            centroid_anisotropies,
            minimum,
            maximum,
            shape,
        )

        postcluster_builder = LabeledStructuralDomainBuilder3(
            base_range=self.base_range,
            support_multiplier=self.support_multiplier,
            minimum_support_points=self.minimum_support_points,
        )
        domains = postcluster_builder.build(
            points,
            labels,
            point_anisotropies,
            model_parameters,
        )
        postcluster = tuple(postcluster_builder.diagnostics_ or ())

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
            postcluster=postcluster,
        )
        return domains


LeapfrogAutomaticDomainBuilder3 = AutomaticStructuralDomainBuilder3
