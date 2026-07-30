"""Corrected Leapfrog-style automatic structural domain builder.

This module keeps the recovered grid, adjacency, merge-matrix, size-limit, and
point-transfer behaviour from :mod:`leapfrog_automatic_domain_builder`, while
fixing the lazy-heap invalidation semantics.

When two domains merge, an unchanged neighbouring domain changes only one
adjacency: edges to the two retired domains are replaced by an edge to the new
merged domain. Its matrix, population, and edges to every other active domain
remain unchanged. Incrementing that neighbour's version invalidates all of
those still-valid heap pairs and can make region growing stop prematurely.
Only retired or otherwise numerically changed domains should invalidate their
existing heap entries.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import floor

import numpy as np

from .leapfrog_automatic_domain_builder import (
    AutomaticStructuralDomainBuilder3 as _PreviousAutomaticStructuralDomainBuilder3,
    _grid_centroids,
    _grid_edges,
    _leapfrog_grid_shape,
    _merged_matrix_and_consistency,
    _normalise_determinant,
    _point_cell_indices,
    _symmetric_determinant,
)


class AutomaticStructuralDomainBuilder3(_PreviousAutomaticStructuralDomainBuilder3):
    """Grid-seeded region grower with correct lazy-heap preservation."""

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
                matrices[first],
                int(sizes[first]),
                matrices[second],
                int(sizes[second]),
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
                matrices[first],
                int(sizes[first]),
                matrices[second],
                int(sizes[second]),
            )
            merged_neighbours = (neighbours[first] | neighbours[second]) - {
                first,
                second,
            }
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
                # Preserve the neighbour's version. Its matrix, size, and all
                # unrelated active edges are unchanged, so their heap entries
                # remain valid. Edges to first/second are rejected because those
                # domains are inactive; the new edge is pushed below.
                neighbours[next_id].add(neighbour)
            for neighbour in sorted(neighbours[next_id]):
                push(next_id, neighbour)
            next_id += 1
            merge_count += 1

        owner = np.empty(total, dtype=np.int64)
        for domain_id in range(next_id):
            if not active[domain_id]:
                continue
            owner[np.asarray(leaves[domain_id], dtype=np.int64)] = domain_id

        point_domains = owner[point_cells]
        populated = np.unique(point_domains)
        ordering: list[tuple[tuple[float, float, float], int]] = []
        for domain_id in populated:
            owned = points[point_domains == domain_id]
            ordering.append(
                (tuple(float(value) for value in owned.mean(axis=0)), int(domain_id))
            )
        ordering.sort()
        labels_by_id = {
            domain_id: label for label, (_, domain_id) in enumerate(ordering)
        }
        labels = np.asarray(
            [labels_by_id[int(domain_id)] for domain_id in point_domains],
            dtype=np.int64,
        )
        centroid_labels = np.full(total, -1, dtype=np.int64)
        for index, domain_id in enumerate(owner):
            label = labels_by_id.get(int(domain_id))
            if label is not None:
                centroid_labels[index] = label
        return labels, centroid_labels, minimum_points, maximum_points, merge_count


LeapfrogAutomaticDomainBuilder3 = AutomaticStructuralDomainBuilder3
