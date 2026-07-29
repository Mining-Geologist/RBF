"""Leapfrog-style grid-seeded structural domaining reference implementation.

This module mirrors the algorithm recovered from Leapfrog Geo 2026.1 runtime
profiles. It stays dependency-light (NumPy only) so it can be used as an
executable specification while the native Polatory builder is brought to exact
parity.

Recovered behaviour implemented here:

* approximately ``num_seeds`` structured grid locations;
* one initial domain per grid location;
* 6-connected face adjacency;
* point-count-weighted arithmetic matrix merging;
* determinant-based affine consistency;
* global greedy best-neighbour merging with lazy heap invalidation;
* deterministic sequential IDs for merged domains;
* a hard maximum merged-domain size;
* transfer of real locations to final grid domains after region growing.

The exact Leapfrog return expression around the determinant has not yet been
observed as bytecode. The default reciprocal-determinant metric is the
best-supported reconstruction and is isolated behind ``consistency_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ConsistencyFunction = Callable[[FloatArray], float]


def symmetric_determinant(matrix: ArrayLike) -> float:
    """Return the determinant of a symmetric 3x3 matrix from its six terms."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3); got {m.shape}")
    a, b, c = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    e, f, i = float(m[1, 1]), float(m[1, 2]), float(m[2, 2])
    return a * e * i + 2.0 * b * c * f - a * f * f - e * c * c - i * b * b


def reciprocal_determinant_consistency(matrix: ArrayLike) -> float:
    """Candidate Leapfrog consistency: ``1 / det(mean affine matrix)``."""
    determinant = symmetric_determinant(matrix)
    if not np.isfinite(determinant) or determinant <= 0.0:
        return float("-inf")
    return float(1.0 / determinant)


def weighted_mean_matrix(
    first: ArrayLike,
    first_size: int,
    second: ArrayLike,
    second_size: int,
) -> FloatArray:
    """Return Leapfrog's point-count-weighted arithmetic matrix mean."""
    if first_size <= 0 or second_size <= 0:
        raise ValueError("domain sizes must be positive")
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("both matrices must have shape (3, 3)")
    weight = first_size / float(first_size + second_size)
    merged = weight * a + (1.0 - weight) * b
    return 0.5 * (merged + merged.T)


def calculate_grid_shape(
    bbox_min: ArrayLike,
    bbox_max: ArrayLike,
    num_seeds: int,
) -> tuple[int, int, int]:
    """Choose a near-isotropic structured grid with at least ``num_seeds`` nodes."""
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("bbox_min and bbox_max must each have shape (3,)")
    span = maximum - minimum
    if np.any(~np.isfinite(span)) or np.any(span <= 0.0):
        raise ValueError("bounding box must be finite with positive side lengths")

    density = (float(num_seeds) / float(np.prod(span))) ** (1.0 / 3.0)
    counts = np.maximum(np.rint(span * density).astype(np.int64), 2)
    while int(np.prod(counts)) < num_seeds:
        cell_sizes = span / np.maximum(counts - 1, 1)
        axis = int(np.argmax(cell_sizes))
        counts[axis] += 1
    return tuple(int(value) for value in counts)


def create_grid_locations(
    bbox_min: ArrayLike,
    bbox_max: ArrayLike,
    shape: Sequence[int],
) -> FloatArray:
    """Create C-order grid locations, with the last axis varying fastest."""
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    if len(shape) != 3 or any(int(value) < 2 for value in shape):
        raise ValueError("shape must contain three integers >= 2")
    axes = [
        np.linspace(minimum[axis], maximum[axis], int(shape[axis]), dtype=np.float64)
        for axis in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([component.ravel(order="C") for component in mesh])


def six_connected_neighbours(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    """Return face neighbours for a flattened C-order 3-D grid index."""
    nx, ny, nz = (int(value) for value in shape)
    if not 0 <= index < nx * ny * nz:
        raise IndexError(index)
    i, remainder = divmod(index, ny * nz)
    j, k = divmod(remainder, nz)
    neighbours: list[int] = []
    if i > 0:
        neighbours.append(index - ny * nz)
    if i + 1 < nx:
        neighbours.append(index + ny * nz)
    if j > 0:
        neighbours.append(index - nz)
    if j + 1 < ny:
        neighbours.append(index + nz)
    if k > 0:
        neighbours.append(index - 1)
    if k + 1 < nz:
        neighbours.append(index + 1)
    return tuple(neighbours)


@dataclass(slots=True)
class Domain:
    id: int
    matrix: FloatArray
    leaves: tuple[int, ...]
    neighbours: set[int] = field(default_factory=set)
    generation: int = 0
    active: bool = True

    @property
    def size(self) -> int:
        return len(self.leaves)


@dataclass(frozen=True, slots=True)
class DomainingResult:
    shape: tuple[int, int, int]
    grid_locations: FloatArray
    grid_domain_ids: IntArray
    domains: Mapping[int, Domain]
    merge_count: int
    real_domain_ids: IntArray | None = None


class GridSeededDomainer:
    """Deterministic Leapfrog-style adjacency-constrained region grower."""

    def __init__(
        self,
        consistency_thresh: float = 0.6,
        max_points: int | None = None,
        consistency_fn: ConsistencyFunction = reciprocal_determinant_consistency,
    ) -> None:
        if not np.isfinite(consistency_thresh):
            raise ValueError("consistency_thresh must be finite")
        if max_points is not None and max_points <= 0:
            raise ValueError("max_points must be positive when provided")
        self.consistency_thresh = float(consistency_thresh)
        self.max_points = max_points
        self.consistency_fn = consistency_fn

    def fit(
        self,
        grid_matrices: ArrayLike,
        shape: Sequence[int],
        grid_locations: ArrayLike | None = None,
        real_locations: ArrayLike | None = None,
    ) -> DomainingResult:
        matrices = np.asarray(grid_matrices, dtype=np.float64)
        grid_shape = tuple(int(value) for value in shape)
        expected = int(np.prod(grid_shape))
        if matrices.shape != (expected, 3, 3):
            raise ValueError(
                f"grid_matrices must have shape ({expected}, 3, 3); got {matrices.shape}"
            )
        matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))

        if grid_locations is None:
            locations = np.column_stack(
                np.unravel_index(np.arange(expected), grid_shape)
            ).astype(np.float64)
        else:
            locations = np.asarray(grid_locations, dtype=np.float64)
            if locations.shape != (expected, 3):
                raise ValueError(
                    f"grid_locations must have shape ({expected}, 3); got {locations.shape}"
                )

        domains: dict[int, Domain] = {}
        leaf_owner = np.arange(expected, dtype=np.int64)
        for index in range(expected):
            domains[index] = Domain(
                id=index,
                matrix=matrices[index].copy(),
                leaves=(index,),
                neighbours=set(six_connected_neighbours(index, grid_shape)),
            )

        # (-consistency, low_id, high_id, gen_low, gen_high). IDs provide
        # deterministic tie handling while generations invalidate stale pairs.
        heap: list[tuple[float, int, int, int, int]] = []
        for domain_id, domain in domains.items():
            for neighbour_id in domain.neighbours:
                if domain_id < neighbour_id:
                    self._push_pair(heap, domain, domains[neighbour_id])

        next_domain_id = expected
        merge_count = 0
        while heap:
            neg_consistency, first_id, second_id, first_gen, second_gen = heapq.heappop(heap)
            first = domains.get(first_id)
            second = domains.get(second_id)
            if first is None or second is None or not first.active or not second.active:
                continue
            if first.generation != first_gen or second.generation != second_gen:
                continue
            if second_id not in first.neighbours or first_id not in second.neighbours:
                continue
            consistency = -neg_consistency
            if consistency < self.consistency_thresh:
                break
            if self.max_points is not None and first.size + second.size > self.max_points:
                continue

            merged_matrix = weighted_mean_matrix(
                first.matrix, first.size, second.matrix, second.size
            )
            merged_neighbours = (first.neighbours | second.neighbours) - {
                first_id,
                second_id,
            }
            merged = Domain(
                id=next_domain_id,
                matrix=merged_matrix,
                leaves=tuple(itertools.chain(first.leaves, second.leaves)),
                neighbours=set(),
            )
            next_domain_id += 1
            merge_count += 1

            first.active = False
            second.active = False
            first.generation += 1
            second.generation += 1

            for neighbour_id in sorted(merged_neighbours):
                neighbour = domains.get(neighbour_id)
                if neighbour is None or not neighbour.active:
                    continue
                neighbour.neighbours.discard(first_id)
                neighbour.neighbours.discard(second_id)
                neighbour.neighbours.add(merged.id)
                neighbour.generation += 1
                merged.neighbours.add(neighbour_id)

            domains[merged.id] = merged
            leaf_owner[np.asarray(merged.leaves, dtype=np.int64)] = merged.id
            for neighbour_id in sorted(merged.neighbours):
                self._push_pair(heap, merged, domains[neighbour_id])

        active_domains = {key: value for key, value in domains.items() if value.active}
        real_domain_ids: IntArray | None = None
        if real_locations is not None:
            real_domain_ids = assign_real_locations(
                np.asarray(real_locations, dtype=np.float64), locations, leaf_owner
            )
        return DomainingResult(
            shape=grid_shape,
            grid_locations=locations,
            grid_domain_ids=leaf_owner,
            domains=active_domains,
            merge_count=merge_count,
            real_domain_ids=real_domain_ids,
        )

    def _push_pair(
        self,
        heap: list[tuple[float, int, int, int, int]],
        first: Domain,
        second: Domain,
    ) -> None:
        if self.max_points is not None and first.size + second.size > self.max_points:
            return
        merged_matrix = weighted_mean_matrix(
            first.matrix, first.size, second.matrix, second.size
        )
        consistency = self.consistency_fn(merged_matrix)
        if not np.isfinite(consistency):
            return
        low, high = (first, second) if first.id < second.id else (second, first)
        heapq.heappush(
            heap,
            (-float(consistency), low.id, high.id, low.generation, high.generation),
        )


def assign_real_locations(
    real_locations: ArrayLike,
    grid_locations: ArrayLike,
    grid_domain_ids: ArrayLike,
    *,
    chunk_size: int = 4096,
) -> IntArray:
    """Assign real locations to the nearest grid seed's final domain."""
    real = np.asarray(real_locations, dtype=np.float64)
    grid = np.asarray(grid_locations, dtype=np.float64)
    owners = np.asarray(grid_domain_ids, dtype=np.int64)
    if real.ndim != 2 or real.shape[1] != 3:
        raise ValueError("real_locations must have shape (n, 3)")
    if grid.ndim != 2 or grid.shape[1] != 3 or owners.shape != (len(grid),):
        raise ValueError("grid_locations/grid_domain_ids shapes are inconsistent")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    result = np.empty(len(real), dtype=np.int64)
    for start in range(0, len(real), chunk_size):
        stop = min(start + chunk_size, len(real))
        delta = real[start:stop, None, :] - grid[None, :, :]
        distances_sq = np.einsum("mni,mni->mn", delta, delta, optimize=True)
        nearest = np.argmin(distances_sq, axis=1)
        result[start:stop] = owners[nearest]
    return result


__all__ = [
    "Domain",
    "DomainingResult",
    "GridSeededDomainer",
    "assign_real_locations",
    "calculate_grid_shape",
    "create_grid_locations",
    "reciprocal_determinant_consistency",
    "six_connected_neighbours",
    "symmetric_determinant",
    "weighted_mean_matrix",
]
