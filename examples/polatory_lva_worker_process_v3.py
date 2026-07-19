"""Generic LVA-geodesic extent propagation for the isolated v10 worker.

The automatic SubDomainer still resolves its populated structural domains from the
input data, using the recovered 6000-centroid clustering controls.  To extend those
domains through the user-defined model extent, this worker does *not* use nearest
Euclidean input points.  Instead it:

1. samples the same LVA anisotropy field on the full user-extent centroid grid;
2. seeds that grid from the populated automatic-domain centroid labels; and
3. performs deterministic multi-source geodesic propagation on the 26-neighbour
   grid using ``||delta @ A||`` as edge cost, matching the anisotropic coordinate
   transform used by the recovered support-selection rule.

This is dataset-independent.  Strength and trend range control the sampled metric;
outside the trend influence the metric naturally returns to isotropic behaviour.
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, Sequence

import numpy as np
import polatory
import polatory.automatic_domain_builder as automatic_builder_module

import polatory_lva_worker_process_v2 as v2


_ORIGINAL_AUTOMATIC_BUILDER = polatory.AutomaticStructuralDomainBuilder3


def _sample_grid_anisotropies(
    centroid_points: np.ndarray,
    inputs: Sequence[object],
    trend_type: object,
) -> np.ndarray:
    inputs = list(inputs)
    if len(inputs) == 1:
        return np.asarray(
            polatory.sample_single_input_anisotropies3(
                centroid_points,
                inputs[0],
                non_decaying=(
                    trend_type == polatory.StructuralTrendType.NON_DECAYING
                ),
            ),
            dtype=float,
        )

    samples = polatory.StructuralDomainBuilder3().sample(
        centroid_points,
        inputs,
        trend_type,
    )
    return np.asarray(samples.anisotropies, dtype=float)


def _seed_full_grid(
    centroid_points: np.ndarray,
    source_centroids: np.ndarray,
    source_labels: np.ndarray,
    data_points: np.ndarray,
    point_labels: np.ndarray,
    domain_count: int,
) -> np.ndarray:
    """Create stable full-grid seeds from automatic domains and their core data."""
    centroid_points = np.asarray(centroid_points, dtype=float)
    source_centroids = np.asarray(source_centroids, dtype=float)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    data_points = np.asarray(data_points, dtype=float)
    point_labels = np.asarray(point_labels, dtype=np.int64)

    votes = np.zeros((len(centroid_points), domain_count), dtype=np.int32)

    valid = (source_labels >= 0) & (source_labels < domain_count)
    if np.any(valid):
        mapped = v2._nearest_indices(centroid_points, source_centroids[valid])
        np.add.at(votes, (mapped, source_labels[valid]), 1)

    # Core observations receive a stronger vote so a coarse user-extent grid does
    # not erase a small but valid automatic domain during source-cell collisions.
    mapped_points = v2._nearest_indices(centroid_points, data_points)
    np.add.at(votes, (mapped_points, point_labels), 4)

    totals = votes.sum(axis=1)
    seeds = np.full(len(centroid_points), -1, dtype=np.int64)
    seeded = totals > 0
    seeds[seeded] = np.argmax(votes[seeded], axis=1).astype(np.int64)

    # Guarantee at least one source cell per automatic domain.  This is a general
    # preservation rule, not a benchmark-specific location or label override.
    occupied: set[int] = set(int(index) for index in np.flatnonzero(seeded))
    for label in range(domain_count):
        if np.any(seeds == label):
            continue
        owned = data_points[point_labels == label]
        if len(owned) == 0:
            raise RuntimeError(
                f"Automatic structural domain {label} owns no interpolation points."
            )
        centre = owned.mean(axis=0)
        order = np.argsort(
            np.sum((centroid_points - centre[None, :]) ** 2, axis=1)
        )
        selected = next(
            (int(index) for index in order if int(index) not in occupied),
            int(order[0]),
        )
        seeds[selected] = label
        occupied.add(selected)

    return seeds


def _grid_neighbours(
    index: int,
    shape: tuple[int, int, int],
):
    nx, ny, nz = (int(value) for value in shape)
    yz = ny * nz
    ix = index // yz
    remainder = index - ix * yz
    iy = remainder // nz
    iz = remainder - iy * nz

    for dx in (-1, 0, 1):
        xx = ix + dx
        if xx < 0 or xx >= nx:
            continue
        for dy in (-1, 0, 1):
            yy = iy + dy
            if yy < 0 or yy >= ny:
                continue
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                zz = iz + dz
                if zz < 0 or zz >= nz:
                    continue
                yield (xx * ny + yy) * nz + zz


def _lva_geodesic_labels(
    centroid_points: np.ndarray,
    anisotropies: np.ndarray,
    shape: tuple[int, int, int],
    seeds: np.ndarray,
) -> np.ndarray:
    """Propagate domain labels along low-cost structural continuity directions."""
    centroid_points = np.asarray(centroid_points, dtype=float)
    anisotropies = np.asarray(anisotropies, dtype=float)
    seeds = np.asarray(seeds, dtype=np.int64)

    if anisotropies.shape != (len(centroid_points), 3, 3):
        raise ValueError("Full-extent LVA samples must have shape (n, 3, 3).")
    if seeds.shape != (len(centroid_points),):
        raise ValueError("Full-extent seed labels must have shape (n,).")

    distances = np.full(len(centroid_points), np.inf, dtype=float)
    labels = np.full(len(centroid_points), -1, dtype=np.int64)
    queue: list[tuple[float, int, int]] = []

    for index in np.flatnonzero(seeds >= 0):
        label = int(seeds[index])
        distances[index] = 0.0
        labels[index] = label
        heappush(queue, (0.0, label, int(index)))

    scale = max(
        float(np.linalg.norm(centroid_points.max(axis=0) - centroid_points.min(axis=0))),
        1.0,
    )
    tolerance = 1.0e-12 * scale

    while queue:
        current_distance, current_label, current = heappop(queue)
        if current_distance > distances[current] + tolerance:
            continue
        if current_label != labels[current]:
            continue

        for neighbour in _grid_neighbours(current, shape):
            delta = centroid_points[neighbour] - centroid_points[current]
            metric = 0.5 * (
                anisotropies[current] + anisotropies[neighbour]
            )
            metric = 0.5 * (metric + metric.T)
            step = float(np.linalg.norm(delta @ metric))
            if not np.isfinite(step) or step <= 0.0:
                step = float(np.linalg.norm(delta))

            candidate = current_distance + step
            better = candidate < distances[neighbour] - tolerance
            tied = abs(candidate - distances[neighbour]) <= tolerance
            deterministic_tie = tied and (
                labels[neighbour] < 0 or current_label < labels[neighbour]
            )
            if better or deterministic_tie:
                distances[neighbour] = candidate
                labels[neighbour] = current_label
                heappush(queue, (candidate, current_label, int(neighbour)))

    if np.any(labels < 0):
        raise RuntimeError(
            f"LVA-geodesic propagation left {int(np.count_nonzero(labels < 0)):,} "
            "centroid cells unassigned."
        )
    return labels


class LvaGeodesicUserExtentAutomaticBuilder:
    """Cluster populated data, then extend domains using the sampled LVA metric."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._builder = _ORIGINAL_AUTOMATIC_BUILDER(*args, **kwargs)

    def build_from_inputs(
        self,
        points: np.ndarray,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
        trend_type: object = polatory.StructuralTrendType.STRONGEST_ALONG_INPUTS,
    ) -> list[Any]:
        points = np.asarray(points, dtype=float)
        inputs = list(inputs)
        if not inputs:
            raise ValueError("inputs must not be empty")

        print(
            "PROGRESS\tClustering automatic structural domains inside the populated "
            "data envelope…",
            flush=True,
        )
        domains = list(
            self._builder.build_from_inputs(
                points,
                inputs,
                model_parameters,
                trend_type,
            )
        )
        point_labels = np.asarray(self._builder.labels_, dtype=np.int64)
        if point_labels.shape != (len(points),):
            raise RuntimeError("Automatic SubDomainer did not return one label per point.")

        source_centroids = np.asarray(
            self._builder.centroid_points_,
            dtype=float,
        )
        source_labels = np.asarray(
            self._builder.centroid_labels_,
            dtype=np.int64,
        )

        span = v2._USER_BBOX_MAX - v2._USER_BBOX_MIN
        active_axes = span > max(float(span.max()), 1.0) * 1.0e-12
        if not np.any(active_axes):
            active_axes[:] = True
        shape = automatic_builder_module._factor_grid_shape(
            self._builder.centroid_count,
            span,
            active_axes,
        )
        centroid_points, _ = automatic_builder_module._grid_centroids(
            v2._USER_BBOX_MIN,
            v2._USER_BBOX_MAX,
            shape,
        )

        print(
            "PROGRESS\tSampling the structural LVA metric across the exact user "
            f"extent on centroid grid {shape}…",
            flush=True,
        )
        centroid_anisotropies = _sample_grid_anisotropies(
            centroid_points,
            inputs,
            trend_type,
        )
        seeds = _seed_full_grid(
            centroid_points,
            source_centroids,
            source_labels,
            points,
            point_labels,
            len(domains),
        )

        print(
            "PROGRESS\tPropagating automatic domains through empty regions along "
            "LVA structural continuity…",
            flush=True,
        )
        centroid_labels = _lva_geodesic_labels(
            centroid_points,
            centroid_anisotropies,
            shape,
            seeds,
        )

        domains, changed_faces, added, active_pairs = (
            v2._rebuild_full_extent_domains(
                domains,
                points,
                point_labels,
                centroid_points,
                centroid_labels,
                shape,
            )
        )

        old_diagnostics = self._builder.diagnostics_
        if old_diagnostics is None:
            raise RuntimeError("Automatic SubDomainer diagnostics are unavailable.")
        self._builder.centroid_points_ = centroid_points.copy()
        self._builder.centroid_labels_ = centroid_labels.copy()
        self._builder.centroid_grid_shape_ = shape
        self._builder.active_axes_ = active_axes.copy()
        self._builder.diagnostics_ = (
            automatic_builder_module.AutomaticStructuralDomainDiagnostics3(
                labels=point_labels.copy(),
                centroid_points=centroid_points.copy(),
                centroid_labels=centroid_labels.copy(),
                centroid_grid_shape=shape,
                active_axes=active_axes.copy(),
                minimum_points=old_diagnostics.minimum_points,
                maximum_points=old_diagnostics.maximum_points,
                consistency_threshold=old_diagnostics.consistency_threshold,
                merge_count=old_diagnostics.merge_count,
                final_domain_count=len(domains),
                postcluster=old_diagnostics.postcluster,
            )
        )

        print(
            "PROGRESS\tRebuilt structural RBF coverage from LVA-geodesic full-extent "
            f"regions ({changed_faces:,} domain faces adjusted; one-cell external "
            "weighting halo).",
            flush=True,
        )
        if len(v2._CONTACT_POINTS):
            print(
                "PROGRESS\t"
                f"Injected {len(v2._CONTACT_POINTS):,} Contact constraints into "
                f"{active_pairs:,} active domain/contact overlaps "
                f"({added:,} added support references).",
                flush=True,
            )
        return domains

    @property
    def diagnostics_(self) -> Any:
        return self._builder.diagnostics_

    @property
    def labels_(self) -> Any:
        return self._builder.labels_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)


def main() -> int:
    v2.FastUserExtentAutomaticBuilder = LvaGeodesicUserExtentAutomaticBuilder
    return v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
