"""Finite LVA-geodesic domain coverage for the isolated Polatory worker.

Leapfrog's automatic SubDomainer does not make every local RBF domain active to the
model boundary.  The synthetic blending benchmarks show that changing only the model
extent leaves the output unchanged and that the zero surface closes roughly one local
spheroidal support radius beyond the populated observations.

This worker therefore keeps the recovered automatic point clustering and the original
Leapfrog-compatible local boxes, but allows each box to bend through a varying LVA
field by no more than that domain's recovered internal support radius.  In a constant
anisotropy field this reduces to the existing analytical box expansion.  Around folds
it can follow structural continuity without propagating a domain indefinitely through
the complete user extent.
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, Sequence

import numpy as np
import polatory
import polatory.automatic_domain_builder as automatic_builder_module

import polatory_lva_worker_process_v2 as v2


_ORIGINAL_AUTOMATIC_BUILDER = polatory.AutomaticStructuralDomainBuilder3
_MAX_PROPAGATION_CELLS = 100_000


def _sample_grid_anisotropies(
    points: np.ndarray,
    inputs: Sequence[object],
    trend_type: object,
) -> np.ndarray:
    inputs = list(inputs)
    if len(inputs) == 1:
        return np.asarray(
            polatory.sample_single_input_anisotropies3(
                points,
                inputs[0],
                non_decaying=(
                    trend_type == polatory.StructuralTrendType.NON_DECAYING
                ),
            ),
            dtype=float,
        )

    samples = polatory.StructuralDomainBuilder3().sample(
        points,
        inputs,
        trend_type,
    )
    return np.asarray(samples.anisotropies, dtype=float)


def _grid_neighbours(index: int, shape: tuple[int, int, int]):
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


def _bounded_geodesic_region(
    centroid_points: np.ndarray,
    anisotropies: np.ndarray,
    shape: tuple[int, int, int],
    seed_indices: np.ndarray,
    maximum_distance: float,
) -> np.ndarray:
    """Return cells within one local support radius of a domain's core points."""
    centroid_points = np.asarray(centroid_points, dtype=float)
    anisotropies = np.asarray(anisotropies, dtype=float)
    seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64))
    maximum_distance = float(maximum_distance)

    if anisotropies.shape != (len(centroid_points), 3, 3):
        raise ValueError("Propagation LVA samples must have shape (n, 3, 3).")
    if len(seed_indices) == 0:
        raise ValueError("Every structural domain needs at least one propagation seed.")
    if not maximum_distance > 0.0:
        raise ValueError("Every structural domain needs a positive internal radius.")

    distances = np.full(len(centroid_points), np.inf, dtype=float)
    queue: list[tuple[float, int]] = []
    for index in seed_indices:
        if index < 0 or index >= len(centroid_points):
            raise IndexError("A propagation seed is outside the centroid grid.")
        distances[index] = 0.0
        heappush(queue, (0.0, int(index)))

    tolerance = max(1.0e-12 * maximum_distance, 1.0e-12)
    while queue:
        current_distance, current = heappop(queue)
        if current_distance > distances[current] + tolerance:
            continue
        if current_distance > maximum_distance + tolerance:
            break

        for neighbour in _grid_neighbours(current, shape):
            delta = centroid_points[neighbour] - centroid_points[current]
            metric = 0.5 * (anisotropies[current] + anisotropies[neighbour])
            metric = 0.5 * (metric + metric.T)
            step = float(np.linalg.norm(delta @ metric))
            if not np.isfinite(step) or step <= 0.0:
                step = float(np.linalg.norm(delta))

            candidate = current_distance + step
            if candidate > maximum_distance + tolerance:
                continue
            if candidate < distances[neighbour] - tolerance:
                distances[neighbour] = candidate
                heappush(queue, (candidate, int(neighbour)))

    return distances <= maximum_distance + tolerance


def _propagation_grid(
    points: np.ndarray,
    domains: Sequence[Any],
    source_shape: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
    """Build an extent-independent grid around the recovered finite domain boxes."""
    specs = v2._domain_specs(domains)
    minimum = np.min(np.vstack([spec["bbox_min"] for spec in specs]), axis=0)
    maximum = np.max(np.vstack([spec["bbox_max"] for spec in specs]), axis=0)

    data_min = points.min(axis=0)
    data_max = points.max(axis=0)
    data_span = data_max - data_min
    source_shape_array = np.maximum(np.asarray(source_shape, dtype=np.int64), 1)

    widths = np.zeros(3, dtype=float)
    valid = (source_shape_array > 1) & (data_span > 0.0)
    widths[valid] = data_span[valid] / source_shape_array[valid]
    positive = widths[widths > 0.0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    widths[~valid] = fallback
    widths = np.maximum(widths, np.finfo(float).eps)

    span = maximum - minimum
    shape_array = np.maximum(np.ceil(span / widths).astype(np.int64), 1)
    total = int(np.prod(shape_array, dtype=np.int64))
    if total > _MAX_PROPAGATION_CELLS:
        factor = (total / float(_MAX_PROPAGATION_CELLS)) ** (1.0 / 3.0)
        widths *= factor
        shape_array = np.maximum(np.ceil(span / widths).astype(np.int64), 1)

    shape = tuple(int(value) for value in shape_array)
    centroid_points, _ = automatic_builder_module._grid_centroids(
        minimum,
        maximum,
        shape,
    )
    cell_width = span / np.maximum(shape_array.astype(float), 1.0)
    return centroid_points, shape, cell_width


def _rebuild_finite_domains(
    domains: Sequence[Any],
    points: np.ndarray,
    point_labels: np.ndarray,
    centroid_points: np.ndarray,
    centroid_anisotropies: np.ndarray,
    shape: tuple[int, int, int],
    internal_radii: np.ndarray,
) -> tuple[list[Any], int, int, int]:
    specs = v2._domain_specs(domains)
    if len(specs) != len(internal_radii):
        raise RuntimeError("Automatic domain diagnostics do not match domain count.")

    tolerance = max(
        1.0e-9 * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))),
        1.0e-9,
    )
    changed_faces = 0

    for label, (spec, internal_radius) in enumerate(zip(specs, internal_radii)):
        owned = points[point_labels == label]
        if len(owned) == 0:
            raise RuntimeError(f"Automatic structural domain {label} owns no data points.")

        seed_indices = v2._nearest_indices(centroid_points, owned)
        active = _bounded_geodesic_region(
            centroid_points,
            centroid_anisotropies,
            shape,
            seed_indices,
            float(internal_radius),
        )
        region = centroid_points[active]
        if len(region) == 0:
            raise RuntimeError(f"Structural domain {label} has no finite LVA region.")

        old_min = spec["bbox_min"].copy()
        old_max = spec["bbox_max"].copy()

        # The recovered analytical box remains authoritative in locally constant
        # fields.  Geodesic cells may enlarge it only where a curved LVA path reaches
        # farther within the same internal support radius.
        new_min = np.minimum(old_min, region.min(axis=0))
        new_max = np.maximum(old_max, region.max(axis=0))
        if not np.all(new_max > new_min):
            raise RuntimeError(f"Invalid finite LVA box for structural domain {label}.")

        changed_faces += int(np.count_nonzero(np.abs(new_min - old_min) > tolerance))
        changed_faces += int(np.count_nonzero(np.abs(new_max - old_max) > tolerance))
        spec["bbox_min"] = new_min
        spec["bbox_max"] = new_max

    added_supports = 0
    active_pairs = 0
    if len(v2._CONTACT_POINTS):
        for spec in specs:
            active = v2._strictly_inside(
                v2._CONTACT_POINTS,
                spec["bbox_min"],
                spec["bbox_max"],
            )
            contact_indices = v2._CONTACT_INDICES[active]
            active_pairs += int(np.count_nonzero(active))
            original = np.unique(spec["support_indices"])
            support = np.unique(
                np.concatenate([original, contact_indices])
            ).astype(np.int64)
            added_supports += int(len(support) - len(original))
            spec["support_indices"] = support

    rebuilt = [
        polatory.StructuralDomain3(
            spec["anisotropy"],
            spec["bbox_min"],
            spec["bbox_max"],
            np.asarray(spec["support_indices"], dtype=np.int64).tolist(),
            spec["model_parameters"],
        )
        for spec in specs
    ]
    return rebuilt, changed_faces, added_supports, active_pairs


class FiniteLvaGeodesicAutomaticBuilder:
    """Cluster input points, then curve each finite box by one support radius."""

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

        diagnostics = self._builder.diagnostics_
        if diagnostics is None:
            raise RuntimeError("Automatic SubDomainer diagnostics are unavailable.")
        internal_radii = np.asarray(
            [item.internal_radius for item in diagnostics.postcluster],
            dtype=float,
        )

        centroid_points, shape, _ = _propagation_grid(
            points,
            domains,
            self._builder.centroid_grid_shape_,
        )
        print(
            "PROGRESS\tSampling the structural LVA metric on a finite, "
            f"extent-independent propagation grid {shape}…",
            flush=True,
        )
        centroid_anisotropies = _sample_grid_anisotropies(
            centroid_points,
            inputs,
            trend_type,
        )

        print(
            "PROGRESS\tCurving each automatic domain through the LVA field, bounded "
            "by its recovered local support radius…",
            flush=True,
        )
        domains, changed_faces, added, active_pairs = _rebuild_finite_domains(
            domains,
            points,
            point_labels,
            centroid_points,
            centroid_anisotropies,
            shape,
            internal_radii,
        )

        print(
            "PROGRESS\tBuilt finite LVA-geodesic structural coverage "
            f"({changed_faces:,} domain faces extended; no domain was propagated "
            "to the model boundary).",
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
    v2.FastUserExtentAutomaticBuilder = FiniteLvaGeodesicAutomaticBuilder
    return v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
