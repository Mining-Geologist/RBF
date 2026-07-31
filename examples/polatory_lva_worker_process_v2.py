"""Fast isolated worker for the v10 Polatory LVA GUI.

The automatic SubDomainer is fitted only across the populated data envelope, where
its clustering controls are meaningful and fast.  Its resulting domain labels are
then propagated across a separate centroid grid covering the exact user-defined
extent.  Local structural-domain blend boxes are rebuilt from that propagated grid
with a one-cell external halo, so the RBF and LVA field cover the requested extent
without forcing thousands of empty centroid cells through agglomeration.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import polatory
import polatory.automatic_domain_builder as automatic_builder_module

import polatory_lva_pyqt_app_v8_category_contacts as v8


class CallbackSignal:
    def __init__(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback

    def emit(self, value: Any = None) -> None:
        self._callback(value)


_ORIGINAL_AUTOMATIC_BUILDER = polatory.AutomaticStructuralDomainBuilder3
_USER_BBOX_MIN = np.zeros(3, dtype=float)
_USER_BBOX_MAX = np.ones(3, dtype=float)
_CONTACT_POINTS = np.empty((0, 3), dtype=float)
_CONTACT_INDICES = np.empty(0, dtype=np.int64)


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        nearest = np.empty(len(query), dtype=np.int64)
        best = np.full(len(query), np.inf)
        for start in range(0, len(reference), 512):
            stop = min(start + 512, len(reference))
            difference = query[:, None, :] - reference[None, start:stop, :]
            squared = np.einsum("qpi,qpi->qp", difference, difference, optimize=True)
            local = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(len(query)), local]
            replace = local_squared < best
            best[replace] = local_squared[replace]
            nearest[replace] = start + local[replace]
        return nearest
    return np.asarray(cKDTree(reference).query(query, k=1)[1], dtype=np.int64)


def _strictly_inside(
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    return np.all(points > minimum[None, :], axis=1) & np.all(
        points < maximum[None, :], axis=1
    )


def _domain_specs(domains: Sequence[Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for domain in domains:
        specs.append(
            {
                "anisotropy": np.asarray(domain.anisotropy, dtype=float),
                "bbox_min": np.asarray(domain.bbox_min, dtype=float).copy(),
                "bbox_max": np.asarray(domain.bbox_max, dtype=float).copy(),
                "support_indices": np.asarray(
                    domain.support_indices, dtype=np.int64
                ).copy(),
                "model_parameters": np.asarray(
                    domain.model_parameters, dtype=float
                ).reshape(-1).tolist(),
            }
        )
    return specs


def _ensure_every_domain_has_cells(
    centroid_points: np.ndarray,
    centroid_labels: np.ndarray,
    points: np.ndarray,
    point_labels: np.ndarray,
    domain_count: int,
) -> np.ndarray:
    labels = np.asarray(centroid_labels, dtype=np.int64).copy()
    used_cells: set[int] = set()
    for label in range(domain_count):
        existing = np.flatnonzero(labels == label)
        if len(existing):
            used_cells.update(int(index) for index in existing[:1])
            continue
        owned = points[point_labels == label]
        if len(owned) == 0:
            raise RuntimeError(f"Automatic structural domain {label} owns no data points.")
        centre = owned.mean(axis=0)
        order = np.argsort(np.sum((centroid_points - centre[None, :]) ** 2, axis=1))
        chosen = next((int(index) for index in order if int(index) not in used_cells), int(order[0]))
        labels[chosen] = label
        used_cells.add(chosen)
    return labels


def _rebuild_full_extent_domains(
    domains: Sequence[Any],
    points: np.ndarray,
    point_labels: np.ndarray,
    centroid_points: np.ndarray,
    centroid_labels: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[list[Any], int, int, int]:
    specs = _domain_specs(domains)
    domain_count = len(specs)
    centroid_labels = _ensure_every_domain_has_cells(
        centroid_points,
        centroid_labels,
        points,
        point_labels,
        domain_count,
    )

    span = _USER_BBOX_MAX - _USER_BBOX_MIN
    cell_width = span / np.maximum(np.asarray(shape, dtype=float), 1.0)
    cell_width = np.maximum(cell_width, np.finfo(float).eps)
    half_cell = 0.5 * cell_width
    overlap = cell_width
    coverage_min = _USER_BBOX_MIN - cell_width
    coverage_max = _USER_BBOX_MAX + cell_width
    tolerance = max(1.0e-9 * float(np.linalg.norm(span)), 1.0e-9)
    changed_faces = 0

    for label, spec in enumerate(specs):
        region = centroid_points[centroid_labels == label]
        core_min = np.maximum(region.min(axis=0) - half_cell, _USER_BBOX_MIN)
        core_max = np.minimum(region.max(axis=0) + half_cell, _USER_BBOX_MAX)
        region_min = core_min - overlap
        region_max = core_max + overlap

        touches_min = core_min <= _USER_BBOX_MIN + tolerance
        touches_max = core_max >= _USER_BBOX_MAX - tolerance
        region_min[touches_min] = coverage_min[touches_min]
        region_max[touches_max] = coverage_max[touches_max]

        old_min = spec["bbox_min"].copy()
        old_max = spec["bbox_max"].copy()
        new_min = np.maximum(np.minimum(old_min, region_min), coverage_min)
        new_max = np.minimum(np.maximum(old_max, region_max), coverage_max)
        if not np.all(new_max > new_min):
            raise RuntimeError(f"Invalid full-extent box for structural domain {label}.")
        changed_faces += int(np.count_nonzero(np.abs(new_min - old_min) > tolerance))
        changed_faces += int(np.count_nonzero(np.abs(new_max - old_max) > tolerance))
        spec["bbox_min"] = new_min
        spec["bbox_max"] = new_max

    added_supports = 0
    active_pairs = 0
    if len(_CONTACT_POINTS):
        for spec in specs:
            active = _strictly_inside(
                _CONTACT_POINTS,
                spec["bbox_min"],
                spec["bbox_max"],
            )
            contact_indices = _CONTACT_INDICES[active]
            active_pairs += int(np.count_nonzero(active))
            original = np.unique(spec["support_indices"])
            support = np.unique(np.concatenate([original, contact_indices])).astype(np.int64)
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


class FastUserExtentAutomaticBuilder:
    """Cluster populated data first, then propagate domains to the user extent."""

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

        span = _USER_BBOX_MAX - _USER_BBOX_MIN
        active_axes = span > max(float(span.max()), 1.0) * 1.0e-12
        if not np.any(active_axes):
            active_axes[:] = True
        shape = automatic_builder_module._factor_grid_shape(
            self._builder.centroid_count,
            span,
            active_axes,
        )
        centroid_points, _ = automatic_builder_module._grid_centroids(
            _USER_BBOX_MIN,
            _USER_BBOX_MAX,
            shape,
        )

        # Propagate already-resolved structural domains to the complete requested
        # extent.  This is O(N log N), unlike agglomerating thousands of empty cells.
        nearest_points = _nearest_indices(points, centroid_points)
        centroid_labels = point_labels[nearest_points]

        domains, changed_faces, added, active_pairs = _rebuild_full_extent_domains(
            domains,
            points,
            point_labels,
            centroid_points,
            centroid_labels,
            shape,
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
            "PROGRESS\tPropagated the populated structural domains across the exact "
            f"user extent on centroid grid {shape} ({changed_faces:,} domain faces "
            "adjusted; one-cell external weighting halo).",
            flush=True,
        )
        if len(_CONTACT_POINTS):
            print(
                "PROGRESS\t"
                f"Injected {len(_CONTACT_POINTS):,} Contact constraints into "
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
    global _USER_BBOX_MIN, _USER_BBOX_MAX, _CONTACT_POINTS, _CONTACT_INDICES

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--obj", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    result_path = Path(args.result)
    output_obj = Path(args.obj)
    with input_path.open("rb") as stream:
        payload = pickle.load(stream)

    points = np.asarray(payload["points"], dtype=float)
    _USER_BBOX_MIN = np.asarray(payload["bbox_min"], dtype=float)
    _USER_BBOX_MAX = np.asarray(payload["bbox_max"], dtype=float)
    if _USER_BBOX_MIN.shape != (3,) or _USER_BBOX_MAX.shape != (3,):
        raise ValueError("The user-defined extent must contain three minima and maxima.")
    if not np.all(_USER_BBOX_MAX > _USER_BBOX_MIN):
        raise ValueError("Every user-defined extent maximum must exceed its minimum.")

    tolerance = max(
        1.0e-9 * float(np.linalg.norm(_USER_BBOX_MAX - _USER_BBOX_MIN)),
        1.0e-9,
    )
    outside = np.any(points < _USER_BBOX_MIN - tolerance, axis=1) | np.any(
        points > _USER_BBOX_MAX + tolerance,
        axis=1,
    )
    if np.any(outside):
        raise ValueError(
            f"The user-defined extent excludes {int(np.count_nonzero(outside)):,} "
            "mapped modelling points. Expand it before running the model."
        )

    indicators = np.asarray(payload["indicators"], dtype=float)
    _CONTACT_INDICES = np.flatnonzero(indicators == 0.0).astype(np.int64)
    _CONTACT_POINTS = points[_CONTACT_INDICES]
    polatory.AutomaticStructuralDomainBuilder3 = FastUserExtentAutomaticBuilder

    holder: dict[str, Any] = {}

    def progress(message: Any) -> None:
        print(f"PROGRESS\t{message}", flush=True)

    runner = SimpleNamespace(
        payload=payload,
        progress=CallbackSignal(progress),
        finished=CallbackSignal(lambda result: holder.__setitem__("result", result)),
        failed=CallbackSignal(lambda details: holder.__setitem__("error", str(details))),
    )

    worker_run = getattr(v8.v5, "scalable_worker_run", None)
    if not callable(worker_run):
        raise RuntimeError("The v5 launcher does not expose scalable_worker_run.")
    worker_run(runner)

    if "error" in holder:
        print(holder["error"], file=sys.stderr, flush=True)
        return 2
    if "result" not in holder:
        print("Worker returned neither a result nor an error.", file=sys.stderr, flush=True)
        return 3

    result = holder["result"]
    plane_points = result.get("lva_points")
    plane_dimension = getattr(plane_points, "dimension", None)
    if plane_dimension is not None:
        result["lva_plane_dimension"] = int(plane_dimension)
        result["lva_points"] = np.asarray(plane_points)

    source_obj = Path(result["temp_obj"])
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_obj, output_obj)
    try:
        source_obj.unlink()
    except OSError:
        pass

    result["temp_obj"] = str(output_obj)
    with result_path.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("RESULT_READY", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        os._exit(4)
