"""Fresh-process worker used by the v10 Polatory LVA GUI launcher.

The native structural fitting stack is executed in this short-lived interpreter.
If the native code aborts, only this worker exits; the Qt GUI remains alive and
reports the child-process exit code.

The user-defined model extent is authoritative for the complete structural
workflow.  The automatic LVA centroid grid is sampled across that extent instead
of the interpolation-point bounds.  Unpopulated centroid cells are assigned to
the nearest populated structural domain, and local domain boxes are rebuilt from
those full-extent regions with an external one-cell blending halo.  The halo is
used only for stable interpolation weights; meshing and display remain clipped to
the exact user extent.

Contact-category runs also rebuild each structural domain so every Contact point
with a non-zero blend weight is included in that domain's support set.  This makes
the blended structural field honour Contact == 0 instead of blending in local
functions that never fitted the contact.
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
_CONTACT_POINTS = np.empty((0, 3), dtype=float)
_CONTACT_INDICES = np.empty(0, dtype=np.int64)
_USER_BBOX_MIN = np.zeros(3, dtype=float)
_USER_BBOX_MAX = np.ones(3, dtype=float)


def _strictly_inside_box(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Match StructuralInterpolant3's positive box-weight condition."""
    return np.all(points > bbox_min[None, :], axis=1) & np.all(
        points < bbox_max[None, :], axis=1
    )


def _domain_specs(domains: list[Any]) -> list[dict[str, Any]]:
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


def _fill_unassigned_centroids(
    centroid_points: np.ndarray,
    centroid_labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Assign every user-extent centroid cell to a populated domain."""
    centroid_points = np.asarray(centroid_points, dtype=float)
    labels = np.asarray(centroid_labels, dtype=np.int64).copy()
    missing = labels < 0
    missing_count = int(np.count_nonzero(missing))
    if missing_count == 0:
        return labels, 0

    populated = ~missing
    if not np.any(populated):
        raise RuntimeError(
            "The automatic SubDomainer produced no populated structural domains."
        )

    populated_points = centroid_points[populated]
    populated_labels = labels[populated]
    query_points = centroid_points[missing]

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        nearest = np.empty(len(query_points), dtype=np.int64)
        best = np.full(len(query_points), np.inf)
        for start in range(0, len(populated_points), 512):
            stop = min(start + 512, len(populated_points))
            difference = (
                query_points[:, None, :] - populated_points[None, start:stop, :]
            )
            squared = np.einsum(
                "qpi,qpi->qp", difference, difference, optimize=True
            )
            local = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(len(query_points)), local]
            replace = local_squared < best
            best[replace] = local_squared[replace]
            nearest[replace] = start + local[replace]
    else:
        nearest = np.asarray(
            cKDTree(populated_points).query(query_points, k=1)[1],
            dtype=np.int64,
        )

    labels[missing] = populated_labels[nearest]
    return labels, missing_count


def _apply_centroid_region_coverage(
    specs: list[dict[str, Any]],
    centroid_points: np.ndarray,
    centroid_labels: np.ndarray,
    shape: tuple[int, int, int],
    user_min: np.ndarray,
    user_max: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Cover the full user extent and keep weights positive on its six faces."""
    if not specs:
        raise RuntimeError("No structural domains were produced.")

    user_min = np.asarray(user_min, dtype=float)
    user_max = np.asarray(user_max, dtype=float)
    span = user_max - user_min
    shape_array = np.asarray(shape, dtype=float)
    cell_width = span / np.maximum(shape_array, 1.0)
    cell_width = np.maximum(cell_width, np.finfo(float).eps)

    # One centroid cell outside the requested extent prevents every local box
    # weight from becoming exactly zero on the meshing boundary.  Output is still
    # evaluated and extracted only inside [user_min, user_max].
    coverage_min = user_min - cell_width
    coverage_max = user_max + cell_width
    half_cell = 0.5 * cell_width
    overlap = cell_width
    tolerance = max(1.0e-9 * float(np.linalg.norm(span)), 1.0e-9)
    changed_faces = 0

    for label, spec in enumerate(specs):
        region = np.asarray(centroid_points, dtype=float)[centroid_labels == label]
        if len(region) == 0:
            raise RuntimeError(
                f"Structural domain {label} owns no centroid cells in the user extent."
            )

        core_min = np.maximum(region.min(axis=0) - half_cell, user_min)
        core_max = np.minimum(region.max(axis=0) + half_cell, user_max)
        region_min = core_min - overlap
        region_max = core_max + overlap

        touches_min = core_min <= user_min + tolerance
        touches_max = core_max >= user_max - tolerance
        region_min[touches_min] = coverage_min[touches_min]
        region_max[touches_max] = coverage_max[touches_max]

        old_min = spec["bbox_min"].copy()
        old_max = spec["bbox_max"].copy()
        new_min = np.maximum(np.minimum(old_min, region_min), coverage_min)
        new_max = np.minimum(np.maximum(old_max, region_max), coverage_max)

        if not np.all(new_max > new_min):
            raise RuntimeError(
                f"Structural domain {label} has an invalid full-extent coverage box."
            )

        changed_faces += int(np.count_nonzero(np.abs(new_min - old_min) > tolerance))
        changed_faces += int(np.count_nonzero(np.abs(new_max - old_max) > tolerance))
        spec["bbox_min"] = new_min
        spec["bbox_max"] = new_max

    return changed_faces, coverage_min, coverage_max


def _augment_specs_with_contacts(
    specs: list[dict[str, Any]],
    points: np.ndarray,
    contact_points: np.ndarray,
    contact_indices: np.ndarray,
    coverage_min: np.ndarray,
    coverage_max: np.ndarray,
) -> tuple[int, int]:
    if len(contact_points) == 0:
        return 0, 0

    points = np.asarray(points, dtype=float)
    contact_points = np.asarray(contact_points, dtype=float)
    contact_indices = np.asarray(contact_indices, dtype=np.int64)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    epsilon = max(1.0e-10 * diagonal, 1.0e-9)

    for contact in contact_points:
        covered = any(
            bool(
                _strictly_inside_box(
                    contact.reshape(1, 3),
                    spec["bbox_min"],
                    spec["bbox_max"],
                )[0]
            )
            for spec in specs
        )
        if covered:
            continue

        best_index = 0
        best_distance = float("inf")
        for index, spec in enumerate(specs):
            closest = np.minimum(
                np.maximum(contact, spec["bbox_min"]),
                spec["bbox_max"],
            )
            distance = float(np.linalg.norm(contact - closest))
            if distance < best_distance:
                best_distance = distance
                best_index = index

        selected = specs[best_index]
        selected["bbox_min"] = np.maximum(
            np.minimum(selected["bbox_min"], contact - epsilon),
            coverage_min,
        )
        selected["bbox_max"] = np.minimum(
            np.maximum(selected["bbox_max"], contact + epsilon),
            coverage_max,
        )

    total_added = 0
    active_pairs = 0
    for spec in specs:
        active = _strictly_inside_box(
            contact_points,
            spec["bbox_min"],
            spec["bbox_max"],
        )
        active_indices = contact_indices[active]
        active_pairs += int(np.count_nonzero(active))

        original_support = np.unique(spec["support_indices"])
        support = np.unique(
            np.concatenate([original_support, active_indices])
        ).astype(np.int64)
        total_added += int(len(support) - len(original_support))
        spec["support_indices"] = support

    return total_added, active_pairs


def _rebuild_domains(specs: list[dict[str, Any]]) -> list[Any]:
    rebuilt: list[Any] = []
    for spec in specs:
        rebuilt.append(
            polatory.StructuralDomain3(
                spec["anisotropy"],
                spec["bbox_min"],
                spec["bbox_max"],
                np.asarray(spec["support_indices"], dtype=np.int64).tolist(),
                spec["model_parameters"],
            )
        )
    return rebuilt


class UserExtentAutomaticBuilder:
    """Automatic SubDomainer whose centroid grid is the exact user extent."""

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

        if len(inputs) == 1:
            non_decaying = trend_type == polatory.StructuralTrendType.NON_DECAYING
            point_anisotropies = polatory.sample_single_input_anisotropies3(
                points,
                inputs[0],
                non_decaying=non_decaying,
            )
        else:
            samples = polatory.StructuralDomainBuilder3().sample(
                points,
                inputs,
                trend_type,
            )
            point_anisotropies = np.asarray(samples.anisotropies, dtype=float)

        minimum = _USER_BBOX_MIN.copy()
        maximum = _USER_BBOX_MAX.copy()
        spans = maximum - minimum
        active_axes = spans > max(float(spans.max()), 1.0) * 1.0e-12
        if not np.any(active_axes):
            active_axes[:] = True

        shape = automatic_builder_module._factor_grid_shape(
            self._builder.centroid_count,
            spans,
            active_axes,
        )
        centroid_points, _ = automatic_builder_module._grid_centroids(
            minimum,
            maximum,
            shape,
        )

        if len(inputs) == 1:
            centroid_anisotropies = polatory.sample_single_input_anisotropies3(
                centroid_points,
                inputs[0],
                non_decaying=trend_type
                == polatory.StructuralTrendType.NON_DECAYING,
            )
        else:
            centroid_samples = polatory.StructuralDomainBuilder3().sample(
                centroid_points,
                inputs,
                trend_type,
            )
            centroid_anisotropies = np.asarray(
                centroid_samples.anisotropies,
                dtype=float,
            )

        (
            labels,
            centroid_labels,
            minimum_points,
            maximum_points,
            merge_count,
        ) = self._builder._automatic_labels(
            points,
            centroid_anisotropies,
            minimum,
            maximum,
            shape,
        )

        postcluster_builder = automatic_builder_module.LabeledStructuralDomainBuilder3(
            base_range=self._builder.base_range,
            support_multiplier=self._builder.support_multiplier,
            minimum_support_points=self._builder.minimum_support_points,
        )
        domains = postcluster_builder.build(
            points,
            labels,
            point_anisotropies,
            model_parameters,
        )
        postcluster = tuple(postcluster_builder.diagnostics_ or ())

        filled_centroid_labels, filled_count = _fill_unassigned_centroids(
            centroid_points,
            centroid_labels,
        )
        specs = _domain_specs(list(domains))
        changed_faces, coverage_min, coverage_max = _apply_centroid_region_coverage(
            specs,
            centroid_points,
            filled_centroid_labels,
            shape,
            minimum,
            maximum,
        )
        added, active_pairs = _augment_specs_with_contacts(
            specs,
            points,
            _CONTACT_POINTS,
            _CONTACT_INDICES,
            coverage_min,
            coverage_max,
        )
        domains = _rebuild_domains(specs)

        self._builder.labels_ = labels.copy()
        self._builder.centroid_labels_ = filled_centroid_labels.copy()
        self._builder.centroid_points_ = centroid_points.copy()
        self._builder.centroid_grid_shape_ = shape
        self._builder.active_axes_ = active_axes.copy()
        self._builder.diagnostics_ = (
            automatic_builder_module.AutomaticStructuralDomainDiagnostics3(
                labels=labels.copy(),
                centroid_points=centroid_points.copy(),
                centroid_labels=filled_centroid_labels.copy(),
                centroid_grid_shape=shape,
                active_axes=active_axes.copy(),
                minimum_points=minimum_points,
                maximum_points=maximum_points,
                consistency_threshold=self._builder.consistency_threshold,
                merge_count=merge_count,
                final_domain_count=len(domains),
                postcluster=postcluster,
            )
        )

        print(
            "PROGRESS\t"
            "Sampled the automatic LVA centroid grid on the exact user-defined "
            f"extent with shape {shape}.",
            flush=True,
        )
        if filled_count:
            print(
                "PROGRESS\t"
                f"Assigned {filled_count:,} unpopulated centroid cells to their "
                "nearest populated structural domain.",
                flush=True,
            )
        print(
            "PROGRESS\t"
            "Rebuilt structural RBF coverage from the full user-extent centroid "
            f"regions ({changed_faces:,} domain faces adjusted; one-cell external "
            "weighting halo).",
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
    global _CONTACT_POINTS, _CONTACT_INDICES, _USER_BBOX_MIN, _USER_BBOX_MAX

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
        raise ValueError(
            "The user-defined extent must contain three minima and maxima."
        )
    if not np.all(_USER_BBOX_MAX > _USER_BBOX_MIN):
        raise ValueError(
            "Every user-defined extent maximum must exceed its minimum."
        )

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

    contact_mask = np.asarray(payload["indicators"], dtype=float) == 0.0
    _CONTACT_INDICES = np.flatnonzero(contact_mask).astype(np.int64)
    _CONTACT_POINTS = points[_CONTACT_INDICES]

    polatory.AutomaticStructuralDomainBuilder3 = UserExtentAutomaticBuilder

    holder: dict[str, Any] = {}

    def progress(message: Any) -> None:
        print(f"PROGRESS\t{message}", flush=True)

    def finished(result: Any) -> None:
        holder["result"] = result

    def failed(details: Any) -> None:
        holder["error"] = str(details)

    runner = SimpleNamespace(
        payload=payload,
        progress=CallbackSignal(progress),
        finished=CallbackSignal(finished),
        failed=CallbackSignal(failed),
    )

    worker_run = getattr(v8.v5, "scalable_worker_run", None)
    if not callable(worker_run):
        raise RuntimeError(
            "The v5 launcher does not expose scalable_worker_run. Pull the full "
            "examples launcher chain and retry."
        )
    worker_run(runner)

    if "error" in holder:
        print(holder["error"], file=sys.stderr, flush=True)
        return 2
    if "result" not in holder:
        print(
            "Worker returned neither a result nor an error.",
            file=sys.stderr,
            flush=True,
        )
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
