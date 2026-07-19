"""Fresh-process worker used by the v10 Polatory LVA GUI launcher.

The native structural fitting stack is executed in this short-lived interpreter.
If the native code aborts, only this worker exits; the Qt GUI remains alive and
reports the child-process exit code.

The user-defined model extent is authoritative. Automatic local structural-domain
boxes are clipped to that extent, and domains on the outside of the automatic
domain envelope are extended to the corresponding user-extent face. This prevents
the blended RBF from stopping at an internal data-derived domain boundary.

Contact-category runs also rebuild each structural domain so every Contact point
with a non-zero blend weight is included in that domain's support set. This makes
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
from typing import Any, Callable

import numpy as np
import polatory

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


def _apply_user_extent(
    specs: list[dict[str, Any]],
    user_min: np.ndarray,
    user_max: np.ndarray,
) -> int:
    """Make the user extent the outer boundary of all structural-domain coverage."""
    if not specs:
        return 0

    user_min = np.asarray(user_min, dtype=float)
    user_max = np.asarray(user_max, dtype=float)
    domain_min = np.vstack([spec["bbox_min"] for spec in specs]).min(axis=0)
    domain_max = np.vstack([spec["bbox_max"] for spec in specs]).max(axis=0)
    diagonal = float(np.linalg.norm(domain_max - domain_min))
    tolerance = max(1.0e-9 * diagonal, 1.0e-9)
    changed_faces = 0

    for spec in specs:
        old_min = spec["bbox_min"].copy()
        old_max = spec["bbox_max"].copy()
        new_min = np.maximum(old_min, user_min)
        new_max = np.minimum(old_max, user_max)

        for axis in range(3):
            if old_min[axis] <= domain_min[axis] + tolerance:
                new_min[axis] = user_min[axis]
            if old_max[axis] >= domain_max[axis] - tolerance:
                new_max[axis] = user_max[axis]

        if not np.all(new_max > new_min):
            raise ValueError(
                "The user-defined extent does not overlap every automatic structural "
                "domain. Expand the extent so it contains all mapped modelling points."
            )

        changed_faces += int(np.count_nonzero(np.abs(new_min - old_min) > tolerance))
        changed_faces += int(np.count_nonzero(np.abs(new_max - old_max) > tolerance))
        spec["bbox_min"] = new_min
        spec["bbox_max"] = new_max

    return changed_faces


def _augment_specs_with_contacts(
    specs: list[dict[str, Any]],
    points: np.ndarray,
    contact_points: np.ndarray,
    contact_indices: np.ndarray,
) -> tuple[int, int]:
    if len(contact_points) == 0:
        return 0, 0

    points = np.asarray(points, dtype=float)
    contact_points = np.asarray(contact_points, dtype=float)
    contact_indices = np.asarray(contact_indices, dtype=np.int64)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    epsilon = max(1.0e-10 * diagonal, 1.0e-9)

    # A point exactly on every box face has zero blend weight. Expand the nearest
    # box by a tiny amount, while staying inside the user's authoritative extent.
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
            _USER_BBOX_MIN,
        )
        selected["bbox_max"] = np.minimum(
            np.maximum(selected["bbox_max"], contact + epsilon),
            _USER_BBOX_MAX,
        )

    total_added = 0
    active_pairs = 0

    # Every domain that has a positive blend weight at a contact must fit that
    # same zero constraint. Otherwise the weighted blend is not interpolatory.
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
    """Proxy the native builder and enforce user extent plus Contact constraints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._builder = _ORIGINAL_AUTOMATIC_BUILDER(*args, **kwargs)

    def build_from_inputs(self, points: np.ndarray, *args: Any, **kwargs: Any):
        domains = list(self._builder.build_from_inputs(points, *args, **kwargs))
        specs = _domain_specs(domains)
        changed_faces = _apply_user_extent(
            specs,
            _USER_BBOX_MIN,
            _USER_BBOX_MAX,
        )
        added, active_pairs = _augment_specs_with_contacts(
            specs,
            np.asarray(points, dtype=float),
            _CONTACT_POINTS,
            _CONTACT_INDICES,
        )

        print(
            "PROGRESS\t"
            "Applied the user-defined extent to the structural RBF domain coverage "
            f"({changed_faces:,} domain faces adjusted).",
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
        return _rebuild_domains(specs)

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
        raise ValueError("The user-defined extent must contain three minima and maxima.")
    if not np.all(_USER_BBOX_MAX > _USER_BBOX_MIN):
        raise ValueError("Every user-defined extent maximum must exceed its minimum.")

    tolerance = max(
        1.0e-9 * float(np.linalg.norm(_USER_BBOX_MAX - _USER_BBOX_MIN)),
        1.0e-9,
    )
    outside = np.any(points < _USER_BBOX_MIN - tolerance, axis=1) | np.any(
        points > _USER_BBOX_MAX + tolerance, axis=1
    )
    if np.any(outside):
        raise ValueError(
            f"The user-defined extent excludes {int(np.count_nonzero(outside)):,} "
            "mapped modelling points. Expand it before running the model."
        )

    contact_mask = np.asarray(payload["indicators"], dtype=float) == 0.0
    _CONTACT_INDICES = np.flatnonzero(contact_mask).astype(np.int64)
    _CONTACT_POINTS = points[_CONTACT_INDICES]

    # Always proxy the builder: the user extent is authoritative even when there
    # are no Contact categories in the current dataset.
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

    # v5 names its safe worker entry point ``scalable_worker_run``. Importing
    # v8 above also installs category-only Contact value preprocessing.
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

    # ndarray-subclass attributes are not preserved by the default pickle path.
    # Store the v5 centre-plane dimension explicitly and reconstruct it in v10.
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
