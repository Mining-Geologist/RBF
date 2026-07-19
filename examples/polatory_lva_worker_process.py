"""Fresh-process worker used by the v10 Polatory LVA GUI launcher.

The native structural fitting stack is executed in this short-lived interpreter.
If the native code aborts, only this worker exits; the Qt GUI remains alive and
reports the child-process exit code.

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


def _strictly_inside_box(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Match StructuralInterpolant3's positive box-weight condition."""
    return np.all(points > bbox_min[None, :], axis=1) & np.all(
        points < bbox_max[None, :], axis=1
    )


def _augment_domains_with_contacts(
    domains: list[Any],
    points: np.ndarray,
    contact_points: np.ndarray,
    contact_indices: np.ndarray,
) -> tuple[list[Any], int, int]:
    if len(contact_points) == 0:
        return list(domains), 0, 0

    points = np.asarray(points, dtype=float)
    contact_points = np.asarray(contact_points, dtype=float)
    contact_indices = np.asarray(contact_indices, dtype=np.int64)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    epsilon = max(1.0e-10 * diagonal, 1.0e-9)

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

    # A point exactly on every box face has zero blend weight. Expand the nearest
    # box by a tiny amount so each contact has at least one active local function.
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
        selected["bbox_min"] = np.minimum(
            selected["bbox_min"], contact - epsilon
        )
        selected["bbox_max"] = np.maximum(
            selected["bbox_max"], contact + epsilon
        )

    rebuilt: list[Any] = []
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

        rebuilt.append(
            polatory.StructuralDomain3(
                spec["anisotropy"],
                spec["bbox_min"],
                spec["bbox_max"],
                support.tolist(),
                spec["model_parameters"],
            )
        )

    return rebuilt, total_added, active_pairs


class ContactAwareAutomaticBuilder:
    """Proxy the native builder and make blended Contact constraints exact."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._builder = _ORIGINAL_AUTOMATIC_BUILDER(*args, **kwargs)

    def build_from_inputs(self, points: np.ndarray, *args: Any, **kwargs: Any):
        domains = self._builder.build_from_inputs(points, *args, **kwargs)
        rebuilt, added, active_pairs = _augment_domains_with_contacts(
            list(domains),
            np.asarray(points, dtype=float),
            _CONTACT_POINTS,
            _CONTACT_INDICES,
        )
        if len(_CONTACT_POINTS):
            print(
                "PROGRESS\t"
                f"Injected {len(_CONTACT_POINTS):,} Contact constraints into "
                f"{active_pairs:,} active domain/contact overlaps "
                f"({added:,} added support references).",
                flush=True,
            )
        return rebuilt

    @property
    def diagnostics_(self) -> Any:
        return self._builder.diagnostics_

    @property
    def labels_(self) -> Any:
        return self._builder.labels_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)


def main() -> int:
    global _CONTACT_POINTS, _CONTACT_INDICES

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
    contact_mask = np.asarray(payload["indicators"], dtype=float) == 0.0
    _CONTACT_INDICES = np.flatnonzero(contact_mask).astype(np.int64)
    _CONTACT_POINTS = points[_CONTACT_INDICES]

    if len(_CONTACT_POINTS):
        polatory.AutomaticStructuralDomainBuilder3 = ContactAwareAutomaticBuilder

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
