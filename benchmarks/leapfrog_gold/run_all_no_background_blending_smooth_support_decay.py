"""Run every Leapfrog gold strength/range case with the best smooth-support field.

This batch runner applies the same settings that produced the successful S5_R300 result:

* automatic finite LVA-geodesic domains;
* StructuralInterpolant3 background_blending=False;
* the fitted structural field unchanged near input data;
* cubic nearest-data support decay toward OUTSIDE_VALUE from 0.60 to 1.00 of
  BASE_RANGE; and
* the aligned global scalar-grid mesher.

By default every available S<strength>_R<range>.obj case is generated.  To run only a
subset, set POLATORY_BENCHMARK_CASES to a comma-separated list such as
``S2_R50,S3_R300,S5_R500``.  The support-decay fractions can still be overridden with
POLATORY_SUPPORT_DECAY_START_FRACTION and POLATORY_SUPPORT_DECAY_END_FRACTION.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polatory
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Installs StructuralInterpolant3 background_blending=False and exposes the unfiltered
# real-gold suite.  Unlike the selected-case runner, this retains all available cases.
import run_no_background_blending_diagnostic as diagnostic  # noqa: E402

suite = diagnostic.suite
_BASE_FACTORY = polatory.StructuralInterpolant3
_START_FRACTION = float(
    os.environ.get("POLATORY_SUPPORT_DECAY_START_FRACTION", "0.60")
)
_END_FRACTION = float(
    os.environ.get("POLATORY_SUPPORT_DECAY_END_FRACTION", "1.00")
)

if not 0.0 <= _START_FRACTION < _END_FRACTION:
    raise ValueError(
        "Support-decay fractions must satisfy 0 <= start < end; got "
        f"{_START_FRACTION:g} and {_END_FRACTION:g}."
    )

_DECAY_START = _START_FRACTION * float(suite.BASE_RANGE)
_DECAY_END = _END_FRACTION * float(suite.BASE_RANGE)
_OUTSIDE_VALUE = float(suite.OUTSIDE_VALUE)
_ORIGINAL_AVAILABLE_CASES = suite.available_cases


class _SmoothSupportDecayInterpolant:
    """Delegate fitting while smoothly completing unsupported field regions."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._tree: cKDTree | None = None

    def fit(self, points: Any, *args: Any, **kwargs: Any):
        fit_points = np.asarray(points, dtype=np.float64)
        if fit_points.ndim != 2 or fit_points.shape[1] != 3:
            raise ValueError("Structural fit points must have shape (n, 3).")
        self._tree = cKDTree(fit_points)
        return self._wrapped.fit(points, *args, **kwargs)

    def evaluate(self, points: Any, *args: Any, **kwargs: Any):
        query = np.asarray(points, dtype=np.float64)
        values = np.asarray(
            self._wrapped.evaluate(points, *args, **kwargs),
            dtype=np.float64,
        )
        original_shape = values.shape
        flat_values = values.reshape(-1)

        if self._tree is None:
            return values
        if query.ndim != 2 or query.shape[1] != 3 or len(query) != len(flat_values):
            raise ValueError(
                "Structural evaluation points must have shape (m, 3) and match the "
                "number of returned values."
            )

        distances = np.asarray(self._tree.query(query, k=1)[0], dtype=np.float64)
        t = np.clip(
            (distances - _DECAY_START) / (_DECAY_END - _DECAY_START),
            0.0,
            1.0,
        )
        # Cubic smoothstep gives zero slope at both support-transition limits.
        weight = t * t * (3.0 - 2.0 * t)
        adjusted = (1.0 - weight) * flat_values + weight * _OUTSIDE_VALUE

        # Exact zero is ambiguous for marching cubes.  Bias exact zeros only; all nonzero
        # values remain governed by the fitted field and smooth support completion.
        exact_zero = adjusted == 0.0
        if np.any(exact_zero):
            nonzero = np.abs(adjusted[~exact_zero])
            scale = max(1.0, float(np.max(nonzero)) if nonzero.size else 1.0)
            adjusted = adjusted.copy()
            adjusted[exact_zero] = -1.0e-6 * scale

        return adjusted.reshape(original_shape)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _smooth_support_factory(*args: Any, **kwargs: Any):
    return _SmoothSupportDecayInterpolant(_BASE_FACTORY(*args, **kwargs))


def _requested_cases():
    """Return all cases, or an exact comma-separated subset when requested."""
    previous_max_cases = suite.MAX_CASES
    suite.MAX_CASES = 0
    try:
        cases = _ORIGINAL_AVAILABLE_CASES()
    finally:
        suite.MAX_CASES = previous_max_cases

    requested_text = os.environ.get("POLATORY_BENCHMARK_CASES", "").strip()
    if not requested_text:
        return cases

    requested = {
        name.strip().upper()
        for name in requested_text.split(",")
        if name.strip()
    }
    matches = [case for case in cases if case.name.upper() in requested]
    found = {case.name.upper() for case in matches}
    missing = sorted(requested.difference(found))
    if missing:
        available = ", ".join(case.name for case in cases)
        raise ValueError(
            f"Unknown Leapfrog benchmark case(s): {', '.join(missing)}. "
            f"Available cases: {available}"
        )
    return matches


polatory.StructuralInterpolant3 = _smooth_support_factory
suite.available_cases = _requested_cases
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "all-no-background-blending-smooth-support-decay"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tBatch mode: background_blending=False with smooth nearest-data "
    f"support decay from {_DECAY_START:g} m to {_DECAY_END:g} m; all requested "
    "strength/range cases use identical settings apart from their encoded strength and "
    "trend range.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
