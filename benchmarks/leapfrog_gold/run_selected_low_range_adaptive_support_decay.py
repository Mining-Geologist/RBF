"""Run one selected case with low-range-aware nearest-data support decay.

The successful smooth-support completion used a fixed 240-400 m transition derived from
BASE_RANGE=400 m.  That works well for R300/R500, but for R50/R100 the structural field
can form an unsupported basal lobe before the fixed transition becomes active.  This
controlled diagnostic preserves the proven 240-400 m behaviour for ranges >= 200 m and
only contracts the support transition for lower trend ranges.

The effective support radius is

    min(BASE_RANGE, max(0.5 * BASE_RANGE, 2.0 * trend_range))

and the smooth transition starts at 60% of that radius.  Consequently:

* R300/R500 remain exactly 240-400 m (unchanged from the successful solution);
* R100 uses 120-200 m;
* R50 also uses the conservative 120-200 m floor rather than an aggressive 30-50 m cut.

Select a case with POLATORY_BENCHMARK_CASE, for example S3_R100.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polatory
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Installs the selected-case filter and background_blending=False.
import run_selected_no_background_blending as selected  # noqa: E402

suite = selected.suite
_BASE_FACTORY = polatory.StructuralInterpolant3
_OUTSIDE_VALUE = float(suite.OUTSIDE_VALUE)

_REQUESTED_CASE = os.environ.get("POLATORY_BENCHMARK_CASE", "").strip().upper()
_MATCH = re.fullmatch(r"S\d+(?:\.\d+)?_R(?P<range>\d+(?:\.\d+)?)", _REQUESTED_CASE)
if _MATCH is None:
    raise ValueError(
        "POLATORY_BENCHMARK_CASE must look like S3_R100 or S5_R50; got "
        f"{_REQUESTED_CASE!r}."
    )

_TREND_RANGE = float(_MATCH.group("range"))
_BASE_RANGE = float(suite.BASE_RANGE)
_EFFECTIVE_SUPPORT_RADIUS = min(
    _BASE_RANGE,
    max(0.5 * _BASE_RANGE, 2.0 * _TREND_RANGE),
)
_DECAY_START = 0.60 * _EFFECTIVE_SUPPORT_RADIUS
_DECAY_END = _EFFECTIVE_SUPPORT_RADIUS


class _AdaptiveSupportDecayInterpolant:
    """Preserve the fitted field near data and complete only unsupported far field."""

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
        # Cubic smoothstep gives zero slope at both ends and avoids a hard support shell.
        weight = t * t * (3.0 - 2.0 * t)
        adjusted = (1.0 - weight) * flat_values + weight * _OUTSIDE_VALUE

        # Marching cubes should not interpret an exact-zero numerical plateau as surface.
        exact_zero = adjusted == 0.0
        if np.any(exact_zero):
            nonzero = np.abs(adjusted[~exact_zero])
            scale = max(1.0, float(np.max(nonzero)) if nonzero.size else 1.0)
            adjusted = adjusted.copy()
            adjusted[exact_zero] = -1.0e-6 * scale

        return adjusted.reshape(original_shape)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _adaptive_support_factory(*args: Any, **kwargs: Any):
    return _AdaptiveSupportDecayInterpolant(_BASE_FACTORY(*args, **kwargs))


polatory.StructuralInterpolant3 = _adaptive_support_factory
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-low-range-adaptive-support-decay"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: low-range adaptive nearest-data support decay; "
    f"case={_REQUESTED_CASE}, trend range={_TREND_RANGE:g} m, transition "
    f"{_DECAY_START:g}-{_DECAY_END:g} m. R300/R500 retain the proven 240-400 m "
    "transition unchanged.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
