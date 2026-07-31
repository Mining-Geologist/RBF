"""Run one selected no-background-blending case with smooth data-support decay.

The best current field uses StructuralInterpolant3 background_blending=False, which
preserves the Leapfrog-like shape near the data but can leave a long unsupported basal
lobe attached to the main component.  This diagnostic keeps the fitted structural field
exactly unchanged inside a data-support radius, then smoothly blends it toward the
outside value according to Euclidean distance from the nearest input point.  Unlike the
old per-domain background blending, the completion has no axis-aligned domain boxes and
therefore should close unsupported lobes without reintroducing flat shelves or walls.

Defaults are expressed as fractions of the base RBF range:
  start = 0.60 * BASE_RANGE
  end   = 1.00 * BASE_RANGE
They can be overridden with POLATORY_SUPPORT_DECAY_START_FRACTION and
POLATORY_SUPPORT_DECAY_END_FRACTION.
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

# Installs the selected-case runner and background_blending=False first.
import run_selected_no_background_blending as selected  # noqa: E402

suite = selected.suite
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
        # Cubic smoothstep: zero slope at both ends avoids a visible support shell.
        weight = t * t * (3.0 - 2.0 * t)
        adjusted = (1.0 - weight) * flat_values + weight * _OUTSIDE_VALUE

        # Exact zero is ambiguous for marching cubes. Bias only exact zeros by a tiny
        # scale-relative amount; every nonzero value remains governed by the smooth blend.
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


polatory.StructuralInterpolant3 = _smooth_support_factory
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-no-background-blending-smooth-support-decay"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: background_blending=False with smooth nearest-data "
    f"support decay from {_DECAY_START:g} m to {_DECAY_END:g} m; the fitted field is "
    "unchanged inside the start radius.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
