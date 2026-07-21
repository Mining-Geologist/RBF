"""Run one selected no-background-blending case with zero-plateau suppression and extra bottom meshing padding.

This diagnostic keeps the fitted structural model, automatic domains, supports, LVA parameters,
and all nonzero field values unchanged. It changes only the meshing extent by lowering the Z
minimum by one base range so a zero surface that reaches the benchmark floor can continue and
close naturally instead of being clipped into a flat open termination.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_selected_no_background_blending_zero_plateau as diagnostic  # noqa: E402

suite = diagnostic.suite

padding = float(
    os.environ.get("POLATORY_BENCHMARK_BOTTOM_PADDING", str(suite.BASE_RANGE))
)
if not np.isfinite(padding) or padding <= 0.0:
    raise ValueError("POLATORY_BENCHMARK_BOTTOM_PADDING must be a positive finite number.")

original_min = np.asarray(suite.MODEL_MIN, dtype=float).copy()
padded_min = original_min.copy()
padded_min[2] -= padding
suite.MODEL_MIN = padded_min

suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-no-background-blending-zero-plateau-padded-bottom"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: preserved the fitted field and lowered only the meshing "
    f"bottom from Z={original_min[2]:g} to Z={padded_min[2]:g} "
    f"(padding {padding:g}).",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
