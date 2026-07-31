"""Run the real-gold benchmark with structural background blending disabled.

This controlled diagnostic keeps the recovered automatic domains, support indices,
local anisotropies, RBF parameters, and aligned global-grid mesher unchanged.  It
changes only the StructuralInterpolant3 background_blending flag from True to False.
The result isolates whether blending each finite domain toward outside_value creates
box-aligned zero-surface shelves, vertical walls, and forced terminations.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import polatory

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_real_gold_suite as suite  # noqa: E402


_ORIGINAL_STRUCTURAL_INTERPOLANT = polatory.StructuralInterpolant3


def _no_background_blending(*args: Any, **kwargs: Any):
    positional = list(args)
    if len(positional) >= 5:
        positional[4] = False
        kwargs.pop("background_blending", None)
    else:
        kwargs["background_blending"] = False
    return _ORIGINAL_STRUCTURAL_INTERPOLANT(*positional, **kwargs)


polatory.StructuralInterpolant3 = _no_background_blending
suite.OUTPUT_DIR = suite.ROOT / "benchmark-results" / "diagnostic-no-background-blending"
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: StructuralInterpolant3 background_blending=False; "
    "all automatic domains and meshing settings are unchanged.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
