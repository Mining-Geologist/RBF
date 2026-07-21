"""Run one selected Leapfrog gold case with background blending disabled.

Select the exact Leapfrog reference case through ``POLATORY_BENCHMARK_CASE`` using
names such as ``S3_R300`` or ``S5_R100``.  The modelling and aligned global-grid
meshing path is identical to ``run_no_background_blending_diagnostic.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing this module installs background_blending=False and exposes its suite.
import run_no_background_blending_diagnostic as diagnostic  # noqa: E402

suite = diagnostic.suite
_ORIGINAL_AVAILABLE_CASES = suite.available_cases


def _selected_cases():
    requested = os.environ.get("POLATORY_BENCHMARK_CASE", "").strip().upper()
    if not requested:
        raise ValueError(
            "Set POLATORY_BENCHMARK_CASE to an existing Leapfrog case, for example "
            "S3_R300 or S5_R100."
        )

    # Disable the old first-N limiter while collecting the complete reference list.
    previous_max_cases = suite.MAX_CASES
    suite.MAX_CASES = 0
    try:
        cases = _ORIGINAL_AVAILABLE_CASES()
    finally:
        suite.MAX_CASES = previous_max_cases

    matches = [case for case in cases if case.name.upper() == requested]
    if not matches:
        available = ", ".join(case.name for case in cases)
        raise ValueError(
            f"Leapfrog benchmark case {requested!r} was not found. Available cases: "
            f"{available}"
        )
    return matches


suite.available_cases = _selected_cases
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-no-background-blending-selected"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"


if __name__ == "__main__":
    raise SystemExit(suite.main())
