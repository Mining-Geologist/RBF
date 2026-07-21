"""Run one selected no-background-blending case with exact-zero plateau suppression.

Disabling structural background blending gave the closest Leapfrog shape so far, but the
uncovered structural field can return exact zeros over finite patches.  Because zero is
also the requested isovalue, marching cubes may turn such a plateau into a thin flat
sheet.  This diagnostic changes only those exact-zero evaluation samples to a tiny
negative value.  Every nonzero field value, automatic domain, RBF fit, LVA parameter,
and global-grid sample location remains unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polatory

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# This installs background_blending=False and exact-case selection first.
import run_selected_no_background_blending as selected  # noqa: E402

suite = selected.suite
_BASE_FACTORY = polatory.StructuralInterpolant3
_RELATIVE_BIAS = 1.0e-6


class _ZeroPlateauSuppressedInterpolant:
    """Delegate the fitted model while biasing only exact zero evaluations negative."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._biased_total = 0

    def evaluate(self, *args: Any, **kwargs: Any):
        values = np.asarray(self._wrapped.evaluate(*args, **kwargs), dtype=float)
        zero = values == 0.0
        count = int(np.count_nonzero(zero))
        if count:
            nonzero = np.abs(values[~zero])
            scale = max(1.0, float(np.max(nonzero)) if nonzero.size else 1.0)
            epsilon = _RELATIVE_BIAS * scale
            values = values.copy()
            values[zero] = -epsilon
            self._biased_total += count
            print(
                "PROGRESS\tZero-plateau suppression biased "
                f"{count:,} exact-zero field sample(s) to {-epsilon:.6g} "
                f"({self._biased_total:,} cumulative).",
                flush=True,
            )
        return values

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _zero_plateau_suppressed_factory(*args: Any, **kwargs: Any):
    return _ZeroPlateauSuppressedInterpolant(_BASE_FACTORY(*args, **kwargs))


polatory.StructuralInterpolant3 = _zero_plateau_suppressed_factory
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-no-background-blending-zero-plateau-fix"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: background_blending=False plus exact-zero plateau "
    "suppression; all nonzero structural-field values remain unchanged.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
