"""Minimal non-interactive compatibility surface for CI benchmark imports.

The production launcher chain historically imports the original PyQt v2 application
at module import time, even when a benchmark only needs the automatic domain builder
and chunk-safe mesher.  The full v2 GUI is intentionally not duplicated in the
repository.  GitHub Actions prepends this directory to ``PYTHONPATH`` so the launcher
patch modules can be imported headlessly without creating widgets.

No benchmark calls these UI methods.  They exist only so v3-v8 can capture and patch
the same attributes they patch in the interactive application.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from PyQt6 import QtCore, QtWidgets


APP_TITLE = "Polatory Automatic Structural LVA"
ROLE_INSIDE = "Inside"
ROLE_OUTSIDE = "Outside"
ROLE_IGNORE = "Ignore"
ROLE_OPTIONS = (ROLE_INSIDE, ROLE_OUTSIDE, ROLE_IGNORE)


def category_keys(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>").astype(str)


def default_role_for_category(value: str) -> str:
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        number = np.nan
    if np.isfinite(number):
        if number > 0.0:
            return ROLE_OUTSIDE
        if number < 0.0:
            return ROLE_INSIDE
        return ROLE_IGNORE
    if any(token in text for token in ("inside", "ore", "deposit", "target")):
        return ROLE_INSIDE
    if any(token in text for token in ("outside", "waste", "background")):
        return ROLE_OUTSIDE
    return ROLE_IGNORE


def structured_points(
    minimum: np.ndarray,
    maximum: np.ndarray,
    dimensions: tuple[int, int, int],
) -> np.ndarray:
    """Return the same regular Fortran-ordered point grid used by the GUI worker."""
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    dimensions = tuple(int(value) for value in dimensions)
    axes = [
        np.linspace(minimum[index], maximum[index], dimensions[index])
        for index in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.column_stack(
        [x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
    )


class ModelWorker:
    def run(self) -> None:
        raise RuntimeError("The CI compatibility worker must be patched before use.")


class MainWindow:
    """Attribute shell required only while the launcher patch modules are imported."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def model_parameters(self) -> dict[str, Any]:
        return {}

    def model_finished(self, result: dict[str, Any]) -> None:
        del result

    def load_csv(self) -> None:
        return None

    def run_model(self) -> None:
        return None

    def populate_category_roles(self, column: str) -> None:
        del column

    def _mapped_arrays(self):
        raise RuntimeError("The headless compatibility shell has no mapped UI data.")

    def apply_data_mapping(self) -> None:
        return None


def main() -> int:
    raise RuntimeError(
        "This module is a CI import shim, not an interactive application launcher."
    )
