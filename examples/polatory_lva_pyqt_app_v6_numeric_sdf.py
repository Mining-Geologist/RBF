"""Numerical-SDF input launcher for the Polatory structural-LVA application.

This launcher extends v5 with two input modes:

1. Binary categories: the existing Leapfrog-compatible indicator conversion.
2. Numerical SDF/scalar values: use a selected CSV value column directly while
   retaining the category-role table only as the Include/Ignore mask.

Run:
    python polatory_lva_pyqt_app_v6_numeric_sdf.py
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import polatory

import polatory_lva_pyqt_app_v5_streamed_lva as v5

app = v5.app

BINARY_MODE = "Binary categories"
NUMERICAL_MODE = "Numerical SDF / scalar values"

_original_indicator_values = polatory.leapfrog_indicator_values3
_original_window_init = app.MainWindow.__init__
_original_load_csv = app.MainWindow.load_csv
_original_model_parameters = app.MainWindow.model_parameters
_numeric_state = threading.local()


def _input_aware_indicator_values(
    points: np.ndarray,
    indicators: np.ndarray,
    fit_accuracy: float = 0.0,
) -> Any:
    """Return direct numerical values when the current worker requests them."""
    values = getattr(_numeric_state, "values", None)
    if values is None:
        return _original_indicator_values(
            points,
            indicators,
            fit_accuracy=fit_accuracy,
        )

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(points):
        raise ValueError("The numerical value column does not match the model points.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The numerical value/SDF column contains non-finite rows.")

    span = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(span))
    resolved_accuracy = float(fit_accuracy)
    if not resolved_accuracy > 0.0:
        resolved_accuracy = max(1.0e-5 * diagonal, np.finfo(float).eps)

    return SimpleNamespace(
        values=values.copy(),
        fit_accuracy=resolved_accuracy,
        clipping_distance=float("nan"),
        data_diagonal=diagonal,
    )


def numerical_worker_run(self: Any) -> None:
    parameters = self.payload.get("parameters", {})
    numerical = parameters.get("input_mode") == NUMERICAL_MODE
    if numerical:
        _numeric_state.values = np.asarray(
            self.payload["numeric_values"],
            dtype=float,
        )
    else:
        _numeric_state.values = None
    try:
        v5.scalable_worker_run(self)
    finally:
        _numeric_state.values = None


def _set_value_mode_enabled(self: Any) -> None:
    numerical = self.input_mode_combo.currentText() == NUMERICAL_MODE
    self.value_column_combo.setEnabled(numerical)
    self.numeric_mode_help.setText(
        (
            "Numerical mode uses the selected value column directly and extracts "
            "the zero isosurface. Category roles still control which rows are "
            "ignored; Inside/Outside labels are not converted to +/-1."
        )
        if numerical
        else (
            "Binary mode converts the category roles to Leapfrog-compatible "
            "inside/outside indicator distances."
        )
    )


def numerical_window_init(self: Any) -> None:
    _original_window_init(self)

    group = app.QtWidgets.QGroupBox("Input field mode")
    form = app.QtWidgets.QFormLayout(group)

    self.input_mode_combo = app.QtWidgets.QComboBox()
    self.input_mode_combo.addItems([BINARY_MODE, NUMERICAL_MODE])

    self.value_column_combo = app.QtWidgets.QComboBox()
    self.numeric_mode_help = app.QtWidgets.QLabel()
    self.numeric_mode_help.setWordWrap(True)

    form.addRow("Mode", self.input_mode_combo)
    form.addRow("Value / SDF column", self.value_column_combo)
    form.addRow(self.numeric_mode_help)

    data_scroll = self.tabs.widget(0)
    data_page = data_scroll.widget() if hasattr(data_scroll, "widget") else None
    if data_page is None or data_page.layout() is None:
        raise RuntimeError("Could not locate the Data-tab layout.")
    data_page.layout().insertWidget(2, group)

    self.input_mode_combo.currentTextChanged.connect(
        lambda _text: _set_value_mode_enabled(self)
    )
    _set_value_mode_enabled(self)


def numerical_load_csv(self: Any) -> None:
    _original_load_csv(self)
    if self.frame is None:
        return

    columns = [str(column) for column in self.frame.columns]
    previous = self.value_column_combo.currentText()
    self.value_column_combo.clear()
    self.value_column_combo.addItems(columns)

    preferred = None
    for candidate in ("Values", "Value", "SDF", "Signed distance", "Distance"):
        for column in columns:
            if column.strip().lower() == candidate.lower():
                preferred = column
                break
        if preferred is not None:
            break

    if preferred is None:
        numeric_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(self.frame[column])
            and column
            not in {
                self.x_combo.currentText(),
                self.y_combo.currentText(),
                self.z_combo.currentText(),
            }
        ]
        if numeric_columns:
            preferred = numeric_columns[0]

    if previous in columns:
        self.value_column_combo.setCurrentText(previous)
    elif preferred is not None:
        self.value_column_combo.setCurrentText(preferred)


def numerical_model_parameters(self: Any) -> dict[str, Any]:
    parameters = _original_model_parameters(self)
    parameters["input_mode"] = self.input_mode_combo.currentText()
    parameters["value_column"] = self.value_column_combo.currentText()
    return parameters


def numerical_run_model(self: Any) -> None:
    if self._thread is not None:
        app.QtWidgets.QMessageBox.information(
            self,
            app.APP_TITLE,
            "A modelling run is already in progress.",
        )
        return

    try:
        if not hasattr(polatory, "AutomaticStructuralDomainBuilder3"):
            raise RuntimeError(
                "This installed Polatory package does not expose "
                "AutomaticStructuralDomainBuilder3. Reinstall the "
                "feature/automatic-subdomainer branch, then restart this app."
            )
        if self.reference_vertices is None or self.reference_faces is None:
            raise ValueError("Load the structural reference OBJ first.")

        points, indicators, roles, source_rows = self._mapped_arrays()
        self.current_points = points
        self.current_indicators = indicators
        self.current_roles = roles
        self.current_source_rows = source_rows

        bbox_min, bbox_max = self.current_bbox()
        parameters = self.model_parameters()

        payload = {
            "points": points.copy(),
            "indicators": indicators.copy(),
            "trend_vertices": self.reference_vertices.copy(),
            "trend_faces": self.reference_faces.copy(),
            "bbox_min": bbox_min.copy(),
            "bbox_max": bbox_max.copy(),
            "parameters": parameters,
        }

        if parameters["input_mode"] == NUMERICAL_MODE:
            if self.frame is None:
                raise ValueError("Load a CSV first.")
            column = parameters["value_column"]
            if not column or column not in self.frame.columns:
                raise ValueError("Select a valid numerical value/SDF column.")

            all_values = pd.to_numeric(
                self.frame[column],
                errors="coerce",
            ).to_numpy(dtype=float)
            numerical_values = all_values[source_rows]
            if not np.all(np.isfinite(numerical_values)):
                bad = int(np.count_nonzero(~np.isfinite(numerical_values)))
                raise ValueError(
                    f"The selected value column has {bad:,} non-finite model rows."
                )
            if not (
                np.any(numerical_values < 0.0)
                and np.any(numerical_values > 0.0)
            ):
                raise ValueError(
                    "The numerical field must contain values on both sides of "
                    "zero to generate a zero isosurface."
                )
            payload["numeric_values"] = numerical_values.copy()

        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.tabs.setCurrentIndex(self.log_tab_index)

        if parameters["input_mode"] == NUMERICAL_MODE:
            values = payload["numeric_values"]
            self._log(
                f"Starting numerical-SDF LVA model from "
                f"'{parameters['value_column']}': range "
                f"{float(np.min(values)):.6g} to "
                f"{float(np.max(values)):.6g}."
            )
        else:
            self._log("Starting binary-category automatic LVA model…")

        thread = app.QtCore.QThread(self)
        worker = app.ModelWorker(payload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self.model_finished)
        worker.failed.connect(self.model_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()
    except Exception as error:
        self._show_error("Could not start modelling", error)


polatory.leapfrog_indicator_values3 = _input_aware_indicator_values
app.ModelWorker.run = numerical_worker_run
app.MainWindow.__init__ = numerical_window_init
app.MainWindow.load_csv = numerical_load_csv
app.MainWindow.model_parameters = numerical_model_parameters
app.MainWindow.run_model = numerical_run_model


if __name__ == "__main__":
    raise SystemExit(app.main())
