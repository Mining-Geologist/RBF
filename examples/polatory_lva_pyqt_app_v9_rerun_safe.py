"""Repeat-run-safe category-contact launcher for the Polatory LVA application.

Extends v8 and fixes two GUI lifecycle problems:

1. A completed ModelWorker was scheduled for deletion from QThread.finished, after
   the worker thread event loop had already stopped. Large payload arrays and Qt
   wrappers could therefore survive between runs. v9 schedules worker deletion
   from the worker's own finished/failed signal before quitting its thread.
2. Previous result actors, scalar bars, datasets and temporary OBJ files are
   retired on the GUI thread before a replacement run starts.

Display policy:
- "Automatic LVA surface" is visible by default.
- every other layer is created hidden by default and remains available in the
  layer list for manual display.

Run:
    python polatory_lva_pyqt_app_v9_rerun_safe.py
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

import polatory

import polatory_lva_pyqt_app_v8_category_contacts as v8

app = v8.app

GENERATED_SURFACE = "Automatic LVA surface"
RESULT_LAYER_NAMES = (
    GENERATED_SURFACE,
    "Automatic domain points",
    "Automatic centroid domains",
    "LVA field slices",
    "LVA principal axes",
)
RESULT_SCALAR_BAR_TITLES = (
    "Automatic domain",
    "LVA ratio",
)

_original_window_init = app.MainWindow.__init__
_original_add_layer = app.MainWindow._add_layer
_original_model_finished = app.MainWindow.model_finished


def result_only_add_layer(
    self: Any,
    name: str,
    dataset: Any,
    **kwargs: Any,
) -> Any:
    """Create every non-result layer hidden and select only the generated mesh."""
    is_generated = name == GENERATED_SURFACE
    kwargs["visible"] = bool(is_generated)
    kwargs["select"] = bool(is_generated)
    return _original_add_layer(self, name, dataset, **kwargs)


def _remove_result_scalar_bars(self: Any) -> None:
    remove_scalar_bar = getattr(self.plotter, "remove_scalar_bar", None)
    if not callable(remove_scalar_bar):
        return
    for title in RESULT_SCALAR_BAR_TITLES:
        try:
            remove_scalar_bar(title, render=False)
        except TypeError:
            try:
                remove_scalar_bar(title)
            except Exception:
                pass
        except Exception:
            pass


def retire_previous_result(self: Any) -> None:
    """Release the complete previous result before allocating the next one."""
    for name in RESULT_LAYER_NAMES:
        self._remove_layer(name)

    _remove_result_scalar_bars(self)

    old_path = self.result_temp_obj
    self.result_temp_obj = None
    self.result_surface = None

    try:
        self.plotter.render()
    except Exception:
        pass

    if old_path is not None:
        path = Path(old_path)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    gc.collect()


def show_only_generated_surface(self: Any) -> None:
    """Apply the requested result visibility policy after every successful run."""
    for name, record in tuple(self.layers.items()):
        visible = name == GENERATED_SURFACE
        try:
            record.actor.SetVisibility(visible)
        except Exception:
            continue

    matches = self.layer_list.findItems(
        GENERATED_SURFACE,
        app.QtCore.Qt.MatchFlag.MatchExactly,
    )
    if matches:
        self.layer_list.setCurrentItem(matches[0])

    try:
        self.plotter.render()
    except Exception:
        pass


def rerun_safe_window_init(self: Any) -> None:
    _original_window_init(self)

    self._rerun_enable_timer = app.QtCore.QTimer(self)
    self._rerun_enable_timer.setSingleShot(True)
    self._rerun_enable_timer.timeout.connect(
        lambda: self.run_button.setEnabled(self._thread is None)
    )

    self.run_button.setToolTip(
        "Runs may be repeated after changing parameters. v9 releases the previous "
        "worker, VTK result layers and temporary OBJ before starting the next run."
    )


def rerun_safe_thread_finished(self: Any) -> None:
    """Clear references only after the worker thread has genuinely stopped."""
    self._worker = None
    self._thread = None
    self.progress_bar.setRange(0, 1)
    self.progress_bar.setValue(0)

    self.run_button.setEnabled(False)
    self._rerun_enable_timer.start(150)


def rerun_safe_model_finished(self: Any, result: dict[str, Any]) -> None:
    _original_model_finished(self, result)
    show_only_generated_surface(self)


def rerun_safe_run_model(self: Any) -> None:
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
        contact_count = int(np.count_nonzero(indicators == 0.0))
        if contact_count and self.nugget_spin.value() != 0.0:
            self.nugget_spin.setValue(0.0)
            self._log(
                "Nugget was forced to 0 because Contact categories are active."
            )

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

        retire_previous_result(self)

        self._rerun_enable_timer.stop()
        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.tabs.setCurrentIndex(self.log_tab_index)

        if contact_count:
            self._log(
                f"Starting category-contact LVA model with {contact_count:,} exact "
                "zero constraints…"
            )
        else:
            self._log("Starting category-based automatic LVA model…")

        thread = app.QtCore.QThread(self)
        worker = app.ModelWorker(payload)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self.model_finished)
        worker.failed.connect(self.model_failed)

        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.finished.connect(self._rerun_safe_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()
    except Exception as error:
        self._show_error("Could not start modelling", error)


app.MainWindow.__init__ = rerun_safe_window_init
app.MainWindow._add_layer = result_only_add_layer
app.MainWindow.model_finished = rerun_safe_model_finished
app.MainWindow.run_model = rerun_safe_run_model
app.MainWindow._rerun_safe_thread_finished = rerun_safe_thread_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
