"""Process-isolated repeat-run launcher for the Polatory LVA application.

This version keeps the Qt/PyVista GUI in the main process but executes every
native Polatory modelling run in a fresh Python subprocess. The fitted
StructuralInterpolant3 and all native solver state are therefore destroyed by
process exit after each run. A native abort can no longer terminate the GUI.

Only the generated "Automatic LVA surface" is visible by default; every other
layer remains available but hidden.

Run:
    python polatory_lva_pyqt_app_v10_process_isolated.py
"""

from __future__ import annotations

import gc
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polatory

import polatory_lva_pyqt_app_v5_streamed_lva as v5
import polatory_lva_pyqt_app_v9_rerun_safe as v9

app = v9.app

_original_window_init = app.MainWindow.__init__
_original_model_finished = app.MainWindow.model_finished
_original_close_event = app.MainWindow.closeEvent


def _cleanup_process_files(self: Any, *, keep_obj: bool = False) -> None:
    work_dir = getattr(self, "_model_process_work_dir", None)
    if work_dir:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
    self._model_process_work_dir = None

    output_obj = getattr(self, "_model_process_output_obj", None)
    if output_obj and not keep_obj:
        path = Path(output_obj)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    self._model_process_output_obj = None
    self._model_process_input = None
    self._model_process_result = None


def _restore_lva_plane_metadata(result: dict[str, Any]) -> None:
    """Rebuild v5's ndarray subclass after crossing the pickle boundary."""
    points = result.get("lva_points")
    if points is None or isinstance(points, v5.PlanePointArray):
        return

    dimension = result.pop("lva_plane_dimension", None)
    if dimension is None:
        dimensions = result.get("lva_dimensions")
        if dimensions is not None and len(dimensions) == 3:
            candidate = int(dimensions[0])
            if tuple(int(value) for value in dimensions) == (
                candidate,
                candidate,
                candidate,
            ) and len(points) == 3 * candidate * candidate:
                dimension = candidate

    if dimension is not None:
        result["lva_points"] = v5.PlanePointArray(
            np.asarray(points, dtype=np.float32),
            int(dimension),
        )


def process_window_init(self: Any) -> None:
    _original_window_init(self)
    self._model_process: app.QtCore.QProcess | None = None
    self._model_process_work_dir: str | None = None
    self._model_process_input: str | None = None
    self._model_process_result: str | None = None
    self._model_process_output_obj: str | None = None
    self._model_process_output_buffer = ""
    self.run_button.setToolTip(
        "Each modelling run starts in a fresh child process. Native fitting state "
        "cannot survive into the next parameter test or terminate the GUI."
    )


def _process_is_running(self: Any) -> bool:
    process = getattr(self, "_model_process", None)
    if process is None:
        return False
    return process.state() != app.QtCore.QProcess.ProcessState.NotRunning


def _read_process_output(self: Any) -> None:
    process = self._model_process
    if process is None:
        return
    chunk = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
    if not chunk:
        return
    self._model_process_output_buffer += chunk
    while "\n" in self._model_process_output_buffer:
        line, self._model_process_output_buffer = self._model_process_output_buffer.split(
            "\n", 1
        )
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith("PROGRESS\t"):
            self._log(line.split("\t", 1)[1])
        elif line != "RESULT_READY":
            self._log(f"Worker: {line}")


def _flush_process_output(self: Any) -> None:
    _read_process_output(self)
    remaining = self._model_process_output_buffer.strip()
    self._model_process_output_buffer = ""
    if remaining:
        for line in remaining.splitlines():
            if line.startswith("PROGRESS\t"):
                self._log(line.split("\t", 1)[1])
            elif line != "RESULT_READY":
                self._log(f"Worker: {line}")


def _process_error(self: Any, error: Any) -> None:
    self._log(f"Native worker process error: {error}")


def _process_finished(self: Any, exit_code: int, exit_status: Any) -> None:
    _flush_process_output(self)
    self._model_process = None
    self.run_button.setEnabled(True)
    self.progress_bar.setRange(0, 1)
    self.progress_bar.setValue(0)

    normal = exit_status == app.QtCore.QProcess.ExitStatus.NormalExit
    result_path = Path(self._model_process_result) if self._model_process_result else None

    if exit_code == 0 and normal and result_path is not None and result_path.exists():
        try:
            with result_path.open("rb") as stream:
                result = pickle.load(stream)
            _restore_lva_plane_metadata(result)
            _cleanup_process_files(self, keep_obj=True)
            _original_model_finished(self, result)
            v9.show_only_generated_surface(self)
            self._log(
                "Isolated modelling process exited cleanly. The next parameter run "
                "will start in a new interpreter."
            )
            gc.collect()
            return
        except Exception as error:
            _cleanup_process_files(self, keep_obj=False)
            self._show_error("Could not load the isolated model result", error)
            return

    status_name = "crashed" if not normal else "failed"
    details = (
        f"The isolated Polatory worker {status_name} during native modelling "
        f"(exit code {exit_code}). The GUI was protected and remains open."
    )
    _cleanup_process_files(self, keep_obj=False)
    self._log(details)
    app.QtWidgets.QMessageBox.critical(self, "Polatory worker failed", details)


def process_run_model(self: Any) -> None:
    if _process_is_running(self):
        app.QtWidgets.QMessageBox.information(
            self,
            app.APP_TITLE,
            "A modelling run is already in progress.",
        )
        return

    try:
        if not hasattr(polatory, "AutomaticStructuralDomainBuilder3"):
            raise RuntimeError(
                "AutomaticStructuralDomainBuilder3 is unavailable. Reinstall the "
                "feature/automatic-subdomainer branch and restart the app."
            )
        if self.reference_vertices is None or self.reference_faces is None:
            raise ValueError("Load the structural reference OBJ first.")

        points, indicators, roles, source_rows = self._mapped_arrays()
        contact_count = int(np.count_nonzero(indicators == 0.0))
        if contact_count and self.nugget_spin.value() != 0.0:
            self.nugget_spin.setValue(0.0)
            self._log("Nugget was forced to 0 because Contact categories are active.")

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

        v9.retire_previous_result(self)
        _cleanup_process_files(self, keep_obj=False)

        work_dir = Path(tempfile.mkdtemp(prefix="polatory_lva_process_"))
        input_path = work_dir / "payload.pkl"
        result_path = work_dir / "result.pkl"
        descriptor, output_obj_name = tempfile.mkstemp(
            prefix="polatory_lva_result_", suffix=".obj"
        )
        os.close(descriptor)
        output_obj = Path(output_obj_name)
        try:
            output_obj.unlink()
        except OSError:
            pass

        with input_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

        helper = Path(__file__).with_name("polatory_lva_worker_process.py")
        if not helper.exists():
            raise FileNotFoundError(f"Missing isolated worker script: {helper}")

        process = app.QtCore.QProcess(self)
        process.setProcessChannelMode(app.QtCore.QProcess.ProcessChannelMode.MergedChannels)
        environment = app.QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                str(helper),
                "--input",
                str(input_path),
                "--result",
                str(result_path),
                "--obj",
                str(output_obj),
            ]
        )
        process.readyReadStandardOutput.connect(
            lambda: _read_process_output(self)
        )
        process.errorOccurred.connect(lambda error: _process_error(self, error))
        process.finished.connect(
            lambda code, status: _process_finished(self, int(code), status)
        )

        self._model_process = process
        self._model_process_work_dir = str(work_dir)
        self._model_process_input = str(input_path)
        self._model_process_result = str(result_path)
        self._model_process_output_obj = str(output_obj)
        self._model_process_output_buffer = ""

        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.tabs.setCurrentIndex(self.log_tab_index)
        self._log(
            "Starting Polatory in an isolated process. A native fitting crash can "
            "no longer close this application."
        )
        process.start()
    except Exception as error:
        _cleanup_process_files(self, keep_obj=False)
        self.run_button.setEnabled(True)
        self._show_error("Could not start isolated modelling", error)


def process_close_event(self: Any, event: Any) -> None:
    process = getattr(self, "_model_process", None)
    if process is not None and process.state() != app.QtCore.QProcess.ProcessState.NotRunning:
        answer = app.QtWidgets.QMessageBox.question(
            self,
            "A model is still running",
            "Stop the isolated modelling process and close the application?",
        )
        if answer != app.QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        process.terminate()
        if not process.waitForFinished(3000):
            process.kill()
            process.waitForFinished(1000)
        self._model_process = None
        _cleanup_process_files(self, keep_obj=False)

    _original_close_event(self, event)


app.MainWindow.__init__ = process_window_init
app.MainWindow.run_model = process_run_model
app.MainWindow.closeEvent = process_close_event


if __name__ == "__main__":
    raise SystemExit(app.main())
