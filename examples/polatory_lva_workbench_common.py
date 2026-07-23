"""Interactive Polatory structural-LVA modelling and visualization workbench.

This launcher extends ``polatory_lva_pyqt_app_v11_leapfrog_defaults.py`` without
changing its process-isolated modelling path.  It adds:

* arbitrary comparison-mesh loading;
* point/orientation-field CSV loading;
* Dip/Azimuth or XYZ-vector arrows;
* variable-scaled point/sphere/arrow glyphs;
* mesh-to-mesh distance comparison;
* per-layer colour, opacity, representation, edge, line and point controls.

The existing Data and Model tabs still provide arbitrary CSV coordinate/category
mapping, Inside/Outside/Contact/Ignore roles, structural reference OBJ loading,
Strength, Trend range, automatic domains, LVA field slices and cluster diagnostics.

Run from the repository virtual environment::

    python examples/polatory_lva_workbench.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as error:  # pragma: no cover - deployment validation
    raise SystemExit("Install PyQt6: python -m pip install PyQt6") from error

try:
    from scipy.spatial import cKDTree
except ImportError as error:  # pragma: no cover - deployment validation
    raise SystemExit("Install SciPy: python -m pip install scipy") from error

import polatory_lva_pyqt_app_v11_leapfrog_defaults as v11

app = v11.app
v10 = v11.v10

UNIFORM = "<Uniform>"
CONSTANT = "<Constant>"
NONE = "<None>"

WORKBENCH_PREFIXES = ("Comparison: ", "Field: ", "Distance: ")

_base_window_init = app.MainWindow.__init__
_original_add_layer = app.MainWindow._add_layer
_process_model_finished = v10._original_model_finished


@dataclass
class FieldState:
    path: Path | None = None
    frame: pd.DataFrame | None = None


def _current_plot_background_is_dark(window: Any) -> bool:
    try:
        red, green, blue = window.plotter.renderer.GetBackground()
        luminance = 0.2126 * float(red) + 0.7152 * float(green) + 0.0722 * float(blue)
        return luminance < 0.5
    except Exception:
        return False


def _update_background_button(window: Any) -> None:
    button = getattr(window, "background_toggle_button", None)
    if button is None:
        return
    dark = bool(getattr(window, "_dark_plot_background", False))
    if dark:
        button.setText("Switch to white background")
        button.setStyleSheet(
            "QPushButton { background: #f2f2f2; color: #111111; "
            "border: 1px solid #777777; border-radius: 4px; padding: 5px 9px; }"
        )
    else:
        button.setText("Switch to black background")
        button.setStyleSheet(
            "QPushButton { background: #202020; color: #ffffff; "
            "border: 1px solid #777777; border-radius: 4px; padding: 5px 9px; }"
        )


def _apply_plot_background(window: Any, dark: bool) -> None:
    window._dark_plot_background = bool(dark)
    window.plotter.set_background("black" if dark else "white")
    _update_background_button(window)
    window.plotter.render()


def _toggle_plot_background(window: Any) -> None:
    _apply_plot_background(
        window,
        not bool(getattr(window, "_dark_plot_background", False)),
    )


def _install_background_toggle(window: Any) -> None:
    """Place a compact black/white switch over the PyVista viewport."""
    parent = getattr(window.plotter, "interactor", None)
    if parent is None or not isinstance(parent, QtWidgets.QWidget):
        parent = window

    button = QtWidgets.QPushButton(parent)
    button.setObjectName("pyvistaBackgroundToggle")
    button.setFixedSize(190, 32)
    button.move(12, 12)
    button.setToolTip("Switch the PyVista viewport between black and white backgrounds.")
    button.clicked.connect(lambda _checked=False: _toggle_plot_background(window))
    window.background_toggle_button = button
    window._dark_plot_background = _current_plot_background_is_dark(window)
    _update_background_button(window)
    button.show()
    button.raise_()


def _window_init_with_background_toggle(window: Any) -> None:
    _base_window_init(window)
    _install_background_toggle(window)


# ``polatory_lva_workbench.py`` calls this captured initializer before adding its
# own Workbench tab, so the viewport switch is available in every workbench run.
_original_window_init = _window_init_with_background_toggle


def _unique_name(window: Any, base: str) -> str:
    if base not in window.layers:
        return base
    index = 2
    while f"{base} ({index})" in window.layers:
        index += 1
    return f"{base} ({index})"


def _record_dataset(record: Any) -> pv.DataSet | pv.MultiBlock:
    if isinstance(record, dict):
        for key in ("dataset", "data", "mesh"):
            if record.get(key) is not None:
                return record[key]
    for attribute in ("dataset", "data", "mesh"):
        value = getattr(record, attribute, None)
        if value is not None:
            return value
    raise ValueError("The selected layer does not expose its PyVista dataset.")


def _record_actor(record: Any) -> Any:
    actor = record.get("actor") if isinstance(record, dict) else getattr(record, "actor", None)
    if actor is None:
        raise ValueError("The selected layer does not expose a VTK actor.")
    return actor


def _actor_property(actor: Any) -> Any:
    getter = getattr(actor, "GetProperty", None)
    if callable(getter):
        return getter()
    prop = getattr(actor, "prop", None)
    if prop is not None:
        return prop
    raise ValueError("The selected actor does not expose editable properties.")


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    qcolor = QtGui.QColor(color)
    if not qcolor.isValid():
        raise ValueError(f"Invalid colour: {color}")
    return qcolor.redF(), qcolor.greenF(), qcolor.blueF()


def _set_button_colour(button: QtWidgets.QPushButton, color: str) -> None:
    qcolor = QtGui.QColor(color)
    if not qcolor.isValid():
        return
    button.setProperty("selectedColor", qcolor.name())
    foreground = "#000000" if qcolor.lightnessF() > 0.55 else "#ffffff"
    button.setStyleSheet(
        f"QPushButton {{ background: {qcolor.name()}; color: {foreground}; }}"
    )
    button.setText(qcolor.name())


def _button_colour(button: QtWidgets.QPushButton, fallback: str) -> str:
    value = button.property("selectedColor")
    return str(value) if value else fallback


def _choose_colour(window: Any, button: QtWidgets.QPushButton) -> None:
    initial = QtGui.QColor(_button_colour(button, "#ffffff"))
    chosen = QtWidgets.QColorDialog.getColor(initial, window, "Choose layer colour")
    if chosen.isValid():
        _set_button_colour(button, chosen.name())


def _mesh_surface(dataset: Any) -> pv.PolyData:
    if isinstance(dataset, pv.MultiBlock):
        combined = dataset.combine()
    else:
        combined = dataset
    surface = combined.extract_surface().triangulate().clean()
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("The selected layer has no triangulated surface cells.")
    return surface


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if int(values.notna().sum()) > 0:
            result.append(str(column))
    return result


def _preferred(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {column.strip().casefold(): column for column in columns}
    for candidate in candidates:
        match = lower.get(candidate.casefold())
        if match is not None:
            return match
    return None


def _set_combo(combo: QtWidgets.QComboBox, value: str | None) -> None:
    if value is None:
        return
    index = combo.findText(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def dip_azimuth_vectors(dip: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Convert geology dip/azimuth to unit down-dip vectors.

    Azimuth is clockwise from north (+Y). Dip is positive downward from horizontal,
    therefore Z is negative for a positive dip in an elevation coordinate system.
    """
    dip_radians = np.deg2rad(np.asarray(dip, dtype=float))
    azimuth_radians = np.deg2rad(np.asarray(azimuth, dtype=float))
    horizontal = np.cos(dip_radians)
    vectors = np.column_stack(
        [
            horizontal * np.sin(azimuth_radians),
            horizontal * np.cos(azimuth_radians),
            -np.sin(dip_radians),
        ]
    )
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 0.0
    vectors[valid] /= lengths[valid, None]
    return vectors


def _normalised_scale(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float))
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.ones(len(values), dtype=np.float32)
    low = float(np.nanpercentile(values[finite], 2.0))
    high = float(np.nanpercentile(values[finite], 98.0))
    if not high > low:
        return np.ones(len(values), dtype=np.float32)
    clipped = np.clip(values, low, high)
    result = 0.2 + 0.8 * (clipped - low) / (high - low)
    result[~finite] = 0.2
    return result.astype(np.float32)


def workbench_add_layer(
    self: Any,
    name: str,
    dataset: Any,
    **kwargs: Any,
) -> Any:
    requested_visible = bool(kwargs.get("visible", True))
    requested_select = bool(kwargs.get("select", True))
    result = _original_add_layer(self, name, dataset, **kwargs)

    # v9 intentionally hides every non-generated modelling result.  Workbench
    # imports should honour the visibility requested by the user.
    if name.startswith(WORKBENCH_PREFIXES):
        record = self.layers.get(name)
        if record is not None:
            try:
                _record_actor(record).SetVisibility(requested_visible)
            except Exception:
                pass
        matches = self.layer_list.findItems(
            name,
            QtCore.Qt.MatchFlag.MatchExactly,
        )
        if matches:
            item = matches[0]
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if requested_visible
                    else QtCore.Qt.CheckState.Unchecked
                )
            if requested_select:
                self.layer_list.setCurrentItem(item)
        try:
            self.plotter.render()
        except Exception:
            pass
    return result


def _selected_layer_name(self: Any) -> str | None:
    item = self.layer_list.currentItem()
    if item is None:
        return None
    text = item.text().strip()
    return text if text in self.layers else None


def _refresh_layer_combos(self: Any) -> None:
    previous_source = self.compare_source_combo.currentText()
    previous_target = self.compare_target_combo.currentText()
    surface_names: list[str] = []
    for name, record in self.layers.items():
        try:
            surface = _mesh_surface(_record_dataset(record))
        except Exception:
            continue
        if surface.n_cells:
            surface_names.append(name)

    for combo, previous in (
        (self.compare_source_combo, previous_source),
        (self.compare_target_combo, previous_target),
    ):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(surface_names)
        if previous in surface_names:
            combo.setCurrentText(previous)
        combo.blockSignals(False)

    generated = "Automatic LVA surface"
    comparison_names = [name for name in surface_names if name.startswith("Comparison: ")]
    if generated in surface_names:
        self.compare_source_combo.setCurrentText(generated)
    if comparison_names:
        self.compare_target_combo.setCurrentText(comparison_names[-1])


def _sync_selected_layer_controls(self: Any) -> None:
    name = _selected_layer_name(self)
    self.selected_layer_label.setText(name or "No layer selected")
    if name is None:
        return
    try:
        record = self.layers[name]
        actor = _record_actor(record)
        prop = _actor_property(actor)
        self.layer_visible_check.setChecked(bool(actor.GetVisibility()))
        self.layer_opacity_spin.setValue(float(prop.GetOpacity()))
        self.layer_line_width_spin.setValue(float(prop.GetLineWidth()))
        self.layer_point_size_spin.setValue(float(prop.GetPointSize()))
        color = prop.GetColor()
        color_hex = QtGui.QColor.fromRgbF(*[float(value) for value in color]).name()
        _set_button_colour(self.layer_color_button, color_hex)
        edge = prop.GetEdgeColor()
        edge_hex = QtGui.QColor.fromRgbF(*[float(value) for value in edge]).name()
        _set_button_colour(self.edge_color_button, edge_hex)
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        self.layer_scalar_colors_check.setChecked(
            bool(mapper.GetScalarVisibility()) if mapper is not None else False
        )
        representation = int(prop.GetRepresentation())
        edge_visible = bool(prop.GetEdgeVisibility())
        if representation == 1:
            text = "Wireframe"
        elif representation == 0:
            text = "Points"
        elif edge_visible:
            text = "Surface + wireframe"
        else:
            text = "Surface"
        self.layer_representation_combo.setCurrentText(text)
    except Exception as error:
        self._log(f"Could not read layer appearance for '{name}': {error}")


def _apply_selected_layer_style(self: Any) -> None:
    name = _selected_layer_name(self)
    if name is None:
        return
    try:
        record = self.layers[name]
        actor = _record_actor(record)
        prop = _actor_property(actor)
        actor.SetVisibility(self.layer_visible_check.isChecked())
        prop.SetOpacity(float(self.layer_opacity_spin.value()))
        prop.SetColor(*_hex_to_rgb(_button_colour(self.layer_color_button, "#ffffff")))
        prop.SetEdgeColor(*_hex_to_rgb(_button_colour(self.edge_color_button, "#202020")))
        prop.SetLineWidth(float(self.layer_line_width_spin.value()))
        prop.SetPointSize(float(self.layer_point_size_spin.value()))
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        if mapper is not None:
            mapper.SetScalarVisibility(self.layer_scalar_colors_check.isChecked())

        representation = self.layer_representation_combo.currentText()
        if representation == "Wireframe":
            prop.SetRepresentationToWireframe()
            prop.SetEdgeVisibility(False)
        elif representation == "Points":
            prop.SetRepresentationToPoints()
            prop.SetEdgeVisibility(False)
        else:
            prop.SetRepresentationToSurface()
            prop.SetEdgeVisibility(representation == "Surface + wireframe")

        matches = self.layer_list.findItems(name, QtCore.Qt.MatchFlag.MatchExactly)
        if matches and matches[0].flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
            matches[0].setCheckState(
                QtCore.Qt.CheckState.Checked
                if self.layer_visible_check.isChecked()
                else QtCore.Qt.CheckState.Unchecked
            )
        self.plotter.render()
    except Exception as error:
        self._show_error("Could not update the selected layer", error)


def _remove_selected_workbench_layer(self: Any) -> None:
    name = _selected_layer_name(self)
    if name is None:
        return
    if not name.startswith(WORKBENCH_PREFIXES):
        QtWidgets.QMessageBox.information(
            self,
            "Protected model layer",
            "Only imported Workbench layers can be removed here. Model result layers "
            "are replaced automatically by the next modelling run.",
        )
        return
    self._remove_layer(name)
    _refresh_layer_combos(self)


__all__ = [name for name in globals() if not name.startswith("__")]
