"""Four-role contact-constrained launcher for the Polatory LVA application.

This launcher extends v5 with the modelling roles requested by the user:

- Inside: positive fitted distance (UI indicator -1)
- Outside: negative fitted distance (UI indicator +1)
- Contact: exact zero constraint
- Ignore: excluded from domain construction and RBF fitting

A selected numeric column can optionally promote rows whose value is near zero to
Contact before the role table is populated. This is useful for CSV files where
contact rows are stored as Outside or Ignored in the categorical column while a
separate Values/SDF column contains the exact zeros.

Run:
    python polatory_lva_pyqt_app_v7_contact_roles.py
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import polatory

try:
    from scipy.spatial import cKDTree
except ImportError as error:
    raise SystemExit(
        "This launcher requires SciPy. Install it with: python -m pip install scipy"
    ) from error

import polatory_lva_pyqt_app_v5_streamed_lva as v5

app = v5.app

ROLE_CONTACT = "Contact"
CONTACT_COLUMN_NONE = "<Use category roles only>"
CONTACT_COLUMN_CANDIDATES = (
    "Values",
    "Value",
    "SDF",
    "Signed distance",
    "Signed Distance",
    "Distance",
)

app.ROLE_CONTACT = ROLE_CONTACT
app.ROLE_OPTIONS = (
    app.ROLE_INSIDE,
    app.ROLE_OUTSIDE,
    ROLE_CONTACT,
    app.ROLE_IGNORE,
)

_original_indicator_values = polatory.leapfrog_indicator_values3
_original_window_init = app.MainWindow.__init__
_original_load_csv = app.MainWindow.load_csv
_original_run_model = app.MainWindow.run_model
_original_model_finished = app.MainWindow.model_finished


def contact_default_role(value: str) -> str:
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        number = np.nan

    if np.isfinite(number):
        if number > 0.0:
            return app.ROLE_OUTSIDE
        if number < 0.0:
            return app.ROLE_INSIDE
        return ROLE_CONTACT

    contact_tokens = ("contact", "boundary", "interface", "surface", "zero")
    ignored_tokens = (
        "ignore",
        "ignored",
        "exclude",
        "excluded",
        "skip",
        "unused",
    )
    inside_tokens = (
        "inside",
        "interior",
        "ore",
        "mineral",
        "deposit",
        "target",
        "true",
        "yes",
    )
    outside_tokens = (
        "outside",
        "exterior",
        "waste",
        "background",
        "false",
        "no",
    )

    if any(token in text for token in contact_tokens):
        return ROLE_CONTACT
    if any(token in text for token in ignored_tokens):
        return app.ROLE_IGNORE
    if any(token in text for token in inside_tokens):
        return app.ROLE_INSIDE
    if any(token in text for token in outside_tokens):
        return app.ROLE_OUTSIDE
    return app.ROLE_IGNORE


app.default_role_for_category = contact_default_role


def _selected_contact_column(self: Any) -> str | None:
    if not hasattr(self, "contact_column_combo"):
        return None
    column = self.contact_column_combo.currentText().strip()
    if not column or column == CONTACT_COLUMN_NONE:
        return None
    if self.frame is None or column not in self.frame.columns:
        return None
    return column


def _contact_override_mask(self: Any) -> np.ndarray:
    if self.frame is None:
        return np.zeros(0, dtype=bool)
    column = _selected_contact_column(self)
    if column is None:
        return np.zeros(len(self.frame), dtype=bool)

    values = pd.to_numeric(self.frame[column], errors="coerce").to_numpy(dtype=float)
    tolerance = float(self.contact_zero_tolerance_spin.value())
    return np.isfinite(values) & (np.abs(values) <= tolerance)


def _effective_category_keys(self: Any, column: str) -> pd.Series:
    if self.frame is None:
        return pd.Series(dtype="string")
    keys = app.category_keys(self.frame[column]).copy()
    contact_mask = _contact_override_mask(self)
    if len(contact_mask) == len(keys) and np.any(contact_mask):
        keys.loc[contact_mask] = ROLE_CONTACT
    return keys


def contact_populate_category_roles(self: Any, column: str) -> None:
    if self.frame is None or not column or column not in self.frame.columns:
        self.role_table.setRowCount(0)
        return

    keys = _effective_category_keys(self, column)
    counts = keys.value_counts(dropna=False, sort=False)
    if len(counts) > 500:
        answer = app.QtWidgets.QMessageBox.question(
            self,
            "Many unique categories",
            f"The selected field contains {len(counts):,} effective categories. "
            "It may be continuous rather than categorical. Populate the table anyway?",
        )
        if answer != app.QtWidgets.QMessageBox.StandardButton.Yes:
            self.role_table.setRowCount(0)
            return

    previous = self.category_role_mapping()
    self.role_table.setRowCount(len(counts))
    for row, (value, count) in enumerate(counts.items()):
        value_text = str(value)
        value_item = app.QtWidgets.QTableWidgetItem(value_text)
        value_item.setFlags(
            value_item.flags() & ~app.QtCore.Qt.ItemFlag.ItemIsEditable
        )
        count_item = app.QtWidgets.QTableWidgetItem(f"{int(count):,}")
        count_item.setTextAlignment(app.QtCore.Qt.AlignmentFlag.AlignRight)
        count_item.setFlags(
            count_item.flags() & ~app.QtCore.Qt.ItemFlag.ItemIsEditable
        )
        role_combo = app.QtWidgets.QComboBox()
        role_combo.addItems(app.ROLE_OPTIONS)
        role_combo.setCurrentText(
            previous.get(value_text, contact_default_role(value_text))
        )
        self.role_table.setItem(row, 0, value_item)
        self.role_table.setItem(row, 1, count_item)
        self.role_table.setCellWidget(row, 2, role_combo)


def contact_mapped_arrays(
    self: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self.frame is None:
        raise ValueError("Load a CSV first.")

    columns = [
        self.x_combo.currentText(),
        self.y_combo.currentText(),
        self.z_combo.currentText(),
        self.category_combo.currentText(),
    ]
    if any(not column for column in columns):
        raise ValueError("Map X, Y, Z, and category columns.")
    if len(set(columns[:3])) != 3:
        raise ValueError("X, Y, and Z must use three different columns.")

    coordinates = self.frame[columns[:3]].apply(pd.to_numeric, errors="coerce")
    coordinate_array = coordinates.to_numpy(dtype=float)
    finite_coordinates = np.all(np.isfinite(coordinate_array), axis=1)

    keys = _effective_category_keys(self, columns[3])
    role_mapping = self.category_role_mapping()
    roles = keys.map(role_mapping).fillna(app.ROLE_IGNORE).to_numpy(dtype=object)
    model_mask = finite_coordinates & (roles != app.ROLE_IGNORE)

    points = coordinate_array[model_mask]
    model_roles = roles[model_mask]
    indicators = np.zeros(len(points), dtype=float)
    indicators[model_roles == app.ROLE_INSIDE] = -1.0
    indicators[model_roles == app.ROLE_OUTSIDE] = 1.0
    indicators[model_roles == ROLE_CONTACT] = 0.0
    source_rows = np.flatnonzero(model_mask)

    if len(points) < 3:
        raise ValueError(
            "At least three non-ignored points with valid coordinates are required."
        )
    if not np.any(indicators < 0.0):
        raise ValueError("Assign at least one category to Inside.")
    if not np.any(indicators > 0.0):
        raise ValueError("Assign at least one category to Outside.")

    return points, indicators, model_roles, source_rows


def contact_apply_data_mapping(self: Any) -> None:
    try:
        points, indicators, roles, source_rows = self._mapped_arrays()
        self.current_points = points
        self.current_indicators = indicators
        self.current_roles = roles
        self.current_source_rows = source_rows

        for name in (
            "Inside points",
            "Outside points",
            "Contact points",
            "Ignored points",
        ):
            self._remove_layer(name)

        inside = points[indicators < 0.0]
        outside = points[indicators > 0.0]
        contacts = points[indicators == 0.0]

        self._add_layer(
            "Inside points",
            app.pv.PolyData(inside),
            kind="points",
            color="#40c463",
            opacity=1.0,
            point_size=9,
        )
        self._add_layer(
            "Outside points",
            app.pv.PolyData(outside),
            kind="points",
            color="#f05b61",
            opacity=1.0,
            point_size=9,
        )
        if len(contacts):
            self._add_layer(
                "Contact points",
                app.pv.PolyData(contacts),
                kind="points",
                color="#ffb300",
                opacity=1.0,
                point_size=12,
            )

        mapped_columns = [
            self.x_combo.currentText(),
            self.y_combo.currentText(),
            self.z_combo.currentText(),
        ]
        all_coordinates = self.frame[mapped_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        all_points = all_coordinates.to_numpy(dtype=float)
        finite = np.all(np.isfinite(all_points), axis=1)
        ignored_mask = finite.copy()
        ignored_mask[source_rows] = False
        ignored = all_points[ignored_mask]
        if len(ignored):
            self._add_layer(
                "Ignored points",
                app.pv.PolyData(ignored),
                kind="points",
                color="#9aa0a6",
                opacity=0.35,
                point_size=5,
                visible=False,
            )

        self.fit_bbox_to_data()
        self._log(
            f"Mapped {len(points):,} modelling points: "
            f"{len(inside):,} inside, {len(outside):,} outside, "
            f"{len(contacts):,} exact contacts; {len(ignored):,} ignored."
        )
    except Exception as error:
        self._show_error("Invalid data mapping", error)


def contact_indicator_values3(
    points: np.ndarray,
    indicators: np.ndarray,
    *,
    fit_accuracy: float = 0.0,
    clipping_distance: float = 0.0,
) -> Any:
    points = np.asarray(points, dtype=float)
    indicators = np.asarray(indicators, dtype=float)

    contact = indicators == 0.0
    if not np.any(contact):
        return _original_indicator_values(
            points,
            indicators,
            fit_accuracy=fit_accuracy,
            clipping_distance=clipping_distance,
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(points) == 0:
        raise ValueError("points must not be empty")
    if indicators.ndim != 1 or len(indicators) != len(points):
        raise ValueError("indicators must have shape (n,)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(indicators)):
        raise ValueError("points and indicators must be finite")

    non_contact = ~contact
    signs = np.sign(indicators[non_contact])
    if not (np.any(signs < 0.0) and np.any(signs > 0.0)):
        raise ValueError(
            "Contact-constrained modelling still requires Inside and Outside points."
        )

    data_diagonal = float(
        np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    )
    if not data_diagonal > 0.0:
        raise ValueError("point bounding box must have a positive diagonal")

    if fit_accuracy == 0.0:
        fit_accuracy = 1.0e-7 * data_diagonal
    elif not fit_accuracy > 0.0:
        raise ValueError("fit_accuracy must be positive or zero for automatic")

    if clipping_distance == 0.0:
        clipping_distance = 1.0e-2 * data_diagonal
    elif not clipping_distance > 0.0:
        raise ValueError(
            "clipping_distance must be positive or zero for automatic"
        )

    tree = cKDTree(points[contact])
    try:
        distances, _ = tree.query(points[non_contact], k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(points[non_contact], k=1)
    distances = np.asarray(distances, dtype=float)

    raw_signed_distances = np.zeros(len(points), dtype=float)
    raw_signed_distances[non_contact] = -np.sign(indicators[non_contact]) * distances
    values = np.clip(
        raw_signed_distances,
        -float(clipping_distance),
        float(clipping_distance),
    )
    values[contact] = 0.0

    return SimpleNamespace(
        values=values,
        raw_signed_distances=raw_signed_distances,
        data_diagonal=data_diagonal,
        fit_accuracy=float(fit_accuracy),
        clipping_distance=float(clipping_distance),
    )


def _refresh_effective_roles(self: Any) -> None:
    if self.frame is not None:
        self.populate_category_roles(self.category_combo.currentText())


def contact_window_init(self: Any) -> None:
    _original_window_init(self)

    group = app.QtWidgets.QGroupBox("Contact constraints")
    form = app.QtWidgets.QFormLayout(group)

    self.contact_column_combo = app.QtWidgets.QComboBox()
    self.contact_column_combo.addItem(CONTACT_COLUMN_NONE)
    self.contact_zero_tolerance_spin = app.QtWidgets.QDoubleSpinBox()
    self.contact_zero_tolerance_spin.setRange(0.0, 1.0e12)
    self.contact_zero_tolerance_spin.setDecimals(10)
    self.contact_zero_tolerance_spin.setValue(1.0e-8)
    self.contact_zero_tolerance_spin.setKeyboardTracking(False)

    help_label = app.QtWidgets.QLabel(
        "Rows assigned Contact are fitted as exact value 0. Optionally select a "
        "Values/SDF column so near-zero rows become a separate Contact category "
        "even when their original category says Outside or Ignored. Ignore rows "
        "are excluded from the RBF unless promoted to Contact by this override."
    )
    help_label.setWordWrap(True)

    form.addRow("Promote zero values from", self.contact_column_combo)
    form.addRow("Zero tolerance", self.contact_zero_tolerance_spin)
    form.addRow(help_label)

    data_scroll = self.tabs.widget(0)
    data_page = data_scroll.widget() if hasattr(data_scroll, "widget") else None
    if data_page is None or data_page.layout() is None:
        raise RuntimeError("Could not locate the Data-tab layout.")
    data_page.layout().insertWidget(3, group)

    self.contact_column_combo.currentTextChanged.connect(
        lambda _text: _refresh_effective_roles(self)
    )
    self.contact_zero_tolerance_spin.valueChanged.connect(
        lambda _value: _refresh_effective_roles(self)
    )

    for label in self.findChildren(app.QtWidgets.QLabel):
        if "Assign any number of categories" in label.text():
            label.setText(
                "Assign every effective category to Inside, Outside, Contact, "
                "or Ignore. Inside = -1, Outside = +1, Contact = exact 0, and "
                "Ignore is excluded from the RBF."
            )
            break

    self.nugget_spin.setToolTip(
        "Contact-constrained runs force Nugget to 0 so contact points remain "
        "interpolation constraints."
    )


def contact_load_csv(self: Any) -> None:
    _original_load_csv(self)
    if self.frame is None:
        return

    columns = [str(column) for column in self.frame.columns]
    previous = self.contact_column_combo.currentText()
    self.contact_column_combo.blockSignals(True)
    self.contact_column_combo.clear()
    self.contact_column_combo.addItem(CONTACT_COLUMN_NONE)
    self.contact_column_combo.addItems(columns)

    selected = None
    if previous in columns:
        selected = previous
    else:
        for candidate in CONTACT_COLUMN_CANDIDATES:
            for column in columns:
                if column.strip().lower() == candidate.lower():
                    selected = column
                    break
            if selected is not None:
                break

    self.contact_column_combo.setCurrentText(
        selected if selected is not None else CONTACT_COLUMN_NONE
    )
    self.contact_column_combo.blockSignals(False)
    self.populate_category_roles(self.category_combo.currentText())


def contact_run_model(self: Any) -> None:
    try:
        _, indicators, _, _ = self._mapped_arrays()
        contact_count = int(np.count_nonzero(indicators == 0.0))
        if contact_count:
            if self.nugget_spin.value() != 0.0:
                self.nugget_spin.setValue(0.0)
                self._log(
                    "Nugget was forced to 0 because exact Contact constraints are active."
                )
            self._log(
                f"Contact mode active: {contact_count:,} points are exact zero "
                "constraints; automatic fit tolerance uses 1e-7 x data diagonal."
            )
    except Exception as error:
        self._show_error("Could not validate contact constraints", error)
        return
    _original_run_model(self)


def contact_model_finished(self: Any, result: dict[str, Any]) -> None:
    _original_model_finished(self, result)
    indicators = self.current_indicators
    if indicators is None:
        return
    contact = np.asarray(indicators, dtype=float) == 0.0
    if not np.any(contact):
        return
    predictions = np.asarray(result.get("predictions", []), dtype=float)
    if len(predictions) != len(contact):
        return

    errors = np.abs(predictions[contact])
    maximum = float(np.max(errors))
    mean = float(np.mean(errors))
    tolerance = float(result.get("fit_tolerance", np.nan))
    self._log(
        f"Contact snap check: {int(np.count_nonzero(contact)):,} contacts; "
        f"mean |field| {mean:.6g}; maximum |field| {maximum:.6g}; "
        f"fit tolerance {tolerance:.6g}."
    )
    if np.isfinite(tolerance) and maximum > 10.0 * tolerance:
        self._log(
            "Warning: some contact residuals exceed 10 x the requested fitting "
            "tolerance. Increase Maximum iterations before interpreting the surface."
        )


polatory.leapfrog_indicator_values3 = contact_indicator_values3
app.MainWindow.__init__ = contact_window_init
app.MainWindow.load_csv = contact_load_csv
app.MainWindow.populate_category_roles = contact_populate_category_roles
app.MainWindow._mapped_arrays = contact_mapped_arrays
app.MainWindow.apply_data_mapping = contact_apply_data_mapping
app.MainWindow.run_model = contact_run_model
app.MainWindow.model_finished = contact_model_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
