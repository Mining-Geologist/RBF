"""Category-only four-role launcher for the Polatory structural-LVA app.

Every selected category value is treated as categorical, including numeric-looking
values. The user assigns each category manually to one role:

- Inside: inside constraint
- Outside: outside constraint
- Contact: exact zero constraint
- Ignore: excluded from domain construction and RBF fitting

There is no numerical/SDF mode and no automatic promotion from another column.

Run:
    python polatory_lva_pyqt_app_v8_category_contacts.py
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
app.ROLE_CONTACT = ROLE_CONTACT
app.ROLE_OPTIONS = (
    app.ROLE_INSIDE,
    app.ROLE_OUTSIDE,
    ROLE_CONTACT,
    app.ROLE_IGNORE,
)

_original_indicator_values = polatory.leapfrog_indicator_values3
_original_window_init = app.MainWindow.__init__
_original_run_model = app.MainWindow.run_model
_original_model_finished = app.MainWindow.model_finished


def category_default_role(value: str) -> str:
    """Use only obvious text labels; all other categories stay manual/ignored."""
    text = str(value).strip().lower()

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


app.default_role_for_category = category_default_role


def category_populate_roles(self: Any, column: str) -> None:
    if self.frame is None or not column or column not in self.frame.columns:
        self.role_table.setRowCount(0)
        return

    keys = app.category_keys(self.frame[column])
    counts = keys.value_counts(dropna=False, sort=False)
    if len(counts) > 500:
        answer = app.QtWidgets.QMessageBox.question(
            self,
            "Many unique categories",
            f"The selected field contains {len(counts):,} unique values. Every "
            "value will be treated as a category. Populate the table anyway?",
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
            previous.get(value_text, category_default_role(value_text))
        )

        self.role_table.setItem(row, 0, value_item)
        self.role_table.setItem(row, 1, count_item)
        self.role_table.setCellWidget(row, 2, role_combo)


def category_mapped_arrays(
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

    coordinate_frame = self.frame[columns[:3]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    coordinate_array = coordinate_frame.to_numpy(dtype=float)
    finite_coordinates = np.all(np.isfinite(coordinate_array), axis=1)

    category_keys = app.category_keys(self.frame[columns[3]])
    role_mapping = self.category_role_mapping()
    roles = (
        category_keys.map(role_mapping)
        .fillna(app.ROLE_IGNORE)
        .to_numpy(dtype=object)
    )

    modelling_mask = finite_coordinates & (roles != app.ROLE_IGNORE)
    points = coordinate_array[modelling_mask]
    model_roles = roles[modelling_mask]
    source_rows = np.flatnonzero(modelling_mask)

    indicators = np.zeros(len(points), dtype=float)
    indicators[model_roles == app.ROLE_INSIDE] = -1.0
    indicators[model_roles == app.ROLE_OUTSIDE] = 1.0
    indicators[model_roles == ROLE_CONTACT] = 0.0

    if len(points) < 3:
        raise ValueError(
            "At least three non-ignored points with valid coordinates are required."
        )
    if not np.any(indicators < 0.0):
        raise ValueError("Assign at least one category to Inside.")
    if not np.any(indicators > 0.0):
        raise ValueError("Assign at least one category to Outside.")

    return points, indicators, model_roles, source_rows


def category_apply_mapping(self: Any) -> None:
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

        all_coordinates = self.frame[columns := [
            self.x_combo.currentText(),
            self.y_combo.currentText(),
            self.z_combo.currentText(),
        ]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.all(np.isfinite(all_coordinates), axis=1)
        ignored_mask = finite.copy()
        ignored_mask[source_rows] = False
        ignored = all_coordinates[ignored_mask]

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
            f"Mapped {len(points):,} modelling points from categorical roles: "
            f"{len(inside):,} inside, {len(outside):,} outside, "
            f"{len(contacts):,} contacts, and {len(ignored):,} ignored."
        )
    except Exception as error:
        self._show_error("Invalid data mapping", error)


def category_contact_values(
    points: np.ndarray,
    indicators: np.ndarray,
    *,
    fit_accuracy: float = 0.0,
    clipping_distance: float = 0.0,
) -> Any:
    """Build signed distances from categorical Inside/Outside/Contact roles."""
    points = np.asarray(points, dtype=float)
    indicators = np.asarray(indicators, dtype=float)
    contact_mask = indicators == 0.0

    if not np.any(contact_mask):
        return _original_indicator_values(
            points,
            indicators,
            fit_accuracy=fit_accuracy,
            clipping_distance=clipping_distance,
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if indicators.ndim != 1 or len(indicators) != len(points):
        raise ValueError("indicators must have shape (n,)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(indicators)):
        raise ValueError("points and indicators must be finite")

    non_contact = ~contact_mask
    if not np.any(indicators[non_contact] < 0.0):
        raise ValueError("Contact modelling requires at least one Inside point.")
    if not np.any(indicators[non_contact] > 0.0):
        raise ValueError("Contact modelling requires at least one Outside point.")

    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if not diagonal > 0.0:
        raise ValueError("point bounding box must have a positive diagonal")

    if fit_accuracy == 0.0:
        fit_accuracy = 1.0e-7 * diagonal
    elif not fit_accuracy > 0.0:
        raise ValueError("fit_accuracy must be positive or zero for automatic")

    if clipping_distance == 0.0:
        clipping_distance = 1.0e-2 * diagonal
    elif not clipping_distance > 0.0:
        raise ValueError(
            "clipping_distance must be positive or zero for automatic"
        )

    contact_tree = cKDTree(points[contact_mask])
    try:
        distances, _ = contact_tree.query(
            points[non_contact],
            k=1,
            workers=-1,
        )
    except TypeError:
        distances, _ = contact_tree.query(points[non_contact], k=1)

    raw_signed_distances = np.zeros(len(points), dtype=float)
    raw_signed_distances[non_contact] = (
        -np.sign(indicators[non_contact]) * np.asarray(distances, dtype=float)
    )

    values = np.clip(
        raw_signed_distances,
        -float(clipping_distance),
        float(clipping_distance),
    )
    values[contact_mask] = 0.0

    return SimpleNamespace(
        values=values,
        raw_signed_distances=raw_signed_distances,
        data_diagonal=diagonal,
        fit_accuracy=float(fit_accuracy),
        clipping_distance=float(clipping_distance),
    )


def category_window_init(self: Any) -> None:
    _original_window_init(self)

    for label in self.findChildren(app.QtWidgets.QLabel):
        if "Assign any number of categories" in label.text():
            label.setText(
                "Every unique value in the selected category column is treated "
                "as a category. Assign it manually to Inside, Outside, Contact, "
                "or Ignore. Contact is an exact zero constraint; Ignore is not "
                "used by the RBF."
            )
            break

    self.nugget_spin.setToolTip(
        "Runs containing Contact categories force Nugget to 0 so contacts remain "
        "zero-value interpolation constraints."
    )


def category_run_model(self: Any) -> None:
    try:
        _, indicators, _, _ = self._mapped_arrays()
        contact_count = int(np.count_nonzero(indicators == 0.0))
        if contact_count:
            if self.nugget_spin.value() != 0.0:
                self.nugget_spin.setValue(0.0)
                self._log(
                    "Nugget was forced to 0 because Contact categories are active."
                )
            self._log(
                f"Category contact mode: {contact_count:,} rows are exact zero "
                "constraints."
            )
    except Exception as error:
        self._show_error("Could not validate category roles", error)
        return

    _original_run_model(self)


def category_model_finished(self: Any, result: dict[str, Any]) -> None:
    _original_model_finished(self, result)

    if self.current_indicators is None:
        return
    contact_mask = np.asarray(self.current_indicators, dtype=float) == 0.0
    if not np.any(contact_mask):
        return

    predictions = np.asarray(result.get("predictions", []), dtype=float)
    if len(predictions) != len(contact_mask):
        return

    absolute_errors = np.abs(predictions[contact_mask])
    maximum = float(np.max(absolute_errors))
    mean = float(np.mean(absolute_errors))
    tolerance = float(result.get("fit_tolerance", np.nan))

    self._log(
        f"Contact snap check: {int(np.count_nonzero(contact_mask)):,} contacts; "
        f"mean |field| {mean:.6g}; maximum |field| {maximum:.6g}; "
        f"fit tolerance {tolerance:.6g}."
    )
    if np.isfinite(tolerance) and maximum > 10.0 * tolerance:
        self._log(
            "Warning: contact residuals exceed 10 x the requested tolerance. "
            "Increase Maximum iterations before trusting the final surface."
        )


polatory.leapfrog_indicator_values3 = category_contact_values
app.MainWindow.__init__ = category_window_init
app.MainWindow.populate_category_roles = category_populate_roles
app.MainWindow._mapped_arrays = category_mapped_arrays
app.MainWindow.apply_data_mapping = category_apply_mapping
app.MainWindow.run_model = category_run_model
app.MainWindow.model_finished = category_model_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
