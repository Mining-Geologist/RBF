"""Polatory structural-LVA Workbench with an interactive partition explorer.

Run from the repository virtual environment::

    python examples/polatory_lva_workbench_partitions.py

This launcher keeps every feature of ``polatory_lva_workbench.py`` and adds a
Partitions tab after each successful modelling run. Final automatic partitions
can be displayed together or isolated one at a time. Each partition contains
three independently toggleable diagnostics:

* modelling input points assigned to that partition;
* centroid-grid cells assigned to that partition;
* an axis-aligned diagnostic envelope around the displayed partition points.

The generated implicit surface is a blend of the structural domains, so it is
not split into mutually exclusive per-partition surface pieces. The explorer
shows the actual point and centroid assignments used by the automatic
SubDomainer and is intended for diagnosing partition edges and support coverage.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

# Importing the normal Workbench first installs its exact/finite domain selector,
# isolated worker, component filter, comparison tools and PyVista background switch.
import polatory_lva_workbench as workbench
from polatory_lva_workbench_common import QtCore, QtGui, QtWidgets, app, v10


_USER_ROLE = QtCore.Qt.ItemDataRole.UserRole
_CHECKED = QtCore.Qt.CheckState.Checked
_PARTIAL = QtCore.Qt.CheckState.PartiallyChecked
_UNCHECKED = QtCore.Qt.CheckState.Unchecked

_previous_window_init = app.MainWindow.__init__
_previous_process_model_finished = v10._original_model_finished
_previous_run_model = app.MainWindow.run_model


def _partition_colour(index: int, count: int) -> str:
    """Return a stable, well-separated Qt colour for one partition."""
    count = max(int(count), 1)
    hue = int(round((359.0 * int(index)) / count)) % 360
    colour = QtGui.QColor.fromHsv(hue, 185, 215)
    return colour.name()


def _checkable(item: QtWidgets.QTreeWidgetItem) -> None:
    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(0, _UNCHECKED)


def _actor_visibility(actor: Any, visible: bool) -> None:
    try:
        actor.SetVisibility(bool(visible))
        return
    except Exception:
        pass
    try:
        actor.visibility = bool(visible)
    except Exception:
        pass


def _remove_partition_actors(self: Any) -> None:
    groups = getattr(self, "_partition_actor_groups", {})
    for components in groups.values():
        for actors in components.values():
            for actor in actors:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except TypeError:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                except Exception:
                    pass
    self._partition_actor_groups = {}
    try:
        self.plotter.render()
    except Exception:
        pass


def _add_point_actor(
    self: Any,
    points: np.ndarray,
    *,
    colour: str,
    point_size: float,
    opacity: float,
    name: str,
) -> Any | None:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    data = pv.PolyData(points)
    actor = self.plotter.add_mesh(
        data,
        style="points",
        color=colour,
        point_size=float(point_size),
        opacity=float(opacity),
        render_points_as_spheres=True,
        show_scalar_bar=False,
        name=name,
        pickable=False,
        render=False,
    )
    _actor_visibility(actor, False)
    return actor


def _add_envelope_actor(
    self: Any,
    points: np.ndarray,
    *,
    colour: str,
    name: str,
) -> Any | None:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    padding = max(1.0e-6 * max(diagonal, 1.0), 1.0e-6)
    flat = maximum <= minimum
    minimum[flat] -= padding
    maximum[flat] += padding

    box = pv.Box(
        bounds=(
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
            float(minimum[2]),
            float(maximum[2]),
        )
    )
    actor = self.plotter.add_mesh(
        box,
        style="wireframe",
        color=colour,
        line_width=2.0,
        opacity=0.7,
        show_scalar_bar=False,
        name=name,
        pickable=False,
        render=False,
    )
    _actor_visibility(actor, False)
    return actor


def _partition_metadata(item: QtWidgets.QTreeWidgetItem) -> tuple[Any, ...]:
    value = item.data(0, _USER_ROLE)
    return tuple(value) if isinstance(value, (tuple, list)) else tuple()


def _set_component_visible(self: Any, label: int, component: str, visible: bool) -> None:
    groups = getattr(self, "_partition_actor_groups", {})
    for actor in groups.get(int(label), {}).get(str(component), []):
        _actor_visibility(actor, visible)


def _apply_partition_tree_state(
    self: Any,
    item: QtWidgets.QTreeWidgetItem,
    state: QtCore.Qt.CheckState,
) -> None:
    """Apply a check state recursively and update the mapped PyVista actors."""
    item.setCheckState(0, state)
    metadata = _partition_metadata(item)
    kind = metadata[0] if metadata else ""

    if kind == "component":
        _set_component_visible(self, int(metadata[1]), str(metadata[2]), state == _CHECKED)
        return

    for index in range(item.childCount()):
        _apply_partition_tree_state(self, item.child(index), state)


def _aggregate_child_state(item: QtWidgets.QTreeWidgetItem) -> QtCore.Qt.CheckState:
    if item.childCount() == 0:
        return item.checkState(0)
    states = [item.child(index).checkState(0) for index in range(item.childCount())]
    if all(state == _CHECKED for state in states):
        return _CHECKED
    if all(state == _UNCHECKED for state in states):
        return _UNCHECKED
    return _PARTIAL


def _sync_partition_parent_states(self: Any, item: QtWidgets.QTreeWidgetItem | None) -> None:
    while item is not None:
        item.setCheckState(0, _aggregate_child_state(item))
        item = item.parent()


def _partition_item_changed(
    self: Any,
    item: QtWidgets.QTreeWidgetItem,
    column: int,
) -> None:
    if column != 0 or bool(getattr(self, "_partition_tree_guard", False)):
        return
    state = item.checkState(0)
    if state == _PARTIAL:
        return

    self._partition_tree_guard = True
    try:
        _apply_partition_tree_state(self, item, state)
        _sync_partition_parent_states(self, item.parent())
    finally:
        self._partition_tree_guard = False
    try:
        self.plotter.render()
    except Exception:
        pass


def _partition_label_from_item(item: QtWidgets.QTreeWidgetItem | None) -> int | None:
    while item is not None:
        metadata = _partition_metadata(item)
        if metadata and metadata[0] in {"partition", "component"}:
            return int(metadata[1])
        item = item.parent()
    return None


def _show_all_partitions(self: Any) -> None:
    root = getattr(self, "partition_root_item", None)
    if root is None:
        return
    self._partition_tree_guard = True
    try:
        _apply_partition_tree_state(self, root, _CHECKED)
    finally:
        self._partition_tree_guard = False
    self.plotter.render()


def _hide_all_partitions(self: Any) -> None:
    root = getattr(self, "partition_root_item", None)
    if root is None:
        return
    self._partition_tree_guard = True
    try:
        _apply_partition_tree_state(self, root, _UNCHECKED)
    finally:
        self._partition_tree_guard = False
    self.plotter.render()


def _isolate_selected_partition(self: Any) -> None:
    item = self.partition_tree.currentItem()
    label = _partition_label_from_item(item)
    if label is None:
        QtWidgets.QMessageBox.information(
            self,
            "Select a partition",
            "Select a Partition item or one of its diagnostic children first.",
        )
        return

    _hide_all_partitions(self)
    partition_item = self._partition_items.get(int(label))
    if partition_item is None:
        return
    self._partition_tree_guard = True
    try:
        _apply_partition_tree_state(self, partition_item, _CHECKED)
        _sync_partition_parent_states(self, partition_item.parent())
    finally:
        self._partition_tree_guard = False
    self.partition_tree.setCurrentItem(partition_item)
    partition_item.setExpanded(True)
    self.plotter.render()


def _build_partition_explorer(self: Any, result: dict[str, Any]) -> None:
    _remove_partition_actors(self)
    self._partition_tree_guard = True
    try:
        self.partition_tree.clear()
        self._partition_items = {}

        data_points = np.asarray(
            getattr(self, "current_points", np.empty((0, 3))),
            dtype=float,
        )
        data_labels = np.asarray(result.get("labels", []), dtype=np.int64)
        centroid_points = np.asarray(result.get("centroid_points", []), dtype=float)
        centroid_labels = np.asarray(result.get("centroid_labels", []), dtype=np.int64)

        if data_points.ndim != 2 or data_points.shape[1] != 3:
            data_points = np.empty((0, 3), dtype=float)
        if centroid_points.ndim != 2 or centroid_points.shape[1] != 3:
            centroid_points = np.empty((0, 3), dtype=float)
        if data_labels.shape != (len(data_points),):
            data_labels = np.empty(0, dtype=np.int64)
        if centroid_labels.shape != (len(centroid_points),):
            centroid_labels = np.empty(0, dtype=np.int64)

        labels: set[int] = set()
        labels.update(int(value) for value in np.unique(data_labels) if int(value) >= 0)
        labels.update(int(value) for value in np.unique(centroid_labels) if int(value) >= 0)
        ordered_labels = sorted(labels)

        root = QtWidgets.QTreeWidgetItem(
            [f"Partitions ({len(ordered_labels)})", "", ""]
        )
        root.setData(0, _USER_ROLE, ("root",))
        _checkable(root)
        self.partition_tree.addTopLevelItem(root)
        self.partition_root_item = root

        for colour_index, label in enumerate(ordered_labels):
            colour = _partition_colour(colour_index, len(ordered_labels))
            owned_data = (
                data_points[data_labels == label]
                if len(data_labels)
                else np.empty((0, 3), dtype=float)
            )
            owned_centroids = (
                centroid_points[centroid_labels == label]
                if len(centroid_labels)
                else np.empty((0, 3), dtype=float)
            )
            envelope_points = np.vstack(
                [part for part in (owned_data, owned_centroids) if len(part)]
            ) if len(owned_data) or len(owned_centroids) else np.empty((0, 3), dtype=float)

            partition_item = QtWidgets.QTreeWidgetItem(
                [
                    f"Partition {label + 1}",
                    f"{len(owned_data):,}",
                    f"{len(owned_centroids):,}",
                ]
            )
            partition_item.setData(0, _USER_ROLE, ("partition", int(label)))
            partition_item.setToolTip(
                0,
                "Final automatic SubDomainer partition used by the structural RBF fit.",
            )
            partition_item.setForeground(0, QtGui.QBrush(QtGui.QColor(colour)))
            _checkable(partition_item)
            root.addChild(partition_item)
            self._partition_items[int(label)] = partition_item

            components = (
                ("data", "Input points", len(owned_data)),
                ("centroids", "Centroid partition cells", len(owned_centroids)),
                ("envelope", "Diagnostic envelope", len(envelope_points)),
            )
            for component, title, count in components:
                child = QtWidgets.QTreeWidgetItem([title, f"{count:,}", ""])
                child.setData(
                    0,
                    _USER_ROLE,
                    ("component", int(label), str(component)),
                )
                _checkable(child)
                partition_item.addChild(child)

            data_actor = _add_point_actor(
                self,
                owned_data,
                colour=colour,
                point_size=11.0,
                opacity=0.95,
                name=f"partition_{label}_data",
            )
            centroid_actor = _add_point_actor(
                self,
                owned_centroids,
                colour=colour,
                point_size=5.0,
                opacity=0.55,
                name=f"partition_{label}_centroids",
            )
            envelope_actor = _add_envelope_actor(
                self,
                envelope_points,
                colour=colour,
                name=f"partition_{label}_envelope",
            )
            self._partition_actor_groups[int(label)] = {
                "data": [actor for actor in (data_actor,) if actor is not None],
                "centroids": [actor for actor in (centroid_actor,) if actor is not None],
                "envelope": [actor for actor in (envelope_actor,) if actor is not None],
            }

        root.setExpanded(True)
        self.partition_tree.resizeColumnToContents(0)
        self.partition_tree.resizeColumnToContents(1)
        self.partition_tree.resizeColumnToContents(2)
        self.partition_summary_label.setText(
            f"{len(ordered_labels):,} final partitions; "
            f"{len(data_points):,} assigned input points; "
            f"{len(centroid_points):,} assigned centroid cells."
        )
    finally:
        self._partition_tree_guard = False

    self._log(
        "Partition explorer updated. Expand Partitions, then check one partition or the "
        "root item to display all final automatic partition diagnostics."
    )
    try:
        self.plotter.render()
    except Exception:
        pass


def _make_partition_tab(self: Any) -> None:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)

    explanation = QtWidgets.QLabel(
        "These are the final automatic SubDomainer partitions used by the generated "
        "structural RBF. Expand a partition to toggle its assigned input points, centroid "
        "cells and diagnostic envelope. Check the root Partitions item to plot all of "
        "them together. The final implicit mesh is blended across domains and therefore "
        "is not divided into exclusive per-partition surface pieces."
    )
    explanation.setWordWrap(True)
    layout.addWidget(explanation)

    self.partition_summary_label = QtWidgets.QLabel("Run a model to populate partitions.")
    self.partition_summary_label.setWordWrap(True)
    layout.addWidget(self.partition_summary_label)

    buttons = QtWidgets.QHBoxLayout()
    show_all = QtWidgets.QPushButton("Show all")
    hide_all = QtWidgets.QPushButton("Hide all")
    isolate = QtWidgets.QPushButton("Isolate selected")
    show_all.clicked.connect(lambda: _show_all_partitions(self))
    hide_all.clicked.connect(lambda: _hide_all_partitions(self))
    isolate.clicked.connect(lambda: _isolate_selected_partition(self))
    buttons.addWidget(show_all)
    buttons.addWidget(hide_all)
    buttons.addWidget(isolate)
    buttons.addStretch(1)
    layout.addLayout(buttons)

    self.partition_tree = QtWidgets.QTreeWidget()
    self.partition_tree.setHeaderLabels(["Partition / diagnostic", "Input", "Centroids"])
    self.partition_tree.setAlternatingRowColors(True)
    self.partition_tree.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )
    self.partition_tree.itemChanged.connect(
        lambda item, column: _partition_item_changed(self, item, int(column))
    )
    self.partition_tree.itemDoubleClicked.connect(
        lambda _item, _column: _isolate_selected_partition(self)
    )
    layout.addWidget(self.partition_tree, 1)

    self.partition_tab_index = self.tabs.addTab(page, "Partitions")


def partition_window_init(self: Any) -> None:
    _previous_window_init(self)
    self._partition_actor_groups: dict[int, dict[str, list[Any]]] = {}
    self._partition_items: dict[int, QtWidgets.QTreeWidgetItem] = {}
    self._partition_tree_guard = False
    self.partition_root_item: QtWidgets.QTreeWidgetItem | None = None
    _make_partition_tab(self)
    self._log(
        "Partition-explorer launcher loaded. A successful modelling run will populate "
        "a nested Partitions tree for one-by-one or all-partition plotting."
    )


def partition_run_model(self: Any) -> None:
    _remove_partition_actors(self)
    tree = getattr(self, "partition_tree", None)
    if tree is not None:
        tree.clear()
    summary = getattr(self, "partition_summary_label", None)
    if summary is not None:
        summary.setText("Modelling is running; partitions will appear after completion.")
    _previous_run_model(self)


def partition_process_model_finished(self: Any, result: dict[str, Any]) -> None:
    _previous_process_model_finished(self, result)
    try:
        _build_partition_explorer(self, result)
    except Exception as error:
        self._log(f"Partition explorer could not be populated: {error}")


app.MainWindow.__init__ = partition_window_init
app.MainWindow.run_model = partition_run_model
v10._original_model_finished = partition_process_model_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
