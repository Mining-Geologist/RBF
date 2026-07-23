"""Interactive Polatory structural-LVA workbench launcher."""
from __future__ import annotations

from typing import Any
from polatory_lva_workbench_results import *

def _make_field_group(self: Any) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox("Point and orientation field")
    layout = QtWidgets.QVBoxLayout(group)

    top = QtWidgets.QHBoxLayout()
    load_button = QtWidgets.QPushButton("Load field CSV…")
    load_button.clicked.connect(lambda: _load_field_csv(self))
    self.field_path_label = QtWidgets.QLabel("No field CSV loaded")
    self.field_path_label.setWordWrap(True)
    top.addWidget(load_button)
    top.addWidget(self.field_path_label, 1)
    layout.addLayout(top)

    mapping = QtWidgets.QGridLayout()
    self.field_x_combo = QtWidgets.QComboBox()
    self.field_y_combo = QtWidgets.QComboBox()
    self.field_z_combo = QtWidgets.QComboBox()
    self.field_dip_combo = QtWidgets.QComboBox()
    self.field_azimuth_combo = QtWidgets.QComboBox()
    self.field_vector_x_combo = QtWidgets.QComboBox()
    self.field_vector_y_combo = QtWidgets.QComboBox()
    self.field_vector_z_combo = QtWidgets.QComboBox()
    widgets = (
        ("X", self.field_x_combo, 0, 0),
        ("Y", self.field_y_combo, 0, 2),
        ("Z", self.field_z_combo, 0, 4),
        ("Dip", self.field_dip_combo, 1, 0),
        ("Azimuth", self.field_azimuth_combo, 1, 2),
        ("Vector X", self.field_vector_x_combo, 2, 0),
        ("Vector Y", self.field_vector_y_combo, 2, 2),
        ("Vector Z", self.field_vector_z_combo, 2, 4),
    )
    for label, widget, row, column in widgets:
        mapping.addWidget(QtWidgets.QLabel(label), row, column)
        mapping.addWidget(widget, row, column + 1)
    layout.addLayout(mapping)

    display = QtWidgets.QGridLayout()
    self.field_mode_combo = QtWidgets.QComboBox()
    self.field_mode_combo.addItems(
        ["Points", "Scaled spheres", "Dip/Azimuth arrows", "Vector arrows"]
    )
    self.field_scale_combo = QtWidgets.QComboBox()
    self.field_scale_combo.addItem(CONSTANT)
    self.field_color_combo = QtWidgets.QComboBox()
    self.field_color_combo.addItem(UNIFORM)
    self.field_cmap_combo = QtWidgets.QComboBox()
    self.field_cmap_combo.addItems(["viridis", "plasma", "turbo", "coolwarm", "terrain"])
    self.field_glyph_factor_spin = QtWidgets.QDoubleSpinBox()
    self.field_glyph_factor_spin.setRange(1.0e-9, 1.0e12)
    self.field_glyph_factor_spin.setDecimals(6)
    self.field_glyph_factor_spin.setValue(10.0)
    self.field_point_size_spin = QtWidgets.QDoubleSpinBox()
    self.field_point_size_spin.setRange(1.0, 100.0)
    self.field_point_size_spin.setValue(8.0)
    self.field_opacity_spin = QtWidgets.QDoubleSpinBox()
    self.field_opacity_spin.setRange(0.0, 1.0)
    self.field_opacity_spin.setSingleStep(0.05)
    self.field_opacity_spin.setValue(1.0)
    self.field_stride_spin = QtWidgets.QSpinBox()
    self.field_stride_spin.setRange(1, 1_000_000)
    self.field_stride_spin.setValue(1)
    self.field_normalise_scale_check = QtWidgets.QCheckBox("Normalize scale to 0.2–1")
    self.field_normalise_scale_check.setChecked(True)
    self.field_round_points_check = QtWidgets.QCheckBox("Round point sprites")
    self.field_round_points_check.setChecked(True)
    self.field_color_button = QtWidgets.QPushButton()
    _set_button_colour(self.field_color_button, "#36a2eb")
    self.field_color_button.clicked.connect(lambda: _choose_colour(self, self.field_color_button))

    display_rows = (
        ("Display mode", self.field_mode_combo, 0, 0),
        ("Scale variable", self.field_scale_combo, 0, 2),
        ("Colour variable", self.field_color_combo, 0, 4),
        ("Glyph factor", self.field_glyph_factor_spin, 1, 0),
        ("Point size", self.field_point_size_spin, 1, 2),
        ("Opacity", self.field_opacity_spin, 1, 4),
        ("Display stride", self.field_stride_spin, 2, 0),
        ("Colormap", self.field_cmap_combo, 2, 2),
        ("Uniform colour", self.field_color_button, 2, 4),
    )
    for label, widget, row, column in display_rows:
        display.addWidget(QtWidgets.QLabel(label), row, column)
        display.addWidget(widget, row, column + 1)
    display.addWidget(self.field_normalise_scale_check, 3, 0, 1, 3)
    display.addWidget(self.field_round_points_check, 3, 3, 1, 3)
    layout.addLayout(display)

    create_button = QtWidgets.QPushButton("Create field layer")
    create_button.clicked.connect(lambda: _build_field_layer(self))
    layout.addWidget(create_button)
    return group


def _make_mesh_group(self: Any) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox("Comparison meshes and distance")
    layout = QtWidgets.QVBoxLayout(group)
    load_button = QtWidgets.QPushButton("Load comparison mesh…")
    load_button.clicked.connect(lambda: _load_comparison_mesh(self))
    layout.addWidget(load_button)

    form = QtWidgets.QFormLayout()
    self.compare_source_combo = QtWidgets.QComboBox()
    self.compare_target_combo = QtWidgets.QComboBox()
    form.addRow("Source mesh", self.compare_source_combo)
    form.addRow("Target mesh", self.compare_target_combo)
    layout.addLayout(form)
    compare_button = QtWidgets.QPushButton("Calculate surface distances")
    compare_button.clicked.connect(lambda: _compare_meshes(self))
    layout.addWidget(compare_button)
    self.comparison_stats = QtWidgets.QPlainTextEdit()
    self.comparison_stats.setReadOnly(True)
    self.comparison_stats.setMaximumHeight(130)
    layout.addWidget(self.comparison_stats)
    return group


def _make_layer_group(self: Any) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox("Selected layer appearance")
    layout = QtWidgets.QVBoxLayout(group)
    self.selected_layer_label = QtWidgets.QLabel("No layer selected")
    self.selected_layer_label.setWordWrap(True)
    layout.addWidget(self.selected_layer_label)

    form = QtWidgets.QFormLayout()
    self.layer_visible_check = QtWidgets.QCheckBox("Visible")
    self.layer_visible_check.setChecked(True)
    self.layer_opacity_spin = QtWidgets.QDoubleSpinBox()
    self.layer_opacity_spin.setRange(0.0, 1.0)
    self.layer_opacity_spin.setSingleStep(0.05)
    self.layer_representation_combo = QtWidgets.QComboBox()
    self.layer_representation_combo.addItems(
        ["Surface", "Surface + wireframe", "Wireframe", "Points"]
    )
    self.layer_scalar_colors_check = QtWidgets.QCheckBox(
        "Use active scalar colours (disable to apply the uniform colour)"
    )
    self.layer_color_button = QtWidgets.QPushButton()
    _set_button_colour(self.layer_color_button, "#ffffff")
    self.layer_color_button.clicked.connect(lambda: _choose_colour(self, self.layer_color_button))
    self.edge_color_button = QtWidgets.QPushButton()
    _set_button_colour(self.edge_color_button, "#202020")
    self.edge_color_button.clicked.connect(lambda: _choose_colour(self, self.edge_color_button))
    self.layer_line_width_spin = QtWidgets.QDoubleSpinBox()
    self.layer_line_width_spin.setRange(1.0, 20.0)
    self.layer_line_width_spin.setValue(1.0)
    self.layer_point_size_spin = QtWidgets.QDoubleSpinBox()
    self.layer_point_size_spin.setRange(1.0, 100.0)
    self.layer_point_size_spin.setValue(5.0)

    form.addRow(self.layer_visible_check)
    form.addRow("Opacity", self.layer_opacity_spin)
    form.addRow("Representation", self.layer_representation_combo)
    form.addRow(self.layer_scalar_colors_check)
    form.addRow("Solid / point colour", self.layer_color_button)
    form.addRow("Wireframe colour", self.edge_color_button)
    form.addRow("Wireframe width", self.layer_line_width_spin)
    form.addRow("Point size", self.layer_point_size_spin)
    layout.addLayout(form)

    buttons = QtWidgets.QHBoxLayout()
    apply_button = QtWidgets.QPushButton("Apply appearance")
    apply_button.clicked.connect(lambda: _apply_selected_layer_style(self))
    export_button = QtWidgets.QPushButton("Export selected…")
    export_button.clicked.connect(lambda: _export_selected_layer(self))
    remove_button = QtWidgets.QPushButton("Remove imported layer")
    remove_button.clicked.connect(lambda: _remove_selected_workbench_layer(self))
    buttons.addWidget(apply_button)
    buttons.addWidget(export_button)
    buttons.addWidget(remove_button)
    layout.addLayout(buttons)
    return group


def workbench_window_init(self: Any) -> None:
    _original_window_init(self)
    self.setWindowTitle("Polatory Structural LVA Workbench")
    self._field_state = FieldState()
    self._last_workbench_result: dict[str, Any] | None = None

    page = QtWidgets.QWidget()
    page_layout = QtWidgets.QVBoxLayout(page)
    intro = QtWidgets.QLabel(
        "Import comparison meshes and arbitrary point/orientation fields. The normal "
        "Data and Model tabs remain the modelling workflow. Model results already "
        "include automatic domain points, centroid clusters, LVA field slices and "
        "principal-axis glyphs."
    )
    intro.setWordWrap(True)
    page_layout.addWidget(intro)
    page_layout.addWidget(_make_mesh_group(self))
    page_layout.addWidget(_make_field_group(self))
    page_layout.addWidget(_make_layer_group(self))
    page_layout.addStretch(1)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    self.workbench_tab_index = self.tabs.addTab(scroll, "Workbench")

    self.layer_list.currentItemChanged.connect(
        lambda _current, _previous: _sync_selected_layer_controls(self)
    )
    self.layer_list.model().rowsInserted.connect(
        lambda *_args: _refresh_layer_combos(self)
    )
    self.layer_list.model().rowsRemoved.connect(
        lambda *_args: _refresh_layer_combos(self)
    )

    _refresh_layer_combos(self)
    self._log(
        "LVA Workbench loaded: comparison meshes, mesh distances, arbitrary field "
        "CSV glyphs and per-layer appearance controls are ready."
    )


app.MainWindow.__init__ = workbench_window_init
app.MainWindow._add_layer = workbench_add_layer
v10._original_model_finished = _enhanced_process_model_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
