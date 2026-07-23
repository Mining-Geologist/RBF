"""Comparison-mesh and arbitrary field visualization actions."""
from polatory_lva_workbench_common import *

def _load_comparison_mesh(self: Any) -> None:
    path_text, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Load comparison mesh",
        "",
        "Surface meshes (*.obj *.stl *.ply *.vtk *.vtp);;All files (*)",
    )
    if not path_text:
        return
    try:
        path = Path(path_text)
        mesh = pv.read(path).extract_surface().triangulate().clean()
        if mesh.n_points == 0 or mesh.n_cells == 0:
            raise ValueError("The selected file contains no usable surface cells.")
        name = _unique_name(self, f"Comparison: {path.stem}")
        self._add_layer(
            name,
            mesh,
            kind="mesh",
            color="#ff8c42",
            opacity=0.55,
            show_edges=True,
            visible=True,
            select=True,
        )
        self._log(
            f"Loaded comparison mesh '{path.name}': {mesh.n_points:,} vertices, "
            f"{mesh.n_cells:,} triangles."
        )
        _refresh_layer_combos(self)
        self.plotter.reset_camera()
    except Exception as error:
        self._show_error("Could not load comparison mesh", error)


def _compare_meshes(self: Any) -> None:
    source_name = self.compare_source_combo.currentText()
    target_name = self.compare_target_combo.currentText()
    if not source_name or not target_name or source_name == target_name:
        QtWidgets.QMessageBox.information(
            self,
            "Select two meshes",
            "Choose two different surface layers to compare.",
        )
        return
    try:
        source = _mesh_surface(_record_dataset(self.layers[source_name]))
        target = _mesh_surface(_record_dataset(self.layers[target_name]))

        source_distance = source.copy(deep=True)
        source_distance.compute_implicit_distance(target, inplace=True)
        forward = np.abs(np.asarray(source_distance["implicit_distance"], dtype=float))
        source_distance["Distance"] = forward

        target_distance = target.copy(deep=True)
        target_distance.compute_implicit_distance(source, inplace=True)
        reverse = np.abs(np.asarray(target_distance["implicit_distance"], dtype=float))
        symmetric = np.concatenate([forward, reverse])

        name = _unique_name(self, f"Distance: {source_name} to {target_name}")
        self._add_layer(
            name,
            source_distance,
            kind="mesh",
            scalars="Distance",
            cmap="viridis",
            opacity=1.0,
            show_edges=False,
            visible=True,
            select=True,
        )
        stats = {
            "forward mean": float(np.mean(forward)),
            "forward median": float(np.median(forward)),
            "forward p95": float(np.percentile(forward, 95.0)),
            "reverse mean": float(np.mean(reverse)),
            "symmetric mean": float(np.mean(symmetric)),
            "symmetric p95": float(np.percentile(symmetric, 95.0)),
            "symmetric maximum": float(np.max(symmetric)),
        }
        self.comparison_stats.setPlainText(
            "\n".join(f"{key}: {value:.6g}" for key, value in stats.items())
        )
        self._log(
            f"Compared '{source_name}' against '{target_name}': symmetric mean "
            f"{stats['symmetric mean']:.6g}, p95 {stats['symmetric p95']:.6g}."
        )
        _refresh_layer_combos(self)
    except Exception as error:
        self._show_error("Could not compare the selected meshes", error)


def _load_field_csv(self: Any) -> None:
    path_text, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Load point or orientation field",
        "",
        "CSV files (*.csv);;All files (*)",
    )
    if not path_text:
        return
    try:
        path = Path(path_text)
        frame = pd.read_csv(path)
        if len(frame) == 0:
            raise ValueError("The selected CSV contains no rows.")
        self._field_state = FieldState(path=path, frame=frame)
        columns = [str(column) for column in frame.columns]
        numeric = _numeric_columns(frame)

        for combo in (
            self.field_x_combo,
            self.field_y_combo,
            self.field_z_combo,
            self.field_dip_combo,
            self.field_azimuth_combo,
            self.field_vector_x_combo,
            self.field_vector_y_combo,
            self.field_vector_z_combo,
        ):
            combo.clear()
            combo.addItem(NONE)
            combo.addItems(columns)

        self.field_scale_combo.clear()
        self.field_scale_combo.addItem(CONSTANT)
        self.field_scale_combo.addItems(numeric)
        self.field_color_combo.clear()
        self.field_color_combo.addItem(UNIFORM)
        self.field_color_combo.addItems(numeric)

        _set_combo(self.field_x_combo, _preferred(columns, ("X", "xm", "easting", "east")))
        _set_combo(self.field_y_combo, _preferred(columns, ("Y", "ym", "northing", "north")))
        _set_combo(self.field_z_combo, _preferred(columns, ("Z", "zm", "elevation", "rl")))
        _set_combo(self.field_dip_combo, _preferred(columns, ("Dip", "dip_deg", "dip degrees")))
        _set_combo(
            self.field_azimuth_combo,
            _preferred(columns, ("Azimuth", "azimuth_deg", "bearing", "dip direction")),
        )
        _set_combo(self.field_vector_x_combo, _preferred(columns, ("vx", "nx", "vector_x")))
        _set_combo(self.field_vector_y_combo, _preferred(columns, ("vy", "ny", "vector_y")))
        _set_combo(self.field_vector_z_combo, _preferred(columns, ("vz", "nz", "vector_z")))

        self.field_path_label.setText(str(path))
        self._log(
            f"Loaded field CSV '{path.name}': {len(frame):,} rows, "
            f"{len(columns):,} columns."
        )
    except Exception as error:
        self._show_error("Could not load point/orientation field", error)


def _selected_column(combo: QtWidgets.QComboBox) -> str | None:
    value = combo.currentText().strip()
    return None if not value or value in {NONE, CONSTANT, UNIFORM} else value


def _field_arrays(self: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = self._field_state.frame
    if frame is None:
        raise ValueError("Load a field CSV first.")
    coordinate_columns = [
        _selected_column(self.field_x_combo),
        _selected_column(self.field_y_combo),
        _selected_column(self.field_z_combo),
    ]
    if any(column is None for column in coordinate_columns):
        raise ValueError("Map valid X, Y and Z columns.")
    coordinates = frame[list(coordinate_columns)].apply(pd.to_numeric, errors="coerce")
    points = coordinates.to_numpy(dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    stride = int(self.field_stride_spin.value())
    selected = np.flatnonzero(finite)[::stride]
    if len(selected) == 0:
        raise ValueError("No finite mapped coordinates remain after filtering.")
    return frame, points[selected], selected


def _build_field_layer(self: Any) -> None:
    try:
        frame, points, rows = _field_arrays(self)
        mode = self.field_mode_combo.currentText()
        cloud = pv.PolyData(points)

        numeric_columns = _numeric_columns(frame)
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[rows]
            if np.any(np.isfinite(values)):
                cloud[column] = values.astype(np.float32)

        scale_column = _selected_column(self.field_scale_combo)
        if scale_column is None:
            glyph_scale = np.ones(len(points), dtype=np.float32)
        else:
            raw_scale = pd.to_numeric(frame[scale_column], errors="coerce").to_numpy(dtype=float)[rows]
            glyph_scale = (
                _normalised_scale(raw_scale)
                if self.field_normalise_scale_check.isChecked()
                else np.nan_to_num(np.abs(raw_scale), nan=0.0).astype(np.float32)
            )
        cloud["Glyph scale"] = glyph_scale

        color_column = _selected_column(self.field_color_combo)
        layer_dataset: pv.DataSet = cloud
        kind = "points"
        kwargs: dict[str, Any] = {
            "color": _button_colour(self.field_color_button, "#36a2eb"),
            "opacity": float(self.field_opacity_spin.value()),
            "visible": True,
            "select": True,
        }
        if color_column is not None and color_column in cloud.array_names:
            kwargs.pop("color", None)
            kwargs["scalars"] = color_column
            kwargs["cmap"] = self.field_cmap_combo.currentText()

        if mode == "Points":
            kwargs["point_size"] = float(self.field_point_size_spin.value())
            kwargs["render_points_as_spheres"] = self.field_round_points_check.isChecked()
        elif mode == "Scaled spheres":
            if len(points) > 150_000:
                raise ValueError(
                    "Scaled-sphere glyphs are limited to 150,000 displayed rows. "
                    "Increase the display stride."
                )
            sphere = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.0)
            layer_dataset = cloud.glyph(
                scale="Glyph scale",
                geom=sphere,
                factor=float(self.field_glyph_factor_spin.value()),
            )
            kind = "mesh"
        else:
            if mode == "Dip/Azimuth arrows":
                dip_column = _selected_column(self.field_dip_combo)
                azimuth_column = _selected_column(self.field_azimuth_combo)
                if dip_column is None or azimuth_column is None:
                    raise ValueError("Map both Dip and Azimuth columns.")
                dip = pd.to_numeric(frame[dip_column], errors="coerce").to_numpy(dtype=float)[rows]
                azimuth = pd.to_numeric(frame[azimuth_column], errors="coerce").to_numpy(dtype=float)[rows]
                valid = np.isfinite(dip) & np.isfinite(azimuth)
                if not np.all(valid):
                    cloud = cloud.extract_points(valid, adjacent_cells=False)
                    dip = dip[valid]
                    azimuth = azimuth[valid]
                vectors = dip_azimuth_vectors(dip, azimuth)
                cloud["Dip"] = dip.astype(np.float32)
                cloud["Azimuth"] = azimuth.astype(np.float32)
            else:
                vector_columns = [
                    _selected_column(self.field_vector_x_combo),
                    _selected_column(self.field_vector_y_combo),
                    _selected_column(self.field_vector_z_combo),
                ]
                if any(column is None for column in vector_columns):
                    raise ValueError("Map all three vector-component columns.")
                vectors = frame[list(vector_columns)].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)[rows]
                valid = np.all(np.isfinite(vectors), axis=1)
                if not np.all(valid):
                    cloud = cloud.extract_points(valid, adjacent_cells=False)
                    vectors = vectors[valid]
                lengths = np.linalg.norm(vectors, axis=1)
                valid_length = lengths > 0.0
                vectors[valid_length] /= lengths[valid_length, None]
            cloud["Direction"] = vectors.astype(np.float32)
            arrow = pv.Arrow(
                tip_length=0.25,
                tip_radius=0.10,
                shaft_radius=0.025,
            )
            layer_dataset = cloud.glyph(
                orient="Direction",
                scale="Glyph scale",
                factor=float(self.field_glyph_factor_spin.value()),
                geom=arrow,
            )
            kind = "mesh"

        source_name = self._field_state.path.stem if self._field_state.path else "field"
        name = _unique_name(self, f"Field: {source_name} - {mode}")
        self._add_layer(name, layer_dataset, kind=kind, **kwargs)
        self._log(
            f"Created '{name}' from {len(points):,} displayed rows; "
            f"mode={mode}, scale={scale_column or 'constant'}, "
            f"colour={color_column or 'uniform'}."
        )
        _refresh_layer_combos(self)
    except Exception as error:
        self._show_error("Could not create field visualization", error)


def _export_selected_layer(self: Any) -> None:
    name = _selected_layer_name(self)
    if name is None:
        return
    record = self.layers[name]
    dataset = _record_dataset(record)
    path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Export selected layer",
        name.replace(":", "_").replace(" ", "_") + ".vtp",
        "VTK PolyData (*.vtp);;Wavefront OBJ (*.obj);;CSV points (*.csv)",
    )
    if not path_text:
        return
    try:
        path = Path(path_text)
        if path.suffix.casefold() == ".csv":
            surface = dataset.combine() if isinstance(dataset, pv.MultiBlock) else dataset
            data: dict[str, Any] = {
                "x": np.asarray(surface.points)[:, 0],
                "y": np.asarray(surface.points)[:, 1],
                "z": np.asarray(surface.points)[:, 2],
            }
            for array_name in surface.point_data.keys():
                array = np.asarray(surface.point_data[array_name])
                if array.ndim == 1 and len(array) == surface.n_points:
                    data[str(array_name)] = array
            pd.DataFrame(data).to_csv(path, index=False)
        else:
            surface = _mesh_surface(dataset)
            surface.save(path)
        self._log(f"Exported layer '{name}' to {path}.")
    except Exception as error:
        self._show_error("Could not export selected layer", error)


__all__ = [name for name in globals() if not name.startswith("__")]
