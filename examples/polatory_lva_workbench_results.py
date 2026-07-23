"""Generated-surface component filtering and model-completion hooks."""
from polatory_lva_workbench_fields import *

def _inside_supported_surface(self: Any, result: dict[str, Any]) -> tuple[pv.PolyData, dict[str, Any]]:
    """Keep disconnected result components that are supported by mapped data points."""
    name = "Automatic LVA surface"
    if name not in self.layers:
        raise ValueError("The generated surface layer is unavailable.")
    surface = _mesh_surface(_record_dataset(self.layers[name]))
    connected = surface.connectivity()
    if "RegionId" not in connected.cell_data:
        return surface, {"raw_components": 1, "kept_components": 1}

    points = np.asarray(getattr(self, "current_points", np.empty((0, 3))), dtype=float)
    if len(points) == 0:
        return surface, {"raw_components": 1, "kept_components": 1}
    if len(points) > 1:
        nearest = np.asarray(cKDTree(points).query(points, k=2)[0][:, 1], dtype=float)
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        median_spacing = float(np.median(nearest)) if len(nearest) else 1.0
    else:
        median_spacing = 1.0

    resolution = float(result.get("surface_resolution", 0.0) or 0.0)
    if not resolution > 0.0:
        widget = getattr(self, "surface_resolution_spin", None)
        resolution = float(widget.value()) if widget is not None else median_spacing
    threshold = max(3.0 * resolution, 2.0 * median_spacing)
    minimum_support = max(3, int(np.ceil(0.01 * len(points))))

    region_values = np.asarray(connected.cell_data["RegionId"], dtype=np.int64)
    components: list[pv.PolyData] = []
    records: list[dict[str, Any]] = []
    for region_id in np.unique(region_values):
        component = connected.extract_cells(region_values == int(region_id))
        component = component.extract_surface().triangulate().clean()
        measured = pv.PolyData(points).compute_implicit_distance(component)
        distances = np.abs(np.asarray(measured["implicit_distance"], dtype=float))
        support_count = int(np.count_nonzero(distances <= threshold))
        components.append(component)
        records.append(
            {
                "region_id": int(region_id),
                "vertices": int(component.n_points),
                "triangles": int(component.n_cells),
                "support_count": support_count,
                "minimum_distance": float(np.min(distances)),
                "median_distance": float(np.median(distances)),
                "kept": support_count >= minimum_support,
            }
        )

    kept = [index for index, record in enumerate(records) if record["kept"]]
    if not kept:
        closest = int(np.argmin([record["median_distance"] for record in records]))
        records[closest]["kept"] = True
        records[closest]["fallback_closest_component"] = True
        kept = [closest]

    cleaned = components[kept[0]].copy(deep=True)
    for index in kept[1:]:
        cleaned = cleaned.merge(components[index], merge_points=False)
    cleaned = cleaned.extract_surface().triangulate().clean()
    return cleaned, {
        "raw_components": len(records),
        "kept_components": len(kept),
        "support_distance": threshold,
        "minimum_support_points": minimum_support,
        "components": records,
    }


def _replace_generated_with_supported_surface(self: Any, result: dict[str, Any]) -> None:
    try:
        cleaned, report = _inside_supported_surface(self, result)
        if report["raw_components"] == 1:
            self._last_component_filter = report
            return
        self._remove_layer("Automatic LVA surface")
        self._add_layer(
            "Automatic LVA surface",
            cleaned,
            kind="mesh",
            color="#d9d9d9",
            opacity=1.0,
            show_edges=False,
            visible=True,
            select=True,
        )
        self.result_surface = cleaned
        result_path = getattr(self, "result_temp_obj", None)
        if result_path:
            try:
                cleaned.save(Path(result_path))
            except Exception as error:
                self._log(f"Could not overwrite the temporary result OBJ: {error}")
        self._last_component_filter = report
        self._log(
            f"Inside-only result filter kept {report['kept_components']}/"
            f"{report['raw_components']} connected components; unsupported enclosing "
            "shells were removed."
        )
    except Exception as error:
        self._log(f"Inside-only result filtering was skipped: {error}")


def _enhanced_process_model_finished(self: Any, result: dict[str, Any]) -> None:
    _process_model_finished(self, result)
    _replace_generated_with_supported_surface(self, result)
    self._last_workbench_result = result
    _refresh_layer_combos(self)
    self._log(
        "Workbench result layers are available in the layer list: generated surface, "
        "automatic domain points, centroid clusters, LVA field slices and principal axes."
    )


__all__ = [name for name in globals() if not name.startswith("__")]
