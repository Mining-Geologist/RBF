"""Analyze the detached/necked basal sheet in an existing benchmark mesh.

This is intentionally a fast, post-meshing diagnostic.  It does not rebuild the RBF or
modify the OBJ.  It reports whether the low flat feature is an independent connected
component or part of the main body, together with its bounds, aspect ratio, area, and
nearest distance to the input data.  That distinction determines whether a conservative
component filter is safe or whether the scalar-field completion itself must be changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CASE = os.environ.get("POLATORY_BENCHMARK_CASE", "S5_R300").strip().upper()
DATA_DIR = Path(
    os.environ.get(
        "LEAPFROG_GOLD_DIR",
        str(ROOT / "benchmark-data" / "leapfrog-benchmark-data-v1"),
    )
)
MESH_PATH = (
    ROOT
    / "benchmark-results"
    / "diagnostic-no-background-blending-zero-plateau-padded-bottom"
    / "meshes"
    / f"{CASE}_polatory.obj"
)
REFERENCE_PATH = DATA_DIR / f"{CASE}.obj"
OUTPUT_PATH = MESH_PATH.with_name(f"{CASE}_basal_artifact_analysis.json")


def _surface(path: Path) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(path)
    return pv.read(path).extract_surface().triangulate().clean()


def _bounds(mesh: pv.PolyData) -> tuple[list[float], list[float], np.ndarray]:
    minimum = [float(mesh.bounds[0]), float(mesh.bounds[2]), float(mesh.bounds[4])]
    maximum = [float(mesh.bounds[1]), float(mesh.bounds[3]), float(mesh.bounds[5])]
    return minimum, maximum, np.asarray(maximum, dtype=float) - np.asarray(minimum, dtype=float)


def main() -> int:
    generated = _surface(MESH_PATH)
    reference = _surface(REFERENCE_PATH)
    frame = pd.read_csv(DATA_DIR / "Used Data(1).csv")
    data_points = frame[["xe", "ye", "ze"]].to_numpy(dtype=float)
    tree = cKDTree(data_points)

    labelled = generated.connectivity(extraction_mode="all", label_regions=True)
    if "RegionId" not in labelled.cell_data:
        raise RuntimeError("PyVista connectivity did not produce cell RegionId labels.")
    region_labels = np.asarray(labelled.cell_data["RegionId"], dtype=np.int64)
    region_ids = np.unique(region_labels)

    components: list[dict[str, object]] = []
    for region_id in region_ids:
        cell_ids = np.flatnonzero(region_labels == region_id)
        component = labelled.extract_cells(cell_ids).extract_surface().triangulate().clean()
        minimum, maximum, span = _bounds(component)
        horizontal_span = float(np.hypot(span[0], span[1]))
        flatness = float(span[2] / max(horizontal_span, 1.0e-12))

        points = np.asarray(component.points, dtype=float)
        stride = max(1, len(points) // 20_000)
        sampled = points[::stride]
        distances, _ = tree.query(sampled, k=1)
        components.append(
            {
                "region_id": int(region_id),
                "vertices": int(component.n_points),
                "triangles": int(component.n_cells),
                "area": float(component.area),
                "bounds_min": minimum,
                "bounds_max": maximum,
                "span": span.tolist(),
                "vertical_to_horizontal_span": flatness,
                "nearest_input_distance_min": float(np.min(distances)),
                "nearest_input_distance_median": float(np.median(distances)),
                "nearest_input_distance_p95": float(np.percentile(distances, 95.0)),
            }
        )

    components.sort(key=lambda item: float(item["area"]), reverse=True)
    for rank, component in enumerate(components, start=1):
        component["area_rank"] = rank

    generated_min, generated_max, generated_span = _bounds(generated)
    reference_min, reference_max, _ = _bounds(reference)
    low_threshold = float(reference_min[2])
    point_z = np.asarray(generated.points[:, 2], dtype=float)
    low_fraction = float(np.mean(point_z < low_threshold))

    # Identify which connectivity regions own cells whose centres lie below the
    # Leapfrog reference's minimum elevation.  One region means the basal sheet is
    # attached to that body; a separate region means it can be removed conservatively.
    centers = np.asarray(labelled.cell_centers().points, dtype=float)
    low_cells = centers[:, 2] < low_threshold
    low_region_ids = sorted(np.unique(region_labels[low_cells]).astype(int).tolist())

    report = {
        "case": CASE,
        "generated_obj": str(MESH_PATH),
        "reference_obj": str(REFERENCE_PATH),
        "generated_bounds_min": generated_min,
        "generated_bounds_max": generated_max,
        "generated_span": generated_span.tolist(),
        "reference_bounds_min": reference_min,
        "reference_bounds_max": reference_max,
        "input_bounds_min": np.min(data_points, axis=0).tolist(),
        "input_bounds_max": np.max(data_points, axis=0).tolist(),
        "connectivity_component_count": int(len(components)),
        "generated_vertex_fraction_below_reference_min_z": low_fraction,
        "regions_with_cell_centres_below_reference_min_z": low_region_ids,
        "components_by_area": components,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved diagnostic: {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
