"""Analyze the detached/necked basal sheet in an existing benchmark mesh.

This is intentionally a fast, post-meshing diagnostic. It does not rebuild the RBF or
modify the OBJ. It reports whether the low flat feature is an independent connected
component or part of the main body, together with its bounds, aspect ratio, area, and
nearest distance to the input data. The implementation avoids extracting one VTK mesh
per component because that path can terminate the Python process on some Windows/VTK
builds.
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
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    mesh = mesh.triangulate().clean()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh is empty: {path}")
    return mesh


def _bounds(points: np.ndarray) -> tuple[list[float], list[float], np.ndarray]:
    minimum_array = np.min(points, axis=0)
    maximum_array = np.max(points, axis=0)
    span = maximum_array - minimum_array
    return minimum_array.tolist(), maximum_array.tolist(), span


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def main() -> int:
    print(f"Reading generated mesh: {MESH_PATH}", flush=True)
    generated = _surface(MESH_PATH)
    print(f"Reading Leapfrog reference: {REFERENCE_PATH}", flush=True)
    reference = _surface(REFERENCE_PATH)

    frame = pd.read_csv(DATA_DIR / "Used Data(1).csv")
    data_points = frame[["xe", "ye", "ze"]].to_numpy(dtype=float)
    tree = cKDTree(data_points)

    print("Labelling connected surface regions…", flush=True)
    labelled = generated.connectivity(extraction_mode="all", label_regions=True)
    if "RegionId" not in labelled.cell_data:
        raise RuntimeError("PyVista connectivity did not produce cell RegionId labels.")

    points = np.asarray(labelled.points, dtype=float)
    face_stream = np.asarray(labelled.faces, dtype=np.int64)
    if face_stream.size % 4 != 0:
        raise RuntimeError("Expected a triangulated PolyData face stream.")
    packed_faces = face_stream.reshape(-1, 4)
    if not np.all(packed_faces[:, 0] == 3):
        raise RuntimeError("Expected only triangular cells after triangulation.")
    triangles = packed_faces[:, 1:]

    region_labels = np.asarray(labelled.cell_data["RegionId"], dtype=np.int64)
    if len(region_labels) != len(triangles):
        raise RuntimeError("Region labels do not match the triangulated cell count.")
    region_ids = np.unique(region_labels)

    components: list[dict[str, object]] = []
    for region_id in region_ids:
        triangle_ids = np.flatnonzero(region_labels == region_id)
        region_triangles = triangles[triangle_ids]
        point_ids = np.unique(region_triangles)
        region_points = points[point_ids]
        minimum, maximum, span = _bounds(region_points)
        horizontal_span = float(np.hypot(span[0], span[1]))
        flatness = float(span[2] / max(horizontal_span, 1.0e-12))

        stride = max(1, len(region_points) // 20_000)
        sampled = region_points[::stride]
        distances, _ = tree.query(sampled, k=1)
        components.append(
            {
                "region_id": int(region_id),
                "vertices": int(len(point_ids)),
                "triangles": int(len(region_triangles)),
                "area": _triangle_area(points, region_triangles),
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

    generated_min, generated_max, generated_span = _bounds(
        np.asarray(generated.points, dtype=float)
    )
    reference_min, reference_max, _ = _bounds(np.asarray(reference.points, dtype=float))
    low_threshold = float(reference_min[2])
    point_z = np.asarray(generated.points[:, 2], dtype=float)
    low_fraction = float(np.mean(point_z < low_threshold))

    cell_centers = points[triangles].mean(axis=1)
    low_cells = cell_centers[:, 2] < low_threshold
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
