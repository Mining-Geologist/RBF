"""Run the basal-range diagnostic with the recovered Leapfrog LVA sampler forced.

Alongside each generated benchmark OBJ, export CSV files describing the actual
finite structural domains, the point/centroid partition, and the recovered LVA
field on a regular model grid.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import polatory
import polatory.automatic_domain_builder as automatic_module  # noqa: E402


def _lva_components(
    points: np.ndarray,
    input_: Any,
    *,
    non_decaying: bool = False,
) -> dict[str, np.ndarray]:
    """Sample the recovered single-mesh Leapfrog LVA field."""
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(input_.vertices, dtype=np.float64)
    faces = np.asarray(input_.faces, dtype=np.int64)
    strength = float(input_.strength)
    range_ = float(input_.range)

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 0.0
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0

    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 0.0
    vertex_normals[valid] /= lengths[valid, None]
    vertex_normals[~valid] = np.array([0.0, 0.0, 1.0])

    tree = cKDTree(vertices)
    try:
        distance, nearest = tree.query(points, k=1, workers=-1)
    except TypeError:
        distance, nearest = tree.query(points, k=1)
    distance = np.asarray(distance, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)

    if non_decaying:
        q = np.ones(len(points), dtype=np.float64)
        inside_cutoff = np.ones(len(points), dtype=bool)
    else:
        q = np.exp(-distance / range_)
        inside_cutoff = distance < 4.0 * range_
        q[~inside_cutoff] = 0.0

    ratio = 1.0 + (strength - 1.0) * q
    normals = vertex_normals[nearest]
    projectors = normals[:, :, None] * normals[:, None, :]
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    tangent = ratio ** (-1.0 / 3.0)
    normal = ratio ** (2.0 / 3.0)
    matrices = (
        tangent[:, None, None] * (identity - projectors)
        + normal[:, None, None] * projectors
    )
    eigenvalues = np.linalg.eigvalsh(matrices)

    return {
        "nearest": nearest,
        "distance": distance,
        "inside_cutoff": inside_cutoff,
        "q": q,
        "ratio": ratio,
        "normals": normals,
        "matrices": matrices,
        "eigenvalues": eigenvalues,
        "glyph_major": 4.0 * q * ratio ** (1.0 / 3.0),
        "glyph_minor": 4.0 * q / ratio ** (2.0 / 3.0),
    }


def exact_leapfrog_single_input_anisotropies3(
    points: np.ndarray,
    input_: Any,
    *,
    non_decaying: bool = False,
) -> np.ndarray:
    return _lva_components(
        points,
        input_,
        non_decaying=non_decaying,
    )["matrices"]


# Patch both paths: the automatic builder and finite-geodesic propagation worker.
automatic_module.sample_single_input_anisotropies3 = (
    exact_leapfrog_single_input_anisotropies3
)
polatory.sample_single_input_anisotropies3 = (
    exact_leapfrog_single_input_anisotropies3
)

import run_basal_range_diagnostic as diagnostic  # noqa: E402

suite = diagnostic.suite
suite.OUTPUT_DIR = suite.ROOT / "benchmark-results" / "exact-leapfrog-lva-forced"
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"
diagnostic.DIAGNOSTIC_DIR = suite.OUTPUT_DIR / "basal-diagnostics"
CSV_DIR = suite.OUTPUT_DIR / "inspection-csv"
LVA_GRID_DIMENSION = int(os.environ.get("POLATORY_LVA_EXPORT_GRID_DIMENSION", "25"))
if not 2 <= LVA_GRID_DIMENSION <= 100:
    raise ValueError("POLATORY_LVA_EXPORT_GRID_DIMENSION must be between 2 and 100")


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")
    return path


def _indices_text(indices: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in np.asarray(indices).reshape(-1))


def _domain_csvs(case: Any, points: np.ndarray) -> dict[str, Path]:
    builder = diagnostic._CAPTURE.get("builder")
    domains = diagnostic._CAPTURE.get("domains")
    if builder is None or not domains:
        raise RuntimeError("The benchmark did not capture automatic domains")
    details = builder.diagnostics_
    if details is None:
        raise RuntimeError("Automatic domain diagnostics are unavailable")

    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(builder.labels_, dtype=np.int64)
    centroid_points = np.asarray(details.centroid_points, dtype=np.float64)
    centroid_labels = np.asarray(details.centroid_labels, dtype=np.int64)
    postcluster = list(details.postcluster)

    summary_rows: list[dict[str, Any]] = []
    for domain_id, domain in enumerate(domains):
        recovered = postcluster[domain_id] if domain_id < len(postcluster) else None
        actual_min = np.asarray(domain.bbox_min, dtype=np.float64)
        actual_max = np.asarray(domain.bbox_max, dtype=np.float64)
        matrix = np.asarray(domain.anisotropy, dtype=np.float64)
        eig = np.linalg.eigvalsh(matrix)
        support = np.asarray(domain.support_indices, dtype=np.int64)
        core = np.flatnonzero(labels == domain_id).astype(np.int64)
        recovered_min = (
            np.asarray(recovered.bbox_min, dtype=np.float64)
            if recovered is not None
            else np.full(3, np.nan)
        )
        recovered_max = (
            np.asarray(recovered.bbox_max, dtype=np.float64)
            if recovered is not None
            else np.full(3, np.nan)
        )
        row: dict[str, Any] = {
            "case": case.name,
            "domain_id": domain_id,
            "core_point_count": len(core),
            "support_point_count": len(support),
            "centroid_count": int(np.count_nonzero(centroid_labels == domain_id)),
            "core_indices": _indices_text(core),
            "support_indices": _indices_text(support),
            "actual_bbox_min_x": actual_min[0],
            "actual_bbox_min_y": actual_min[1],
            "actual_bbox_min_z": actual_min[2],
            "actual_bbox_max_x": actual_max[0],
            "actual_bbox_max_y": actual_max[1],
            "actual_bbox_max_z": actual_max[2],
            "recovered_bbox_min_x": recovered_min[0],
            "recovered_bbox_min_y": recovered_min[1],
            "recovered_bbox_min_z": recovered_min[2],
            "recovered_bbox_max_x": recovered_max[0],
            "recovered_bbox_max_y": recovered_max[1],
            "recovered_bbox_max_z": recovered_max[2],
            "extension_below_x": recovered_min[0] - actual_min[0],
            "extension_below_y": recovered_min[1] - actual_min[1],
            "extension_below_z": recovered_min[2] - actual_min[2],
            "extension_above_x": actual_max[0] - recovered_max[0],
            "extension_above_y": actual_max[1] - recovered_max[1],
            "extension_above_z": actual_max[2] - recovered_max[2],
            "desired_internal_radius": (
                float(recovered.desired_internal_radius)
                if recovered is not None
                else np.nan
            ),
            "internal_radius": (
                float(recovered.internal_radius) if recovered is not None else np.nan
            ),
            "local_kernel_range": (
                float(recovered.local_kernel_range)
                if recovered is not None
                else np.nan
            ),
            "anisotropy_ratio": eig[-1] / eig[0],
            "matrix_determinant": np.linalg.det(matrix),
            "matrix_00": matrix[0, 0],
            "matrix_01": matrix[0, 1],
            "matrix_02": matrix[0, 2],
            "matrix_10": matrix[1, 0],
            "matrix_11": matrix[1, 1],
            "matrix_12": matrix[1, 2],
            "matrix_20": matrix[2, 0],
            "matrix_21": matrix[2, 1],
            "matrix_22": matrix[2, 2],
        }
        for index, value in enumerate(np.asarray(domain.model_parameters).reshape(-1)):
            row[f"model_parameter_{index}"] = float(value)
        summary_rows.append(row)

    own_support = np.zeros(len(points), dtype=bool)
    inside_own_bbox = np.zeros(len(points), dtype=bool)
    for domain_id, domain in enumerate(domains):
        owned = labels == domain_id
        support = np.asarray(domain.support_indices, dtype=np.int64)
        support_mask = np.zeros(len(points), dtype=bool)
        support_mask[support] = True
        own_support[owned] = support_mask[owned]
        minimum = np.asarray(domain.bbox_min, dtype=np.float64)
        maximum = np.asarray(domain.bbox_max, dtype=np.float64)
        inside = np.all(points >= minimum, axis=1) & np.all(points <= maximum, axis=1)
        inside_own_bbox[owned] = inside[owned]

    point_frame = pd.DataFrame(
        {
            "case": case.name,
            "point_index": np.arange(len(points), dtype=np.int64),
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "automatic_domain": labels,
            "is_support_of_own_domain": own_support,
            "inside_own_actual_bbox": inside_own_bbox,
        }
    )

    shape = tuple(int(value) for value in details.centroid_grid_shape)
    grid_indices = np.column_stack(
        np.unravel_index(np.arange(len(centroid_points)), shape)
    )
    centroid_frame = pd.DataFrame(
        {
            "case": case.name,
            "centroid_index": np.arange(len(centroid_points), dtype=np.int64),
            "grid_i": grid_indices[:, 0],
            "grid_j": grid_indices[:, 1],
            "grid_k": grid_indices[:, 2],
            "x": centroid_points[:, 0],
            "y": centroid_points[:, 1],
            "z": centroid_points[:, 2],
            "automatic_domain": centroid_labels,
        }
    )

    return {
        "domains": _write(
            pd.DataFrame(summary_rows), CSV_DIR / f"{case.name}_domains.csv"
        ),
        "domain_points": _write(
            point_frame, CSV_DIR / f"{case.name}_domain_points.csv"
        ),
        "domain_centroids": _write(
            centroid_frame, CSV_DIR / f"{case.name}_domain_centroids.csv"
        ),
    }


def _lva_field_csv(case: Any, trend_vertices: np.ndarray, trend_faces: np.ndarray) -> Path:
    axes = [
        np.linspace(
            float(suite.MODEL_MIN[axis]),
            float(suite.MODEL_MAX[axis]),
            LVA_GRID_DIMENSION,
        )
        for axis in range(3)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack(
        [xx.ravel(order="C"), yy.ravel(order="C"), zz.ravel(order="C")]
    )
    indices = np.column_stack(
        np.unravel_index(
            np.arange(len(points)),
            (LVA_GRID_DIMENSION, LVA_GRID_DIMENSION, LVA_GRID_DIMENSION),
        )
    )
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        float(case.strength),
        float(case.trend_range),
    )
    lva = _lva_components(points, trend_input)
    matrix = lva["matrices"]
    normal = lva["normals"]
    eig = lva["eigenvalues"]
    frame = pd.DataFrame(
        {
            "case": case.name,
            "grid_i": indices[:, 0],
            "grid_j": indices[:, 1],
            "grid_k": indices[:, 2],
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "nearest_trend_vertex": lva["nearest"],
            "distance_to_trend_vertex": lva["distance"],
            "inside_4r_cutoff": lva["inside_cutoff"],
            "influence_q": lva["q"],
            "anisotropy_ratio": lva["ratio"],
            "normal_x": normal[:, 0],
            "normal_y": normal[:, 1],
            "normal_z": normal[:, 2],
            "glyph_major": lva["glyph_major"],
            "glyph_semi_major": lva["glyph_major"],
            "glyph_minor": lva["glyph_minor"],
            "eigenvalue_min": eig[:, 0],
            "eigenvalue_mid": eig[:, 1],
            "eigenvalue_max": eig[:, 2],
            "matrix_determinant": np.linalg.det(matrix),
            "matrix_00": matrix[:, 0, 0],
            "matrix_01": matrix[:, 0, 1],
            "matrix_02": matrix[:, 0, 2],
            "matrix_10": matrix[:, 1, 0],
            "matrix_11": matrix[:, 1, 1],
            "matrix_12": matrix[:, 1, 2],
            "matrix_20": matrix[:, 2, 0],
            "matrix_21": matrix[:, 2, 1],
            "matrix_22": matrix[:, 2, 2],
        }
    )
    return _write(frame, CSV_DIR / f"{case.name}_lva_field.csv")


_BASE_BUILD_CASE = suite.build_case


def _exporting_build_case(
    case: Any,
    points: np.ndarray,
    indicators: np.ndarray,
    trend_vertices: np.ndarray,
    trend_faces: np.ndarray,
) -> dict[str, Any]:
    result = _BASE_BUILD_CASE(
        case,
        points,
        indicators,
        trend_vertices,
        trend_faces,
    )
    paths = _domain_csvs(case, points)
    paths["lva_field"] = _lva_field_csv(case, trend_vertices, trend_faces)
    result["inspection_csv"] = {
        name: str(path.relative_to(suite.ROOT)) for name, path in paths.items()
    }
    print(
        f"[{case.name}] Wrote domain and LVA CSV files to {CSV_DIR}",
        flush=True,
    )
    return result


suite.build_case = _exporting_build_case

print(
    "PROGRESS\tExact Leapfrog LVA forced. Each generated case also exports domain "
    "summary/point/centroid CSVs and a regular-grid LVA field CSV "
    f"({LVA_GRID_DIMENSION} x {LVA_GRID_DIMENSION} x {LVA_GRID_DIMENSION}).",
    flush=True,
)

if __name__ == "__main__":
    exit_code = suite.main()
    diagnostic._write_cross_case_comparison()
    raise SystemExit(exit_code)
