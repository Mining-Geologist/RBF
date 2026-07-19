"""Run the synthetic Leapfrog structural-LVA benchmarks without the GUI.

This script intentionally uses the same automatic finite LVA-geodesic builder and
native structural interpolant used by the process-isolated application.  It writes
OBJ meshes and a JSON report suitable for GitHub Actions artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import polatory
from polatory import three as p3


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

# The builder module currently lives beside the GUI launchers.  Importing it here
# applies no GUI actions; it only exposes the exact finite geodesic domain builder.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import polatory_lva_worker_process_v3 as lva_worker  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    csv_name: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    base_range: float


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    indicators: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"X", "Y", "Z", "Category"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            category = str(row["Category"]).strip().lower()
            if category == "inside":
                indicator = -1.0
            elif category == "outside":
                indicator = 1.0
            else:
                raise ValueError(f"Unsupported category {row['Category']!r} in {path}")
            points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
            indicators.append(indicator)
    return np.asarray(points, dtype=float), np.asarray(indicators, dtype=float)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                fields = line.split()[1:]
                polygon = [int(field.split("/", 1)[0]) - 1 for field in fields]
                if len(polygon) < 3:
                    continue
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])
    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or len(vertex_array) == 0:
        raise ValueError(f"No OBJ vertices found in {path}")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise ValueError(f"No OBJ triangles found in {path}")
    return vertex_array, face_array


def mesh_metrics(path: Path) -> dict[str, Any]:
    vertices, faces = read_obj(path)
    suspicious: dict[str, list[dict[str, float | int]]] = {}
    for axis, name in enumerate(("x", "y", "z")):
        rounded = np.round(vertices[:, axis], decimals=6)
        values, counts = np.unique(rounded, return_counts=True)
        threshold = max(20, int(np.ceil(0.01 * len(vertices))))
        order = np.argsort(counts)[::-1]
        suspicious[name] = [
            {"coordinate": float(values[index]), "vertices": int(counts[index])}
            for index in order[:10]
            if int(counts[index]) >= threshold
        ]
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
        "suspicious_coordinate_planes": suspicious,
    }


def symmetric_vertex_distance(first_path: Path, second_path: Path) -> dict[str, float]:
    from scipy.spatial import cKDTree

    first, _ = read_obj(first_path)
    second, _ = read_obj(second_path)
    first_to_second = np.asarray(cKDTree(second).query(first, k=1)[0], dtype=float)
    second_to_first = np.asarray(cKDTree(first).query(second, k=1)[0], dtype=float)
    combined = np.concatenate([first_to_second, second_to_first])
    return {
        "mean": float(np.mean(combined)),
        "median": float(np.median(combined)),
        "p95": float(np.percentile(combined, 95.0)),
        "p99": float(np.percentile(combined, 99.0)),
        "maximum": float(np.max(combined)),
    }


def run_case(case: Case, trend_vertices: np.ndarray, trend_faces: np.ndarray, output: Path) -> dict[str, Any]:
    data_path = Path(__file__).resolve().parent / case.csv_name
    points, indicators = read_points(data_path)
    bbox_min = np.asarray(case.bbox_min, dtype=float)
    bbox_max = np.asarray(case.bbox_max, dtype=float)

    value_info = polatory.leapfrog_indicator_values3(
        points,
        indicators,
        fit_accuracy=0.0,
    )
    values = np.asarray(value_info.values, dtype=float)

    trend_input = polatory.StructuralTrendInput3(
        trend_vertices,
        trend_faces,
        5.0,
        100.0,
    )
    rbf = p3.CovSpheroidal3([10.0, float(case.base_range)])
    model = p3.Model(rbf, 0)
    model.nugget = 0.0
    model_parameters = np.asarray(model.parameters, dtype=float).reshape(-1).tolist()

    # Use the same finite LVA-geodesic automatic SubDomainer as the current app.
    builder = lva_worker.FiniteLvaGeodesicAutomaticBuilder(
        centroid_count=6000,
        minimum_cluster_fraction=0.001,
        maximum_cluster_fraction=0.10,
        consistency_threshold=0.60,
        base_range=float(case.base_range),
        support_multiplier=5,
        minimum_support_points=1,
    )
    domains = builder.build_from_inputs(
        points,
        [trend_input],
        model_parameters=model_parameters,
        trend_type=polatory.StructuralTrendType.STRONGEST_ALONG_INPUTS,
    )
    diagnostics = builder.diagnostics_

    structural = polatory.StructuralInterpolant3(
        model,
        -1.0,
        1.0,
        0.0,
        True,
    )
    structural.fit(
        points,
        values,
        domains,
        tolerance=float(value_info.fit_accuracy),
        max_iter=100,
    )
    predictions = np.asarray(structural.evaluate(points), dtype=float)
    errors = predictions - values

    field = polatory.StructuralRbfFieldFunction(structural)
    bbox = p3.Bbox(bbox_min.reshape(1, 3), bbox_max.reshape(1, 3))
    result = polatory.Isosurface(bbox, 5.0, np.eye(3)).generate(
        field,
        isovalue=0.0,
        refine=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.export_obj(str(output))

    metrics = mesh_metrics(output)
    metrics.update(
        {
            "case": case.name,
            "input_points": int(len(points)),
            "domain_count": int(len(domains)),
            "training_rmse": float(np.sqrt(np.mean(errors**2))),
            "training_max_abs": float(np.max(np.abs(errors))),
            "parameters": {
                "strength": 5.0,
                "trend_range": 100.0,
                "base_range": float(case.base_range),
                "sill": 10.0,
                "blend_power": 1.0,
                "background_blending": True,
                "centroid_count": 6000,
                "minimum_cluster_fraction": 0.001,
                "maximum_cluster_fraction": 0.10,
                "consistency_threshold": 0.60,
                "support_multiplier": 5,
                "minimum_support_points": 1,
                "surface_resolution": 5.0,
            },
            "centroid_grid_shape": (
                list(diagnostics.centroid_grid_shape)
                if diagnostics is not None
                else None
            ),
        }
    )
    return metrics


def main() -> int:
    benchmark_dir = Path(__file__).resolve().parent
    output_dir = ROOT / "benchmark-results"
    output_dir.mkdir(parents=True, exist_ok=True)

    trend_vertices, trend_faces = read_obj(benchmark_dir / "folded_trend_mesh.obj")
    cases = (
        Case(
            "test1_tight_s5_r100",
            "01_two_domain_blend_binary.csv",
            (-300.0, -150.0, -150.0),
            (300.0, 150.0, 350.0),
            80.0,
        ),
        Case(
            "test1_deep_s5_r100",
            "01_two_domain_blend_binary.csv",
            (-300.0, -150.0, -500.0),
            (300.0, 150.0, 350.0),
            80.0,
        ),
        Case(
            "test2_background_s5_r100",
            "02_background_decay_binary.csv",
            (-300.0, -150.0, -500.0),
            (300.0, 150.0, 350.0),
            70.0,
        ),
    )

    report: dict[str, Any] = {"cases": {}}
    output_paths: dict[str, Path] = {}
    for case in cases:
        print(f"Running {case.name}...", flush=True)
        output_path = output_dir / f"{case.name}.obj"
        report["cases"][case.name] = run_case(
            case,
            trend_vertices,
            trend_faces,
            output_path,
        )
        output_paths[case.name] = output_path

    report["extent_invariance"] = symmetric_vertex_distance(
        output_paths["test1_tight_s5_r100"],
        output_paths["test1_deep_s5_r100"],
    )
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
