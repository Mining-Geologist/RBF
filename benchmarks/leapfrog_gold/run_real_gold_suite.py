"""Run the real WolfPass/Leapfrog structural-LVA gold benchmark headlessly.

The data are downloaded from the repository release asset rather than committed to
source control. Every available S<strength>_R<range>.obj reference is regenerated
with the same production automatic SubDomainer, structural interpolant, and
chunk-safe isosurface path used by the standalone GUI.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

import polatory
from polatory import three as p3


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Importing the compatibility worker installs the exact production corrections:
# finite LVA-geodesic domains, smooth outside-field completion, and overlap-safe
# chunk extraction.
import polatory_lva_worker_process as production_worker  # noqa: E402

LVA_WORKER = production_worker.worker
SAFE_MESHER = LVA_WORKER.v2.v8.v5.v3

DATA_DIR = Path(
    os.environ.get(
        "LEAPFROG_GOLD_DIR",
        str(ROOT / "benchmark-data" / "leapfrog-benchmark-data-v1"),
    )
)
OUTPUT_DIR = ROOT / "benchmark-results" / "real-gold"
MESH_DIR = OUTPUT_DIR / "meshes"
PLOT_DIR = OUTPUT_DIR / "overlays"

MODEL_MIN = np.array([444600.0, 492600.0, 2000.0], dtype=float)
MODEL_MAX = np.array([445900.0, 494600.0, 3600.0], dtype=float)

BASE_RANGE = 400.0
TOTAL_SILL = 100.0
OUTSIDE_VALUE = -1.0
SURFACE_RESOLUTION = float(os.environ.get("POLATORY_BENCHMARK_RESOLUTION", "10"))
MAX_CASES = int(os.environ.get("POLATORY_BENCHMARK_MAX_CASES", "0"))

REFERENCE_PATTERN = re.compile(
    r"^S(?P<strength>\d+(?:\.\d+)?)_R(?P<range>\d+(?:\.\d+)?)\.obj$",
    re.I,
)


@dataclass(frozen=True)
class Case:
    name: str
    strength: float
    trend_range: float
    reference: Path


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(
                    [float(fields[1]), float(fields[2]), float(fields[3])]
                )
            elif line.startswith("f "):
                polygon = [
                    int(field.split("/", 1)[0]) - 1
                    for field in line.split()[1:]
                ]
                for index in range(1, len(polygon) - 1):
                    faces.append(
                        [polygon[0], polygon[index], polygon[index + 1]]
                    )
    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    if (
        vertex_array.ndim != 2
        or vertex_array.shape[1] != 3
        or len(vertex_array) == 0
    ):
        raise ValueError(f"No OBJ vertices found in {path}")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise ValueError(f"No OBJ triangles found in {path}")
    return vertex_array, face_array


def read_dataset() -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "Used Data(1).csv"
    frame = pd.read_csv(path)
    required = {"xe", "ye", "ze", "SDF"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    points = frame[["xe", "ye", "ze"]].to_numpy(dtype=float)
    indicators = frame["SDF"].to_numpy(dtype=float)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(indicators)):
        raise ValueError("Benchmark points and indicators must be finite.")
    categories = set(np.unique(indicators).tolist())
    if not categories.issubset({-1.0, 1.0}):
        raise ValueError(
            f"Expected only -1/+1 SDF indicators, found {sorted(categories)}"
        )
    return points, indicators


def available_cases() -> list[Case]:
    cases: list[Case] = []
    for path in sorted(DATA_DIR.glob("S*_R*.obj")):
        match = REFERENCE_PATTERN.match(path.name)
        if match is None:
            continue
        cases.append(
            Case(
                name=path.stem,
                strength=float(match.group("strength")),
                trend_range=float(match.group("range")),
                reference=path,
            )
        )
    cases.sort(key=lambda item: (item.strength, item.trend_range))
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]
    if not cases:
        raise FileNotFoundError(
            f"No S<strength>_R<range>.obj references found in {DATA_DIR}"
        )
    return cases


def polydata(path: Path) -> pv.PolyData:
    return pv.read(path).extract_surface().triangulate().clean()


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
    }


def point_to_surface_distance(
    source: pv.PolyData,
    target: pv.PolyData,
) -> np.ndarray:
    measured = source.copy(deep=True)
    measured.compute_implicit_distance(target, inplace=True)
    return np.abs(np.asarray(measured["implicit_distance"], dtype=float))


def suspicious_planes(
    vertices: np.ndarray,
) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {}
    threshold = max(20, int(math.ceil(0.01 * len(vertices))))
    for axis, name in enumerate(("x", "y", "z")):
        values, counts = np.unique(
            np.round(vertices[:, axis], 6),
            return_counts=True,
        )
        order = np.argsort(counts)[::-1]
        result[name] = [
            {"coordinate": float(values[index]), "vertices": int(counts[index])}
            for index in order[:10]
            if int(counts[index]) >= threshold
        ]
    return result


def connected_components(mesh: pv.PolyData) -> int:
    labelled = mesh.connectivity()
    if "RegionId" not in labelled.cell_data or labelled.n_cells == 0:
        return 0
    return int(np.max(np.asarray(labelled.cell_data["RegionId"]))) + 1


def mesh_metrics(mesh: pv.PolyData) -> dict[str, Any]:
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    return {
        "vertices": int(mesh.n_points),
        "triangles": int(mesh.n_cells),
        "area": float(mesh.area),
        "bounds_min": [
            float(mesh.bounds[0]),
            float(mesh.bounds[2]),
            float(mesh.bounds[4]),
        ],
        "bounds_max": [
            float(mesh.bounds[1]),
            float(mesh.bounds[3]),
            float(mesh.bounds[5]),
        ],
        "components": connected_components(mesh),
        "boundary_edge_cells": int(boundary.n_cells),
        "suspicious_coordinate_planes": suspicious_planes(
            np.asarray(mesh.points, dtype=float)
        ),
    }


def compare_meshes(
    generated_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    generated = polydata(generated_path)
    reference = polydata(reference_path)
    generated_to_reference = point_to_surface_distance(generated, reference)
    reference_to_generated = point_to_surface_distance(reference, generated)
    symmetric = np.concatenate(
        [generated_to_reference, reference_to_generated]
    )
    return {
        "generated": mesh_metrics(generated),
        "reference": mesh_metrics(reference),
        "generated_to_reference": distribution(generated_to_reference),
        "reference_to_generated": distribution(reference_to_generated),
        "symmetric_surface_distance": distribution(symmetric),
        "area_ratio_generated_over_reference": float(
            generated.area / reference.area
        ),
    }


def render_overlay(case: Case, generated_path: Path) -> None:
    generated, _ = read_obj(generated_path)
    reference, _ = read_obj(case.reference)
    limits = [
        (MODEL_MIN[0], MODEL_MAX[0]),
        (MODEL_MIN[1], MODEL_MAX[1]),
        (MODEL_MIN[2], MODEL_MAX[2]),
    ]
    projections = (
        ("XY plan", 0, 1),
        ("XZ section projection", 0, 2),
        ("YZ section projection", 1, 2),
    )
    labels = ("X", "Y", "Z")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (title, first, second) in zip(axes, projections):
        stride_reference = max(1, len(reference) // 80_000)
        stride_generated = max(1, len(generated) // 80_000)
        axis.scatter(
            reference[::stride_reference, first],
            reference[::stride_reference, second],
            s=0.35,
            alpha=0.35,
            label="Leapfrog",
        )
        axis.scatter(
            generated[::stride_generated, first],
            generated[::stride_generated, second],
            s=0.35,
            alpha=0.35,
            label="Polatory",
        )
        axis.set_title(title)
        axis.set_xlabel(labels[first])
        axis.set_ylabel(labels[second])
        axis.set_xlim(limits[first])
        axis.set_ylim(limits[second])
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(markerscale=10)
    figure.suptitle(
        f"{case.name}: Polatory vs Leapfrog, "
        f"resolution {SURFACE_RESOLUTION:g} m"
    )
    figure.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_DIR / f"{case.name}_overlay.png", dpi=180)
    plt.close(figure)


def build_case(
    case: Case,
    points: np.ndarray,
    indicators: np.ndarray,
    trend_vertices: np.ndarray,
    trend_faces: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    print(
        f"Running {case.name}: strength={case.strength:g}, "
        f"trend range={case.trend_range:g}",
        flush=True,
    )

    value_info = polatory.leapfrog_indicator_values3(
        points,
        indicators,
        fit_accuracy=0.0,
    )
    values = np.asarray(value_info.values, dtype=float)

    trend_input = polatory.StructuralTrendInput3(
        trend_vertices,
        trend_faces,
        float(case.strength),
        float(case.trend_range),
    )
    model = p3.Model(
        p3.CovSpheroidal3([TOTAL_SILL, BASE_RANGE]),
        0,
    )
    model.nugget = 0.0
    model_parameters = (
        np.asarray(model.parameters, dtype=float).reshape(-1).tolist()
    )

    builder = LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder(
        centroid_count=6000,
        minimum_cluster_fraction=0.001,
        maximum_cluster_fraction=0.10,
        consistency_threshold=0.60,
        base_range=BASE_RANGE,
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
        OUTSIDE_VALUE,
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

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MESH_DIR / f"{case.name}_polatory.obj"

    # Allow the complete documented model extent while retaining small native
    # calls. The search cycle defaults to a 10 m surface; final candidates are
    # re-run at 5 m through the workflow input.
    SAFE_MESHER.MAX_TOTAL_BASE_CELLS = 50_000_000
    SAFE_MESHER.MAX_CHUNKS = 256
    SAFE_MESHER.generate_safe_isosurface(
        structural=structural,
        bbox_min=MODEL_MIN,
        bbox_max=MODEL_MAX,
        resolution=SURFACE_RESOLUTION,
        refine=0,
        output_obj=output_path,
        progress=lambda message: print(
            f"[{case.name}] {message}",
            flush=True,
        ),
    )

    comparison = compare_meshes(output_path, case.reference)
    render_overlay(case, output_path)

    elapsed = time.perf_counter() - started
    return {
        "case": case.name,
        "strength": case.strength,
        "trend_range": case.trend_range,
        "base_range": BASE_RANGE,
        "total_sill": TOTAL_SILL,
        "outside_value": OUTSIDE_VALUE,
        "surface_resolution": SURFACE_RESOLUTION,
        "input_points": int(len(points)),
        "domain_count": int(len(domains)),
        "centroid_grid_shape": (
            list(diagnostics.centroid_grid_shape)
            if diagnostics is not None
            else None
        ),
        "training_rmse": float(np.sqrt(np.mean(errors**2))),
        "training_max_abs": float(np.max(np.abs(errors))),
        "elapsed_seconds": float(elapsed),
        "generated_obj": str(output_path.relative_to(ROOT)),
        "reference_obj": case.reference.name,
        "comparison": comparison,
    }


def response_metrics(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize whether range changes move broad bounds in the same direction."""
    output: list[dict[str, Any]] = []
    grouped: dict[float, list[dict[str, Any]]] = {}
    for item in results.values():
        grouped.setdefault(float(item["strength"]), []).append(item)
    for strength, group in sorted(grouped.items()):
        group.sort(key=lambda item: float(item["trend_range"]))
        for lower, upper in zip(group, group[1:]):
            lower_comp = lower["comparison"]
            upper_comp = upper["comparison"]
            generated_delta_min = (
                np.asarray(upper_comp["generated"]["bounds_min"])
                - np.asarray(lower_comp["generated"]["bounds_min"])
            )
            generated_delta_max = (
                np.asarray(upper_comp["generated"]["bounds_max"])
                - np.asarray(lower_comp["generated"]["bounds_max"])
            )
            reference_delta_min = (
                np.asarray(upper_comp["reference"]["bounds_min"])
                - np.asarray(lower_comp["reference"]["bounds_min"])
            )
            reference_delta_max = (
                np.asarray(upper_comp["reference"]["bounds_max"])
                - np.asarray(lower_comp["reference"]["bounds_max"])
            )
            generated_delta = np.concatenate(
                [generated_delta_min, generated_delta_max]
            )
            reference_delta = np.concatenate(
                [reference_delta_min, reference_delta_max]
            )
            denominator = float(
                np.linalg.norm(generated_delta)
                * np.linalg.norm(reference_delta)
            )
            cosine = (
                float(np.dot(generated_delta, reference_delta) / denominator)
                if denominator > 0.0
                else None
            )
            output.append(
                {
                    "strength": strength,
                    "from_range": float(lower["trend_range"]),
                    "to_range": float(upper["trend_range"]),
                    "bounds_change_cosine_similarity": cosine,
                    "generated_bounds_delta": generated_delta.tolist(),
                    "reference_bounds_delta": reference_delta.tolist(),
                }
            )
    return output


def write_summary(report: dict[str, Any]) -> None:
    lines = [
        "# Real Leapfrog LVA benchmark",
        "",
        f"- Surface resolution: {SURFACE_RESOLUTION:g} m",
        f"- Cases: {len(report['cases'])}",
        "",
        "| Case | Domains | Symmetric mean | Symmetric p95 | Area ratio | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["cases"].items():
        distance = item["comparison"]["symmetric_surface_distance"]
        lines.append(
            f"| {name} | {item['domain_count']} | {distance['mean']:.3f} | "
            f"{distance['p95']:.3f} | "
            f"{item['comparison']['area_ratio_generated_over_reference']:.3f} | "
            f"{item['elapsed_seconds']:.1f} |"
        )
    (OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Benchmark data directory does not exist: {DATA_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    points, indicators = read_dataset()
    trend_vertices, trend_faces = read_obj(DATA_DIR / "Reference Mesh(2).obj")
    cases = available_cases()

    report: dict[str, Any] = {
        "data_directory": str(DATA_DIR),
        "model_bounds_min": MODEL_MIN.tolist(),
        "model_bounds_max": MODEL_MAX.tolist(),
        "surface_resolution": SURFACE_RESOLUTION,
        "cases": {},
    }

    for case in cases:
        report["cases"][case.name] = build_case(
            case,
            points,
            indicators,
            trend_vertices,
            trend_faces,
        )
        # Persist after every case so a timeout still leaves useful partial artifacts.
        (OUTPUT_DIR / "metrics.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    report["parameter_response"] = response_metrics(report["cases"])
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_summary(report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
